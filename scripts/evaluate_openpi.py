"""Evaluate verified OpenPI or OpenVLA-OFT services through the LIBERO runner."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from embodiedlab.envs.libero_env import LiberoEnv, SUITE_STEPS
from embodiedlab.evaluation import evaluate_task, save_preview, task_diagnostics
from embodiedlab.benchmark import summarize_predicates
from embodiedlab.policies.openpi import OpenPIPolicy

ROOT = Path(__file__).resolve().parents[1]
OPENPI_STEPS = {'libero_spatial': 220, 'libero_object': 280,
                'libero_goal': 300, 'libero_10': 520, 'libero_90': 400}


def json_safe(value):
    """Keep invalid raw model diagnostics printable without nonstandard JSON NaN."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # A diagnostic string, never a replacement action.
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    """Save one immutable report; diagnostic NaN/Inf are explicit strings."""
    with path.open('x', encoding='utf-8') as stream:
        json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write('\n')


def latency_summary(values: list[float]) -> dict | None:
    """Describe actual measured durations, without fabricating an empty sample."""
    if not values:
        return None
    return {'count': len(values), 'first_ms': float(values[0]),
            'mean_ms': float(np.mean(values)), 'p50_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)), 'max_ms': float(max(values))}


def _challenge_match(field: str, observed, expected) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise ValueError(f'Frozen challenge mismatch: {field}; expected {expected!r}, observed {observed!r}')


def validate_challenge_environment(env: LiberoEnv, task: dict, environment: dict) -> None:
    """Check actual selected assets/states/runtime before the first native reset."""
    from libero.libero import get_libero_path

    for field in ('suite', 'task_id', 'task_name', 'language', 'max_episode_steps'):
        _challenge_match(field, getattr(env, field), task[field])
    _challenge_match('native task_name', env._task.name, task['task_name'])
    _challenge_match('native language', env._task.language, task['language'])
    assets = [('bddl_files', 'bddl_file', 'bddl_relative_path', 'bddl_sha256', 'bddl_files'),
              ('init_states', 'init_states_file', 'init_file_relative_path', 'init_file_sha256', 'init_files')]
    for root_key, native_key, relative_key, hash_key, prefix in assets:
        relative = Path(env._task.problem_folder) / getattr(env._task, native_key)
        _challenge_match(relative_key, str(Path(prefix) / relative), task[relative_key])
        actual_path = Path(get_libero_path(root_key)) / relative
        try:
            digest = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f'Frozen challenge mismatch: {hash_key}; cannot read actual asset {actual_path}: {error}') from error
        _challenge_match(f'{hash_key} ({actual_path})', digest, task[hash_key])
    states = np.asarray(env._initial_states)
    _challenge_match('initial_states_shape', list(states.shape), task['initial_states_shape'])
    hashes = [hashlib.sha256(state.tobytes(order='C')).hexdigest() for state in states[:10]]
    _challenge_match('first_10_state_sha256', hashes, task['first_10_state_sha256'])
    metadata = env.episode_metadata()
    for actual_key, frozen_key in [('libero_commit', 'libero_commit'),
                                    ('libero_version', 'libero_version'),
                                    ('assets_revision', 'assets_revision'),
                                    ('mujoco_version', 'mujoco'), ('robosuite_version', 'robosuite'),
                                    ('control_dt', 'control_dt'), ('settle_steps', 'settle_steps')]:
        _challenge_match(actual_key, metadata.get(actual_key), environment[frozen_key])


def validate_server_manifest(manifest: dict, handshake: dict, frozen_model: dict | None = None) -> None:
    """Reject stale manifests and servers with another model or launch identity."""
    pin = json.loads((ROOT / 'openpi-upstream.json').read_text())
    checkpoint = pin['checkpoint']
    expected = {'openpi_commit': pin['source']['commit'],
                'config': checkpoint['config'], 'checkpoint': checkpoint['uri'],
                'checkpoint_sha256': checkpoint['content_sha256'],
                'backend': 'jax', 'action_horizon': 10,
                'norm_stats_sha256': next(item['sha256'] for item in checkpoint['files']
                                          if item['path'].endswith('/norm_stats.json'))}
    if not isinstance(manifest, dict) or not isinstance(handshake, dict):
        raise ValueError('OpenPI manifest and handshake must be JSON objects')
    for key, value in expected.items():
        if manifest.get(key) != value or handshake.get(key) != value:
            raise ValueError(f'OpenPI verified model identity mismatch: {key}')
    for key in ('run_id', 'gpu'):
        if not manifest.get(key) or handshake.get(key) != manifest[key]:
            raise ValueError(f'OpenPI launch identity mismatch: {key}')
    if frozen_model is not None:
        _challenge_match('openpi.model_seed', frozen_model['model_seed'], 0)
        rng = 'official key(0), advances per request; reset does not reset server'
        for label, source in [('manifest', manifest), ('handshake', handshake)]:
            _challenge_match(f'openpi.{label}.server_rng', source.get('server_rng'), rng)


class StepClock:
    """Only time native step calls; delegate the unchanged public environment API."""

    def __init__(self, env: LiberoEnv) -> None:
        self.env = env
        self.step_times_ms: list[float] = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **options):
        self.step_times_ms.clear()
        return self.env.reset(**options)

    def step(self, action):
        started = perf_counter()
        result = self.env.step(action)
        self.step_times_ms.append((perf_counter() - started) * 1000.)
        return result


def validate_openvla_manifest(manifest: dict, handshake: dict, frozen_model: dict | None = None) -> None:
    """Verify the fixed OFT checkpoint and this particular server launch."""
    pin = json.loads((ROOT / 'openvla-upstream.json').read_text())
    expected = {'model': 'openvla_oft', 'source_commit': pin['source']['commit'],
                'transformers_commit': pin['transformers']['commit'],
                'checkpoint': pin['checkpoint']['repo_id'],
                'revision': pin['checkpoint']['revision'],
                'checkpoint_sha256': pin['checkpoint']['content_sha256'],
                'backend': 'torch', 'action_horizon': 8,
                'seed': pin['protocol']['model_seed'], 'dtype': pin['protocol']['dtype'],
                'quantization': pin['protocol']['quantization'], 'attention': 'sdpa'}
    if not isinstance(manifest, dict) or not isinstance(handshake, dict):
        raise ValueError('OpenVLA-OFT manifest and handshake must be JSON objects')
    for key, value in expected.items():
        if (type(manifest.get(key)) is not type(value) or manifest.get(key) != value
                or type(handshake.get(key)) is not type(value) or handshake.get(key) != value):
            raise ValueError(f'OpenVLA-OFT verified model identity mismatch: {key}')
    for key in ('run_id', 'gpu', 'seed'):
        if key not in manifest or handshake.get(key) != manifest[key]:
            raise ValueError(f'OpenVLA-OFT launch identity mismatch: {key}')
    if frozen_model is not None:
        _challenge_match('openvla_oft.model_seed', frozen_model['model_seed'], expected['seed'])
        runtime = frozen_model['runtime_required']
        for field in ('dtype', 'quantization'):
            _challenge_match(f'openvla_oft.runtime_required.{field}', runtime[field], expected[field])
        _challenge_match('openvla_oft.runtime_required.attention', runtime['attention'],
                         'official fork SDPA bidirectional')


def main(model_name: str = 'openpi') -> None:
    # Two actual integrations share only rollout/report orchestration. Each
    # policy owns its model-specific input, action and chunk contract.
    if model_name not in ('openpi', 'openvla_oft'):
        raise ValueError('Supported VLA models: openpi, openvla_oft')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=tuple(SUITE_STEPS), default='libero_goal')
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--init-state-indices', type=int, nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8006 if model_name == 'openpi' else 8007)
    parser.add_argument('--native-action-bounds', action='store_true',
                        help='Explicit native-equivalent input saturation; retain every raw action and projection')
    parser.add_argument('--replan-steps', type=int, default=5 if model_name == 'openpi' else 8)
    parser.add_argument('--benchmark-manifest', type=Path)
    parser.add_argument('--benchmark-phase', choices=('smoke', 'formal'), default='formal')
    parser.add_argument('--connect-timeout', type=float, default=10.)
    parser.add_argument('--request-timeout', type=float, default=120.)
    parser.add_argument('--server-manifest', type=Path, required=True,
                        help='Verified launch-side checkpoint/backend/GPU/version manifest')
    parser.add_argument('--evaluation-profile', choices=('embodiedlab', 'openpi_horizon', 'model_reference'),
                        default='embodiedlab' if model_name == 'openpi' else 'model_reference',
                        help='Only selects the documented horizon; native dynamics are unchanged')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'evaluation_results' / model_name)
    args = parser.parse_args()
    indices = args.init_state_indices
    if (args.seed < 0 or len(set(indices)) != len(indices)
            or any(index < 0 for index in indices)):
        parser.error('Seed and initial-state indices must be nonnegative; indices must be unique')
    if (not 1 <= args.port <= 65535 or args.replan_steps < 1
            or args.connect_timeout <= 0 or args.request_timeout <= 0):
        parser.error('Port and positive replanning/timeouts must be valid')
    server_manifest = json.loads(args.server_manifest.read_text())
    if not isinstance(server_manifest, dict) or not server_manifest:
        parser.error('Server manifest must be a nonempty JSON object with verified provenance')

    output = args.output_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                               + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    horizon = (OPENPI_STEPS if args.evaluation_profile in ('openpi_horizon', 'model_reference')
               else SUITE_STEPS)[args.suite]
    challenge = None
    challenge_environment = challenge_model = None
    if args.benchmark_manifest is not None:
        challenge = json.loads(args.benchmark_manifest.read_text())
        task = next((task for task in challenge['tasks']
                     if (task['suite'], task['task_id']) == (args.suite, args.task_id)), None)
        if task is None:
            raise ValueError('Task is not in the frozen challenge manifest')
        planned = task[args.benchmark_phase]
        if indices != planned['initial_state_indices'] or args.seed != planned['seeds'][0]:
            raise ValueError('Initial states / seed differ from the frozen challenge manifest')
        if planned['seeds'] != list(range(args.seed, args.seed + len(indices))):
            raise ValueError('Manifest seeds must match the evaluator ordinal seed rule')
        expected_replan = challenge['models'][model_name]['replan_steps']
        if args.replan_steps != expected_replan:
            raise ValueError('Action replanning differs from frozen model protocol')
        challenge_environment = challenge['environment']
        challenge_model = challenge['models'][model_name]
        horizon = task['max_episode_steps']
        challenge = {'sha256': hashlib.sha256(args.benchmark_manifest.read_bytes()).hexdigest(),
                     'version': challenge['manifest_version'], 'phase': args.benchmark_phase,
                     'task': task}
    episodes = []
    all_rpc_ms, all_act_ms, all_step_ms = [], [], []
    policy = None
    fatal_error = None
    try:
        if model_name == 'openpi':
            policy = OpenPIPolicy(host=args.host, port=args.port,
                              replan_steps=args.replan_steps, native_action_bounds=args.native_action_bounds,
                              connect_timeout=args.connect_timeout,
                              request_timeout=args.request_timeout)
            validate_server_manifest(server_manifest, policy.server_metadata, challenge_model)
        else:
            from embodiedlab.policies.openvla_oft import OpenVLAOFTPolicy
            if args.native_action_bounds or args.replan_steps != 8:
                raise ValueError('OFT uses official action processing and fixed eight-step chunks')
            policy = OpenVLAOFTPolicy(suite=args.suite, host=args.host, port=args.port,
                                     connect_timeout=args.connect_timeout,
                                     request_timeout=args.request_timeout)
            validate_openvla_manifest(server_manifest, policy.server_metadata, challenge_model)
        with LiberoEnv(suite=args.suite, task_id=args.task_id, image_size=256,
                       max_episode_steps=horizon, settle_steps=10) as native:
            if challenge is not None:
                validate_challenge_environment(native, challenge['task'], challenge_environment)
            if any(index >= native.initial_state_count for index in indices):
                raise ValueError('An initial-state index is outside the official state list')
            env = StepClock(native)
            policy.assert_compatible(env)
            protocol = {
                **native.episode_metadata(), 'policy': model_name,
                'benchmark': challenge,
                'evaluation_profile_selected': args.evaluation_profile,
                'initial_state_indices': indices,
                'seeds': list(range(args.seed, args.seed + len(indices))),
                'render_size': 256, 'model_image_size': 224,
                'image_transform': ('OpenPI: rotate180/PIL pad resize224' if model_name == 'openpi'
                                    else 'OFT server: rotate180/TF JPEG Lanczos224/90% area center crop/official processor'),
                'replan_steps': args.replan_steps, 'action_horizon': 10 if model_name == 'openpi' else 8,
                'native_action_bounds': args.native_action_bounds,
                'bounds_evidence': ('84-step raw vs projected native replay: full state/RGB/reward/flags exactly equal; no historical state injection'
                                    if model_name == 'openpi' else None),
                'server': {'host': args.host, 'port': args.port,
                           'manifest': server_manifest,
                           'handshake_metadata': policy.server_metadata},
                'server_rng': {'reset_at_episode_boundary': False,
                               'note': ('OpenPI key(0) advances per request; launch/order recorded' if model_name == 'openpi'
                                        else 'OFT L1 deterministic eval/do_sample=False; launch seed/request order recorded')},
                'timing': {'rpc': 'wall-clock request/response including conversion and transfer',
                           'act': 'adapter wall clock, including RPC only when replanning',
                           'step': 'env.step wall clock, excluding recording/observer',
                           'gpu': 'use launch-side synchronized GPU measurements; RPC is not GPU-only'},
                'error_policy': 'Stop on first exception; incomplete episodes have null success and are excluded',
                'contact_sampling': 'post-control-step endpoint, not substep peaks',
                'preview': 'display rotated 180 degrees; HDF5 pixels remain raw',
            }
            write_json(output / 'protocol.json', protocol)

            for ordinal, state_index in enumerate(indices):
                rows = []
                episode_started = perf_counter()
                record_dir = output / 'records' / f'episode_{ordinal:03d}'
                row = {'episode_index': ordinal, 'init_state_index': state_index,
                       'seed': args.seed + ordinal, 'status': 'incomplete', 'success': None,
                       'record_path': str((record_dir / 'episode_000.hdf5').resolve())}

                def observe(observation, action, info):
                    rows.append({'step': info['steps'],
                                 'simulation_time': info['simulation_time'],
                                 'native_success': info['success'],
                                 'action': action.tolist(),
                                 'robot_state': observation['robot_state'].tolist(),
                                 'joint_positions': observation['joint_positions'].tolist(),
                                 'act_ms': policy.act_times_ms[-1],
                                 'env_step_ms': env.step_times_ms[-1],
                                 **task_diagnostics(native)})
                    if info['steps'] == 1 or info['steps'] % 100 == 0 or info['success']:
                        print(f'init={state_index} step={info["steps"]} success={info["success"]}',
                              flush=True)

                try:
                    # Runner resets policy after env.reset. Clear logs here too so
                    # a failed native reset cannot reuse the preceding episode's
                    # diagnostics. Official client.reset does not reset server RNG.
                    policy.reset()
                    result = evaluate_task(env, policy=policy, policy_name=model_name,
                                           episodes=1, seed=args.seed + ordinal,
                                           initial_state_indices=[state_index],
                                           record_dir=record_dir, on_step=observe)
                    row.update(result['episodes'][0])
                    row.update(episode_index=ordinal, status='complete')
                    row['outcome'] = 'success' if row['success'] else 'timeout'
                    row['native_predicate_progress'] = (summarize_predicates(rows)
                        if rows and 'native_goal_predicates' in rows[0] else [])
                    preview = output / f'episode_{ordinal:03d}_preview.png'
                    # Preview failure is an artifact error, not a policy/task failure.
                    try:
                        save_preview(Path(row['record_path']), preview)
                        row['preview'] = str(preview.resolve())
                    except Exception as error:
                        row['preview_error'] = {'type': type(error).__name__, 'message': str(error)}
                except Exception as error:
                    row['error'] = {'type': type(error).__name__, 'message': str(error)}
                    row['completed_steps'] = len(rows)
                    fatal_error = row['error']
                finally:
                    rpc_ms = [request['rpc_ms'] for request in policy.request_log
                              if request.get('rpc_ms') is not None]
                    row.update(wall_seconds=perf_counter() - episode_started,
                               inference_requests=len(policy.request_log),
                               native_projected_action_count=policy.projected_action_count,
                               native_projected_dimension_counts=list(policy.projected_dimension_counts),
                               max_action_overshoot=policy.max_action_overshoot,
                               synchronized_sample_latency=latency_summary([r['synchronized_sample_ms'] for r in policy.request_log if r.get('synchronized_sample_ms') is not None]),
                               rpc_latency=latency_summary(rpc_ms),
                               act_latency=latency_summary(policy.act_times_ms),
                               environment_step_latency=latency_summary(env.step_times_ms))
                    all_rpc_ms.extend(rpc_ms)
                    all_act_ms.extend(policy.act_times_ms)
                    all_step_ms.extend(env.step_times_ms)
                    write_json(output / f'episode_{ordinal:03d}_diagnostics.json',
                               {'steps': rows, 'requests': policy.request_log, 'native_action_projection': policy.projection_log})
                    write_json(output / f'episode_{ordinal:03d}.json', row)
                    episodes.append(row)
                if fatal_error is not None:
                    break
    except Exception as error:
        fatal_error = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if policy is not None:
            try:
                policy.close()
            except Exception as error:
                fatal_error = fatal_error or {'type': type(error).__name__, 'message': str(error)}
        completed = [row for row in episodes if row['status'] == 'complete']
        report = {
            'status': 'complete' if fatal_error is None and len(completed) == len(indices) else 'incomplete',
            'suite': args.suite, 'task_id': args.task_id,
            'requested': len(indices), 'attempted': len(episodes), 'completed': len(completed),
            'incomplete': len(episodes) - len(completed),
            'unattempted': len(indices) - len(episodes),
            'success_rate': float(np.mean([row['success'] for row in completed])) if completed else None,
            'success_rate_denominator': len(completed), 'episodes': episodes,
            'rpc_latency': latency_summary(all_rpc_ms), 'act_latency': latency_summary(all_act_ms),
            'environment_step_latency': latency_summary(all_step_ms),
            'total_wall_seconds': perf_counter() - started, 'error': fatal_error,
        }
        write_json(output / 'results.json', report)
        print(f'{report["status"]}: complete={len(completed)}/{len(indices)}, '
              f'success_rate={report["success_rate"]}, reports={output}', flush=True)
    if fatal_error is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
