"""Evaluate single-task BC with native LIBERO suite settings and official states."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch

from embodiedlab.envs.libero_env import LiberoEnv
from embodiedlab.evaluation import evaluate_task, save_preview, task_diagnostics
from embodiedlab.interfaces import Action, Info, Observation
from embodiedlab.policies.bc import BCPolicy
from embodiedlab.policies.simple import ZeroPolicy

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """Fingerprint a checkpoint without changing it."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Create a finite JSON report exclusively, preserving previous experiments."""
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def resource_usage(device: str) -> dict:
    """Current RSS/file descriptors and Torch memory (not all native EGL VRAM)."""
    status = Path('/proc/self/status').read_text().splitlines()
    rss_kib = next(float(line.split()[1]) for line in status if line.startswith('VmRSS:'))
    result = {'rss_mib': rss_kib / 1024., 'open_file_descriptors': len(list(Path('/proc/self/fd').iterdir()))}
    if device == 'cuda':
        result.update(cuda_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                      cuda_reserved_mib=torch.cuda.memory_reserved() / 2**20)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'training_results/phase5b3_train_seed42/best.pt')
    parser.add_argument('--policy', choices=('bc', 'zero'), default='bc')
    parser.add_argument('--suite', default=None, help='Default: checkpoint task; required for zero policy')
    parser.add_argument('--task-id', type=int, default=None)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--episodes', type=int, default=None, help='Default: explicit index count, otherwise 5')
    states = parser.add_mutually_exclusive_group()
    states.add_argument('--init-state-indices', type=int, nargs='+')
    states.add_argument('--init-state-rule', choices=('first', 'random'), default='first')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'evaluation_results/bc')
    args = parser.parse_args()
    if args.threads < 1 or args.seed < 0 or (args.episodes is not None and args.episodes < 1):
        parser.error('threads/episodes must be positive; seed must be nonnegative')
    if (args.suite is None) != (args.task_id is None):
        parser.error('Specify --suite and --task-id together')
    if args.policy == 'zero' and args.suite is None:
        parser.error('Zero policy requires an explicit --suite and --task-id')
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = perf_counter()
    policy = BCPolicy(args.checkpoint, device=args.device) if args.policy == 'bc' else ZeroPolicy()
    device = str(policy.device) if isinstance(policy, BCPolicy) else 'cpu'
    suite = args.suite if args.suite is not None else policy.suite
    task_id = args.task_id if args.task_id is not None else policy.task_id
    image_size = policy.image_size if isinstance(policy, BCPolicy) else 128
    count = args.episodes or (len(args.init_state_indices) if args.init_state_indices is not None else 5)
    output = args.output_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    traces: dict[int, list[dict]] = {}
    resources = [{'stage': 'before_environment', **resource_usage(device)}]
    fingerprint = sha256(args.checkpoint) if isinstance(policy, BCPolicy) else None
    print(f'Policy={args.policy}, device={device}, task={suite}/{task_id}', flush=True)
    with LiberoEnv(suite=suite, task_id=task_id, image_size=image_size) as env:
        if isinstance(policy, BCPolicy):
            policy.assert_compatible(env)
        if count > env.initial_state_count:
            parser.error('Requested more episodes than official initial states')
        if args.init_state_indices is not None:
            indices, rule = args.init_state_indices, 'explicit'
        elif args.init_state_rule == 'random':
            indices = np.random.default_rng(args.seed).choice(env.initial_state_count, size=count, replace=False).tolist()
            rule = 'random_without_replacement'
        else:
            indices, rule = list(range(count)), 'first_n'
        if (len(indices) != count or len(set(indices)) != count
                or any(not 0 <= i < env.initial_state_count for i in indices)):
            parser.error('Initial indices must be valid, unique and match --episodes')
        output.mkdir(parents=True, exist_ok=False)
        protocol = {**env.episode_metadata(), 'policy': args.policy, 'device': device,
                    'selection_rule': rule, 'initial_state_indices': indices,
                    'seeds': list(range(args.seed, args.seed + count)),
                    'checkpoint': str(args.checkpoint.resolve()) if fingerprint else None,
                    'checkpoint_sha256': fingerprint,
                    'contract': policy.contract if isinstance(policy, BCPolicy) else None,
                    'torch_version': torch.__version__, 'torch_cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                    'contact_sampling': 'post-control-step endpoint only, not substep peaks',
                    'preview': 'display rotated 180 degrees; HDF5 and inference pixels unchanged'}
        write_json(output / 'protocol.json', protocol)

        def observe(observation: Observation, action: Action, info: Info) -> None:
            row = {'step': info['steps'], 'simulation_time': info['simulation_time'],
                   'native_success': info['success'], **task_diagnostics(env)}
            if isinstance(policy, BCPolicy):
                row['inference_ms'] = policy.inference_times_ms[-1]
            traces.setdefault(info['init_state_index'], []).append(row)
            if info['steps'] == 1 or info['steps'] % 100 == 0 or info['success']:
                resources.append({'stage': 'rollout', 'init_state_index': info['init_state_index'],
                                  'step': info['steps'], **resource_usage(device)})
                print(f"init={info['init_state_index']} step={info['steps']} success={info['success']}", flush=True)

        try:
            result = evaluate_task(env, policy=policy, policy_name=args.policy, episodes=count,
                                   seed=args.seed, initial_state_indices=indices,
                                   record_dir=output / 'records', on_step=observe)
        except Exception as error:
            write_json(output / 'error.json', {'error_type': type(error).__name__, 'error': str(error)})
            raise
    resources.append({'stage': 'after_environment_close', **resource_usage(device)})
    for episode in result['episodes']:
        rows = traces[episode['init_state_index']]
        index = episode['episode_index']
        write_json(output / f'episode_{index:03d}_diagnostics.json', {'steps': rows})
        episode['robot_external_contact_steps'] = sum(bool(r['robot_external_contacts']) for r in rows)
        if isinstance(policy, BCPolicy):
            times = [row['inference_ms'] for row in rows]
            episode['inference_ms'] = {'mean': float(np.mean(times)), 'median': float(np.median(times)),
                                       'p95': float(np.percentile(times, 95)), 'max': max(times)}
        preview = output / f'episode_{index:03d}_preview.png'
        save_preview(Path(episode['record_path']), preview)
        episode['preview'] = str(preview.resolve())
    resources.append({'stage': 'after_previews_close', **resource_usage(device)})
    result['resources'] = resources
    result['total_wall_seconds'] = perf_counter() - started
    result['checkpoint_unchanged'] = sha256(args.checkpoint) == fingerprint if fingerprint else None
    write_json(output / 'results.json', result)
    print(f"Success={result['success_rate']:.1%} ({count} episodes), "
          f"wall={result['total_wall_seconds']:.2f}s\nReports: {output}", flush=True)


if __name__ == '__main__':
    main()
