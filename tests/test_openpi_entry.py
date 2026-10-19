"""CLI bookkeeping/resource tests with explicit fixtures, never a model benchmark."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip('openpi_client', reason='Optional OpenPI client is not installed')

from embodiedlab.data import episode as episode_module


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def entry():
    spec = importlib.util.spec_from_file_location('openpi_entry_under_test', ROOT / 'scripts/evaluate_openpi.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launch_identity() -> dict:
    pin = json.loads((ROOT / 'openpi-upstream.json').read_text())
    checkpoint = pin['checkpoint']
    return {'run_id': 'explicit-test-fixture-run', 'openpi_commit': pin['source']['commit'],
            'config': checkpoint['config'], 'checkpoint': checkpoint['uri'],
            'checkpoint_sha256': checkpoint['content_sha256'], 'backend': 'jax',
            'action_horizon': 10, 'gpu': 'NVIDIA GeForce RTX 5090 D v2',
            'norm_stats_sha256': next(item['sha256'] for item in checkpoint['files']
                                      if item['path'].endswith('/norm_stats.json'))}


def test_launch_identity_accepts_matching_verified_manifest(entry, launch_identity):
    entry.validate_server_manifest(launch_identity, deepcopy(launch_identity))


@pytest.mark.parametrize('damage', ['wrong_checkpoint', 'wrong_backend', 'different_run', 'missing_run'])
def test_launch_identity_rejects_wrong_model_or_stale_handshake(entry, launch_identity, damage):
    manifest, handshake = deepcopy(launch_identity), deepcopy(launch_identity)
    if damage == 'wrong_checkpoint':
        manifest['checkpoint_sha256'] = handshake['checkpoint_sha256'] = 'wrong-fixture-checksum'
    elif damage == 'wrong_backend':
        manifest['backend'] = handshake['backend'] = 'pytorch'
    elif damage == 'different_run':
        handshake['run_id'] = 'another-explicit-fixture-launch'
    else:
        del manifest['run_id']
        del handshake['run_id']
    with pytest.raises(ValueError, match='identity mismatch'):
        entry.validate_server_manifest(manifest, handshake)


@pytest.fixture
def frozen_environment_fixture(tmp_path, monkeypatch):
    """Invented bytes/states validate the guard, never model/physics performance."""
    bddl_root, init_root = tmp_path / 'bddl_files', tmp_path / 'init_files'
    for root in (bddl_root, init_root):
        (root / 'explicit_fixture').mkdir(parents=True)
    bddl, init = bddl_root / 'explicit_fixture/fixture.bddl', init_root / 'explicit_fixture/fixture.init'
    bddl.write_bytes(b'EXPLICIT SYNTHETIC BDDL BYTES')
    init.write_bytes(b'EXPLICIT SYNTHETIC INIT BYTES')
    module = ModuleType('libero.libero')
    module.get_libero_path = lambda key: str({'bddl_files': bddl_root, 'init_states': init_root}[key])
    monkeypatch.setitem(sys.modules, 'libero.libero', module)
    states = np.arange(36, dtype=np.float64).reshape(12, 3)
    native_task = SimpleNamespace(name='explicit_fixture', language='explicit fixture language',
                                 problem_folder='explicit_fixture', bddl_file='fixture.bddl',
                                 init_states_file='fixture.init')
    metadata = {'libero_commit': 'explicit_fixture_commit', 'libero_version': '0.1.4',
                'assets_revision': 'explicit_fixture_assets', 'mujoco_version': '3.3.2',
                'robosuite_version': '1.4.0', 'control_dt': .05, 'settle_steps': 10}
    environment = {'libero_commit': metadata['libero_commit'], 'libero_version': '0.1.4',
                   'assets_revision': metadata['assets_revision'], 'mujoco': '3.3.2',
                   'robosuite': '1.4.0', 'control_dt': .05, 'settle_steps': 10}
    task = {'suite': 'libero_goal', 'task_id': 7, 'task_name': native_task.name,
            'language': native_task.language, 'max_episode_steps': 2,
            'bddl_relative_path': 'bddl_files/explicit_fixture/fixture.bddl',
            'init_file_relative_path': 'init_files/explicit_fixture/fixture.init',
            'bddl_sha256': hashlib.sha256(bddl.read_bytes()).hexdigest(),
            'init_file_sha256': hashlib.sha256(init.read_bytes()).hexdigest(),
            'initial_states_shape': [12, 3],
            'first_10_state_sha256': [hashlib.sha256(row.tobytes(order='C')).hexdigest() for row in states[:10]],
            'formal': {'initial_state_indices': [0], 'seeds': [42]}}

    class FixtureEnvironment(SimpleNamespace):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def reset(self, **_kwargs):
            self.resets += 1
            raise AssertionError('Guard fixtures must never reset')

    env = FixtureEnvironment(suite='libero_goal', task_id=7, task_name=native_task.name,
        language=native_task.language, max_episode_steps=2, _task=native_task,
        _initial_states=states, _env=None, initial_state_count=12, resets=0,
        closed=False, episode_metadata=lambda: deepcopy(metadata))
    return env, task, environment, metadata, bddl, init


def test_frozen_environment_accepts_exact_fixture_without_reset(entry, frozen_environment_fixture):
    env, task, environment, *_ = frozen_environment_fixture
    before = deepcopy(task), deepcopy(environment), env._initial_states.copy()
    entry.validate_challenge_environment(env, task, environment)
    assert env.resets == 0 and env._env is None
    assert task == before[0] and environment == before[1]
    assert np.array_equal(env._initial_states, before[2])


@pytest.mark.parametrize('damage,field', [
    ('task_name', 'task_name'), ('language', 'language'),
    ('native_task_name', 'native task_name'), ('native_language', 'native language'),
    ('bddl_bytes', 'bddl_sha256'), ('init_bytes', 'init_file_sha256'),
    ('bddl_missing', 'cannot read actual asset'), ('native_path', 'bddl_relative_path'),
    ('state_shape', 'initial_states_shape'), ('state_value', 'first_10_state_sha256'),
    ('libero_version', 'libero_version'), ('mujoco_version', 'mujoco_version'),
    ('robosuite_version', 'robosuite_version'), ('libero_commit', 'libero_commit'),
    ('assets_revision', 'assets_revision'), ('control_dt', 'control_dt'),
    ('settle_steps', 'settle_steps')])
def test_frozen_environment_rejects_actual_runtime_and_asset_drift_before_reset(
        entry, frozen_environment_fixture, damage, field):
    env, task, environment, metadata, bddl, init = frozen_environment_fixture
    if damage in ('task_name', 'language'):
        setattr(env, damage, 'different fixture')
    elif damage.startswith('native_'):
        setattr(env._task, {'native_task_name': 'name', 'native_language': 'language',
                           'native_path': 'bddl_file'}[damage], 'different fixture')
    elif damage == 'bddl_bytes':
        bddl.write_bytes(b'DIFFERENT BDDL BYTES')
    elif damage == 'init_bytes':
        init.write_bytes(b'DIFFERENT INIT BYTES')
    elif damage == 'bddl_missing':
        bddl.unlink()
    elif damage == 'state_shape':
        env._initial_states = env._initial_states[:11]
    elif damage == 'state_value':
        env._initial_states[0, 0] += 1.
    else:
        metadata[damage] = 'different fixture'
    with pytest.raises(ValueError, match=field):
        entry.validate_challenge_environment(env, task, environment)
    assert env.resets == 0


def test_benchmark_cli_runs_guard_before_first_reset_and_closes_resources(
        entry, launch_identity, frozen_environment_fixture, monkeypatch, tmp_path):
    env, task, environment, metadata, *_ = frozen_environment_fixture
    metadata['mujoco_version'] = 'explicit fixture wrong version'
    launch_identity['server_rng'] = 'official key(0), advances per request; reset does not reset server'

    class FixturePolicy:
        def __init__(self, **_kwargs):
            self.server_metadata = deepcopy(launch_identity)
            self.closed = False

        def close(self):
            self.closed = True

    policy = FixturePolicy()
    monkeypatch.setattr(entry, 'OpenPIPolicy', lambda **_kwargs: policy)
    monkeypatch.setattr(entry, 'LiberoEnv', lambda **_kwargs: env)
    monkeypatch.setattr(entry, 'evaluate_task', lambda *_args, **_kwargs: pytest.fail('Guard must run before evaluation'))
    launch_path, plan_path = tmp_path / 'launch.json', tmp_path / 'plan.json'
    launch_path.write_text(json.dumps(launch_identity))
    plan_path.write_text(json.dumps({'manifest_version': 1, 'tasks': [task], 'environment': environment,
        'models': {'openpi': {'replan_steps': 5, 'model_seed': 0}}}))
    output = tmp_path / 'guard_output'
    monkeypatch.setattr(sys, 'argv', ['evaluate_openpi.py', '--suite', 'libero_goal', '--task-id', '7',
        '--init-state-indices', '0', '--seed', '42', '--server-manifest', str(launch_path),
        '--benchmark-manifest', str(plan_path), '--output-dir', str(output)])
    with pytest.raises(SystemExit) as stopped:
        entry.main()
    assert stopped.value.code == 1
    assert env.closed and policy.closed and env.resets == 0
    report = json.loads(next(output.glob('*/results.json')).read_text())
    assert report['attempted'] == 0 and report['unattempted'] == 1
    assert 'mujoco_version' in report['error']['message']


def test_frozen_openpi_rng_accepts_fixed_key_zero_and_rejects_drift(entry, launch_identity):
    launch_identity['server_rng'] = 'official key(0), advances per request; reset does not reset server'
    entry.validate_server_manifest(launch_identity, deepcopy(launch_identity), {'model_seed': 0})
    handshake = deepcopy(launch_identity)
    handshake['server_rng'] = 'key(1)'
    with pytest.raises(ValueError, match='server_rng'):
        entry.validate_server_manifest(launch_identity, handshake, {'model_seed': 0})
    with pytest.raises(ValueError, match='model_seed'):
        entry.validate_server_manifest(launch_identity, launch_identity, {'model_seed': 1})


@pytest.fixture
def oft_identity():
    pin = json.loads((ROOT / 'openvla-upstream.json').read_text())
    return {'run_id': 'explicit_fixture_oft_run', 'gpu': 'explicit_fixture_gpu', 'model': 'openvla_oft',
        'source_commit': pin['source']['commit'], 'transformers_commit': pin['transformers']['commit'],
        'checkpoint': pin['checkpoint']['repo_id'], 'revision': pin['checkpoint']['revision'],
        'checkpoint_sha256': pin['checkpoint']['content_sha256'], 'backend': 'torch',
        'action_horizon': 8, 'seed': 42, 'dtype': 'bfloat16', 'quantization': False, 'attention': 'sdpa'}


def test_oft_frozen_runtime_accepts_pin_and_plan(entry, oft_identity):
    plan = {'model_seed': 42, 'runtime_required': {'dtype': 'bfloat16', 'quantization': False,
                                                'attention': 'official fork SDPA bidirectional'}}
    entry.validate_openvla_manifest(oft_identity, deepcopy(oft_identity), plan)


@pytest.mark.parametrize('field,value', [('seed', 43), ('dtype', 'float16'), ('quantization', True),
                                       ('quantization', 0), ('attention', 'flash_attention_2')])
def test_oft_rejects_self_consistent_wrong_runtime(entry, oft_identity, field, value):
    # Agreement between the launch and handshake alone must not permit drift.
    oft_identity[field] = value
    with pytest.raises(ValueError, match=f'identity mismatch: {field}'):
        entry.validate_openvla_manifest(oft_identity, deepcopy(oft_identity))


@pytest.mark.parametrize('field,value', [('model_seed', 43), ('dtype', 'float16'),
                                       ('quantization', True), ('attention', 'different attention')])
def test_oft_rejects_frozen_plan_runtime_drift(entry, oft_identity, field, value):
    plan = {'model_seed': 42, 'runtime_required': {'dtype': 'bfloat16', 'quantization': False,
                                                'attention': 'official fork SDPA bidirectional'}}
    (plan if field == 'model_seed' else plan['runtime_required'])[field] = value
    with pytest.raises(ValueError, match=field):
        entry.validate_openvla_manifest(oft_identity, deepcopy(oft_identity), plan)


@pytest.mark.parametrize('model,expected_horizon,expected_profile', [
    ('openpi', 280, 'embodiedlab'), ('openvla_oft', 220, 'model_reference')])
def test_standalone_spatial_cli_keeps_model_specific_default_horizon(
        entry, launch_identity, oft_identity, monkeypatch, tmp_path,
        model, expected_horizon, expected_profile):
    """Stop at environment construction; no simulator, policy request, or step."""
    identity = launch_identity if model == 'openpi' else oft_identity
    captured = {}

    class FixturePolicy:
        server_metadata = deepcopy(identity)

        def close(self):
            captured['policy_closed'] = True

    def stop_at_constructor(**kwargs):
        captured.update(kwargs)
        raise ValueError('explicit fixture stop after collecting the selected horizon')

    monkeypatch.setattr(entry, 'LiberoEnv', stop_at_constructor)
    monkeypatch.setattr(entry, 'OpenPIPolicy', lambda **_kwargs: FixturePolicy())
    oft_module = ModuleType('embodiedlab.policies.openvla_oft')
    oft_module.OpenVLAOFTPolicy = lambda **_kwargs: FixturePolicy()
    monkeypatch.setitem(sys.modules, 'embodiedlab.policies.openvla_oft', oft_module)
    launch_path = tmp_path / 'launch.json'
    launch_path.write_text(json.dumps(identity))
    monkeypatch.setattr(sys, 'argv', ['evaluate_entry.py', '--suite', 'libero_spatial',
        '--task-id', '0', '--init-state-indices', '0', '--server-manifest', str(launch_path),
        '--output-dir', str(tmp_path / 'output')])
    # Capture the selected parser default as well as the actual constructor horizon.
    parse = entry.argparse.ArgumentParser.parse_args

    def capture_parse(parser, *args, **kwargs):
        parsed = parse(parser, *args, **kwargs)
        captured['evaluation_profile'] = parsed.evaluation_profile
        return parsed

    monkeypatch.setattr(entry.argparse.ArgumentParser, 'parse_args', capture_parse)
    with pytest.raises(SystemExit) as stopped:
        entry.main(model)
    assert stopped.value.code == 1
    assert captured['max_episode_steps'] == expected_horizon
    assert captured['evaluation_profile'] == expected_profile
    assert captured['policy_closed'] is True


@pytest.mark.parametrize('failure', [None, 'rpc', 'reset'])
def test_cli_preserves_completed_episode_and_closes_resources(
        entry, launch_identity, monkeypatch, tmp_path, failure):
    """Use the real Evaluator/Runner/Recorder with two-step fixture dynamics.

    Episode 0 succeeds. Episode 1 times out, or deliberately fails during RPC
    or reset. None of these fixtures represent LIBERO model performance.
    """
    resources = {'env': None, 'policy': None, 'recorders': []}

    class FixtureEnvironment:
        def __init__(self, *, suite, task_id, image_size, max_episode_steps, settle_steps):
            self.suite, self.task_id = suite, task_id
            self.task_name, self.language = 'explicit_fixture', 'explicit fixture task'
            self.initial_state_count, self.max_episode_steps = 2, 2
            self.image_size, self.settle_steps = image_size, settle_steps
            self.index, self.steps, self.closed, self.resets = 0, 0, False, []
            resources['env'] = self

        def observation(self):
            return {'robot_state': np.full(8, self.index + self.steps / 10., dtype=np.float64),
                    'joint_positions': np.zeros(7, dtype=np.float64),
                    'agentview_image': np.zeros((256, 256, 3), dtype=np.uint8),
                    'wrist_image': np.zeros((256, 256, 3), dtype=np.uint8),
                    'language': self.language}

        def info(self, success=False):
            return {'steps': self.steps, 'success': success,
                    'simulation_time': self.steps / 20., 'init_state_index': self.index}

        def reset(self, *, seed, init_state_index):
            self.index, self.steps = init_state_index, 0
            self.resets.append((seed, init_state_index))
            if failure == 'reset' and self.index == 1:
                raise RuntimeError('explicit fixture second reset failure')
            return self.observation(), self.info()

        def step(self, action):
            assert action.shape == (7,) and np.isfinite(action).all()
            self.steps += 1
            ended, success = self.steps == 2, self.steps == 2 and self.index == 0
            return self.observation(), float(success), success, ended and not success, self.info(success)

        def episode_metadata(self):
            return {'suite': self.suite, 'task_id': self.task_id, 'task_name': self.task_name,
                    'image_size': self.image_size, 'image_orientation': 'raw_robosuite_opengl',
                    'control_mode': 'osc_pose_delta_normalized', 'action_dim': 7,
                    'control_dt': .05, 'max_episode_steps': self.max_episode_steps,
                    'settle_steps': self.settle_steps, 'position_delta_scale_m': .05,
                    'axis_angle_delta_scale_rad': .5, 'gripper': '-1 open, +1 close'}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

    class FixturePolicy:
        def __init__(self, **_kwargs):
            self.server_metadata, self.closed = deepcopy(launch_identity), False
            resources['policy'] = self
            self.reset()

        def assert_compatible(self, env):
            assert env.episode_metadata()['image_size'] == 256

        def reset(self):
            self.request_log, self.act_times_ms, self.projection_log = [], [], []
            self.projected_action_count, self.projected_dimension_counts = 0, [0] * 7
            self.max_action_overshoot = 0.

        def act(self, observation):
            self.act_times_ms.append(.2)
            request = {'rpc_ms': .1, 'raw_chunk': np.zeros((10, 7)).tolist(),
                       'synchronized_sample_ms': .05}
            self.request_log.append(request)
            env = resources['env']
            if failure == 'rpc' and env.index == 1 and env.steps == 1:
                request['error'] = 'explicit fixture RPC deadline'
                raise TimeoutError('explicit fixture RPC deadline')
            return np.array([0., 0., 0., 0., 0., 0., -1.])

        def close(self):
            self.closed = True

    class TrackedRecorder(episode_module.EpisodeRecorder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            resources['recorders'].append(self)

        def close(self):
            self.close_calls += 1
            super().close()

    monkeypatch.setattr(entry, 'LiberoEnv', FixtureEnvironment)
    monkeypatch.setattr(entry, 'OpenPIPolicy', FixturePolicy)
    monkeypatch.setattr(entry, 'task_diagnostics', lambda _env: {'explicit_fixture': True})
    monkeypatch.setattr(episode_module, 'EpisodeRecorder', TrackedRecorder)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(launch_identity))
    output = tmp_path / 'evaluation'
    monkeypatch.setattr(sys, 'argv', ['evaluate_openpi.py', '--suite', 'libero_goal',
                                    '--task-id', '7', '--init-state-indices', '0', '1',
                                    '--seed', '7', '--server-manifest', str(manifest_path),
                                    '--output-dir', str(output)])
    if failure is None:
        entry.main()
    else:
        with pytest.raises(SystemExit) as stopped:
            entry.main()
        assert stopped.value.code == 1

    run = next(output.iterdir())
    report = json.loads((run / 'results.json').read_text())
    assert resources['env'].closed and resources['policy'].closed
    assert resources['env'].resets == [(7, 0), (8, 1)]
    assert len(resources['recorders']) == 2
    assert all(rec.close_calls >= 1 and rec._file is None for rec in resources['recorders'])
    assert report['attempted'] == 2 and report['unattempted'] == 0
    first, second = report['episodes']
    assert first['status'] == 'complete' and first['success'] is True and first['length'] == 2
    assert first == json.loads((run / 'episode_000.json').read_text())
    with episode_module.EpisodeDataset(first['record_path']) as saved:
        assert len(saved) == 2 and saved.metadata['success'] is True
    if failure is None:
        assert report['status'] == 'complete' and report['completed'] == 2
        assert report['success_rate_denominator'] == 2 and report['success_rate'] == .5
        assert second['status'] == 'complete' and second['success'] is False
        with episode_module.EpisodeDataset(second['record_path']) as saved:
            assert len(saved) == 2 and saved.metadata['success'] is False
    else:
        assert report['status'] == 'incomplete' and report['completed'] == report['incomplete'] == 1
        assert report['success_rate_denominator'] == 1 and report['success_rate'] == 1.
        assert second['status'] == 'incomplete' and second['success'] is None
        assert second['completed_steps'] == (1 if failure == 'rpc' else 0)
        diagnostics = json.loads((run / 'episode_001_diagnostics.json').read_text())
        assert len(diagnostics['steps']) == second['completed_steps']
        assert second['inference_requests'] == (2 if failure == 'rpc' else 0)
        assert len(diagnostics['requests']) == second['inference_requests']
        if failure == 'rpc':
            # Reopening for write proves the interrupted writer has been closed.
            with h5py.File(second['record_path'], 'r+') as recorded:
                metadata = json.loads(recorded['data/demo_0'].attrs['metadata_json'])
                assert metadata['complete'] is False
            with pytest.raises(ValueError, match='incomplete'):
                episode_module.EpisodeDataset(second['record_path'])
        else:
            assert not Path(second['record_path']).exists()
