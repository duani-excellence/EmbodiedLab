"""Exceptional resource cleanup and explicit evaluation-state selection.

The tiny environment below checks orchestration only, never robot performance.
Actual native multi-episode runs are reported separately in Phase 5C.
"""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from embodiedlab.data.episode import EpisodeDataset, EpisodeRecorder
from embodiedlab.evaluation import evaluate_task
from embodiedlab.policies.simple import RandomPolicy, ZeroPolicy
from embodiedlab.runners.rollout import run_episode


class TinyEnvironment:
    """Two transitions, three initial-state IDs, and valid recording metadata."""

    suite, task_id = 'libero_goal', 7
    task_name, language = 'turn_on_the_stove', 'turn on the stove'
    initial_state_count = 3

    def __init__(self, *, fail_step: bool = False):
        self.fail_step = fail_step
        self.resets, self.actions = [], []

    def _observation(self):
        return {'robot_state': np.zeros(8), 'joint_positions': np.zeros(7),
                'agentview_image': np.zeros((4, 4, 3), dtype=np.uint8),
                'wrist_image': np.zeros((4, 4, 3), dtype=np.uint8),
                'language': self.language}

    def reset(self, *, seed=None, init_state_index=None):
        self.steps, self.index = 0, init_state_index
        self.resets.append((seed, init_state_index))
        self.actions.append([])
        return self._observation(), {'simulation_time': 0.}

    def step(self, action):
        if self.fail_step:
            raise RuntimeError('environment step failed')
        self.actions[-1].append(action.copy())
        self.steps += 1
        return self._observation(), 0., False, self.steps == 2, {
            'steps': self.steps, 'simulation_time': self.steps * .05,
            'success': False, 'init_state_index': self.index}

    def episode_metadata(self):
        return {'suite': self.suite, 'task_id': self.task_id,
                'task_name': self.task_name, 'language': self.language,
                'max_episode_steps': 2, 'control_dt': .05, 'image_size': 4,
                'init_state_index': self.index, 'settle_steps': 0}


def test_explicit_initial_states_keep_order_and_ordinal_random_seeds():
    env = TinyEnvironment()
    report = evaluate_task(env, policy_name='random', episodes=2, seed=42,
                           initial_state_indices=[2, 0])
    assert env.resets == [(42, 2), (43, 0)]
    assert [row['init_state_index'] for row in report['episodes']] == [2, 0]
    assert [row['seed'] for row in report['episodes']] == [42, 43]
    for ordinal, actions in enumerate(env.actions):
        np.testing.assert_array_equal(actions[0], RandomPolicy(42 + ordinal).act({}))


@pytest.mark.parametrize('indices, episodes', [
    ([], 1), ([0], 2), ([0, 1], 1), ([2, 2], 2),
    ([-1], 1), ([3], 1), ([True], 1), ([0.], 1),
])
def test_invalid_initial_state_selections_are_rejected_before_reset(indices, episodes):
    env = TinyEnvironment()
    with pytest.raises(ValueError, match='initial_state_indices|indices|state'):
        evaluate_task(env, episodes=episodes, initial_state_indices=indices)
    assert not env.resets


@pytest.mark.parametrize('failure', ['environment', 'observer'])
def test_runner_interruptions_close_real_hdf_and_mark_it_incomplete(tmp_path, failure):
    path = tmp_path / f'{failure}.hdf5'
    recorder = EpisodeRecorder(path)
    env = TinyEnvironment(fail_step=failure == 'environment')

    def broken_observer(observation, action, info):
        raise RuntimeError('observer failed')

    with pytest.raises(RuntimeError, match='environment step failed|observer failed'):
        run_episode(env, ZeroPolicy(), seed=42, init_state_index=0,
                    recorder=recorder,
                    on_step=broken_observer if failure == 'observer' else None)
    assert recorder._file is None
    # Observer failure happens after append(), so one valid transition survives.
    with h5py.File(path, 'r') as file:
        group = file['data/demo_0']
        assert group['actions'].shape[0] == (1 if failure == 'observer' else 0)
        assert json.loads(group.attrs['metadata_json'])['complete'] is False
    with pytest.raises(ValueError, match='incomplete'):
        EpisodeDataset.load(path)


def test_episode_reader_closes_after_validation_failure_and_context_exception(monkeypatch, tmp_path):
    path = tmp_path / 'complete.hdf5'
    run_episode(TinyEnvironment(), ZeroPolicy(), seed=42, init_state_index=0,
                recorder=EpisodeRecorder(path))
    original_file = h5py.File
    handles = []

    def tracked_file(*args, **kwargs):
        handle = original_file(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(h5py, 'File', tracked_file)
    with pytest.raises(RuntimeError, match='reader caller failed'):
        with EpisodeDataset.load(path) as episode:
            assert len(episode) == 2
            episode.observation(1)
            raise RuntimeError('reader caller failed')
    assert handles and all(not handle.id.valid for handle in handles)
    episode.close()  # Also safe after context-manager cleanup.

    with original_file(path, 'r+') as file:
        file['data/demo_0/actions'][0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        EpisodeDataset.load(path)
    assert len(handles) == 2 and all(not handle.id.valid for handle in handles)


def test_libero_context_exception_closes_native_handle_once():
    from embodiedlab.envs.libero_env import LiberoEnv

    # Isolate Python resource ownership; native rendering is tested separately.
    env = LiberoEnv.__new__(LiberoEnv)
    close_calls = []
    env._env = SimpleNamespace(close=lambda: close_calls.append(True))
    env._closed, env._ended = False, False
    with pytest.raises(RuntimeError, match='caller failed'):
        with env:
            raise RuntimeError('caller failed')
    env.close()
    assert close_calls == [True]
    assert env._env is None and env._closed and env._ended
def test_real_headless_observer_failure_closes_simulator_and_incomplete_record(tmp_path):
    """A genuine native step followed by a diagnostic exception, not a fake env."""
    import h5py
    import json
    from embodiedlab.envs.libero_env import LiberoEnv
    from embodiedlab.data.episode import EpisodeDataset, EpisodeRecorder
    from embodiedlab.policies.simple import ZeroPolicy
    from embodiedlab.runners.rollout import run_episode

    env = LiberoEnv(suite='libero_goal', task_id=0, image_size=32, max_episode_steps=3)
    path = tmp_path / 'interrupted_native.hdf5'
    recorder = EpisodeRecorder(path)

    def interrupt(observation, action, info):
        assert info['steps'] == 1
        raise RuntimeError('deliberate observer failure')

    with pytest.raises(RuntimeError, match='deliberate observer failure'):
        with env:
            run_episode(env, ZeroPolicy(), seed=42, init_state_index=0,
                        recorder=recorder, on_step=interrupt)
    assert env._closed and env._env is None and recorder._file is None
    with h5py.File(path, 'r') as file:
        assert file['data/demo_0/actions'].shape == (1, 7)
        assert json.loads(file['data/demo_0'].attrs['metadata_json'])['complete'] is False
    with pytest.raises(ValueError, match='incomplete'):
        EpisodeDataset.load(path)

