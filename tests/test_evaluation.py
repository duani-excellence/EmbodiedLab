"""Policy/runner/evaluator unit tests; simulator smoke tests live separately."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from embodiedlab.evaluation import _write_json, evaluate_suites, evaluate_task
from embodiedlab.policies.simple import RandomPolicy, ZeroPolicy
from embodiedlab.runners.rollout import run_episode


class FakeEnvironment:
    """A two-step public environment for verifying orchestration, not physics."""

    def __init__(self, suite='libero_spatial', task_id=0, image_size=8,
                 max_episode_steps=None, settle_steps=10, ending='truncated'):
        self.suite, self.task_id = suite, task_id
        self.task_name, self.language = f'task_{task_id}', 'pick up the object'
        self.initial_state_count = 3
        self.ending = ending
        self.resets, self.episode_actions = [], []
        self.closed = False
        self.mutate_actions = False

    def _observation(self):
        return {'robot_state': np.zeros(8), 'joint_positions': np.zeros(7),
                'agentview_image': np.zeros((8, 8, 3), dtype=np.uint8),
                'wrist_image': np.zeros((8, 8, 3), dtype=np.uint8), 'language': self.language}

    def reset(self, *, seed=None, init_state_index=None):
        self.steps = 0
        self.index = 0 if init_state_index is None else init_state_index
        self.resets.append((seed, self.index))
        self.episode_actions.append([])
        return self._observation(), {'success': False, 'simulation_time': 0., 'init_state_index': self.index}

    def step(self, action):
        assert action.shape == (7,)
        assert np.all(np.abs(action) <= 1)
        self.episode_actions[-1].append(action.copy())
        if self.mutate_actions:
            action[:] = 0
        self.steps += 1
        done = self.steps == 2
        terminated = done and self.ending == 'terminated'
        truncated = done and self.ending == 'truncated'
        return self._observation(), float(self.steps), terminated, truncated, {
            'success': bool(terminated), 'simulation_time': self.steps * .05,
            'steps': self.steps, 'init_state_index': self.index,
        }

    def episode_metadata(self):
        return {'suite': self.suite, 'task_id': self.task_id, 'task_name': self.task_name,
                'language': self.language, 'max_episode_steps': 2}

    def close(self):
        self.closed = True


class RecorderSpy:
    def __init__(self):
        self.actions, self.closed = [], False

    def set_environment_metadata(self, metadata):
        self.metadata = metadata

    def start(self, observation, info, *, seed):
        self.seed, self.initial_observation = seed, observation

    def append(self, action, observation, reward, terminated, truncated, info):
        self.actions.append(action)

    def close(self):
        self.closed = True


def test_zero_policy_is_literal_zero_and_independent():
    policy = ZeroPolicy()
    action = policy.act({'language': 'anything'})
    assert action.shape == (7,)
    assert np.array_equal(action, np.zeros(7))
    action[0] = 1
    assert policy.act({})[0] == 0
    policy.reset()


def test_random_policy_reset_and_seed_are_reproducible():
    first = RandomPolicy(seed=42)
    sequence = [first.act({}) for _ in range(3)]
    first.reset()
    assert all(np.array_equal(action, first.act({})) for action in sequence)
    assert np.array_equal(sequence[0], RandomPolicy(42).act({}))
    assert not np.array_equal(sequence[0], RandomPolicy(43).act({}))
    assert all(action.shape == (7,) and np.all(np.abs(action) <= 1) for action in sequence)


@pytest.mark.parametrize('seed', [-1, True, 1.2])
def test_random_policy_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match='seed'):
        RandomPolicy(seed)


@pytest.mark.parametrize('ending', ['terminated', 'truncated'])
def test_runner_ends_on_either_flag_and_accumulates_native_reward(ending):
    env = FakeEnvironment(ending=ending)
    result = run_episode(env, ZeroPolicy(), seed=42, init_state_index=2)
    assert env.resets == [(42, 2)]
    assert result.length == 2 and result.reward == 3
    assert result.success == (ending == 'terminated')
    assert result.terminated == (ending == 'terminated')
    assert result.truncated == (ending == 'truncated')
    assert result.simulation_time == .1 and result.init_state_index == 2


def test_runner_records_action_before_environment_mutates_it():
    env = FakeEnvironment()
    env.mutate_actions = True
    recorder = RecorderSpy()
    result = run_episode(env, RandomPolicy(42), seed=42, recorder=recorder)
    assert result.length == 2
    assert recorder.closed and recorder.seed == 42
    assert np.array_equal(recorder.actions[0], env.episode_actions[0][0])
    assert np.any(recorder.actions[0] != 0)
    assert recorder.initial_observation['agentview_image'].dtype == np.uint8
    assert recorder.initial_observation['language'] == env.language


def test_runner_closes_recorder_when_policy_raises():
    class BrokenPolicy(ZeroPolicy):
        def act(self, observation):
            raise RuntimeError('policy failed')
    recorder = RecorderSpy()
    with pytest.raises(RuntimeError, match='policy failed'):
        run_episode(FakeEnvironment(), BrokenPolicy(), seed=1, recorder=recorder)
    assert recorder.closed


def test_task_evaluation_uses_explicit_official_state_indices_and_seeds():
    env = FakeEnvironment()
    result = evaluate_task(env, policy_name='random', episodes=3, seed=42)
    assert env.resets == [(42, 0), (43, 1), (44, 2)]
    assert result['success_rate'] == 0 and result['mean_reward'] == 3
    assert result['mean_length'] == 2
    assert [episode['init_state_index'] for episode in result['episodes']] == [0, 1, 2]
    for index, actions in enumerate(env.episode_actions):
        assert np.array_equal(actions[0], RandomPolicy(42 + index).act({}))


@pytest.mark.parametrize('options', [{'episodes': 0}, {'episodes': 4},
                                    {'seed': -1}, {'policy_name': 'trained'}])
def test_task_evaluation_rejects_invalid_settings(options):
    with pytest.raises(ValueError):
        evaluate_task(FakeEnvironment(), **options)


def test_reports_are_exclusive_and_never_overwrite(tmp_path):
    path = tmp_path / 'settings.json'
    _write_json(path, {'seed': 42})
    with pytest.raises(FileExistsError):
        _write_json(path, {'seed': 99})
    assert json.loads(path.read_text()) == {'seed': 42}


def test_suite_evaluation_default_task_scope_and_separate_reports(monkeypatch, tmp_path):
    environments = []
    def make_env(**settings):
        env = FakeEnvironment(**settings)
        environments.append(env)
        return env
    monkeypatch.setitem(sys.modules, 'embodiedlab.envs.libero_env',
                        SimpleNamespace(LiberoEnv=make_env, list_task_ids=lambda suite: [0, 1]))
    first, directory = evaluate_suites(suites=['libero_spatial'], output_dir=tmp_path, episodes=2, seed=42)
    settings = json.loads((directory / 'settings.json').read_text())
    results = json.loads((directory / 'results.json').read_text())
    assert settings['task_ids'] == [0] and settings['episodes_per_task'] == 2
    assert results == first and results['episode_count'] == 2
    assert environments[0].closed
    second, other_directory = evaluate_suites(suites=['libero_spatial'], task_ids=None, output_dir=tmp_path)
    assert other_directory != directory and second['episode_count'] == 2
    assert [task['task_id'] for task in second['tasks']] == [0, 1]
    assert all(env.closed for env in environments)


def test_suite_evaluation_closes_environment_on_failure(monkeypatch, tmp_path):
    env = FakeEnvironment()
    monkeypatch.setitem(sys.modules, 'embodiedlab.envs.libero_env',
                        SimpleNamespace(LiberoEnv=lambda **settings: env, list_task_ids=lambda suite: [0]))
    with pytest.raises(ValueError, match='episodes'):
        evaluate_suites(suites=['libero_spatial'], episodes=4, output_dir=tmp_path)
    assert env.closed
