"""Inference bridge checks using the original E5 checkpoint and real observations."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from embodiedlab.data.episode import EpisodeDataset
from embodiedlab.data.torch_dataset import preprocess_observation
from embodiedlab.evaluation import evaluate_task
from embodiedlab.policies.bc import BCPolicy
from embodiedlab.policies.visual_bc import VisualBC, decode_actions


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'training_results/phase5b3_train_seed42/best.pt'
OFFICIAL = ROOT / 'data/libero/libero_goal/turn_on_the_stove_demo.hdf5'


@pytest.fixture(scope='module', autouse=True)
def require_legacy_checkpoint() -> None:
    if not CHECKPOINT.is_file():
        pytest.skip('Optional legacy E5 checkpoint is not distributed; '
                    'restore training_results/phase5b3_train_seed42/best.pt to run these regression tests')


@pytest.fixture(scope='module')
def saved_checkpoint() -> dict:
    before_hash = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    yield torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == before_hash


@pytest.fixture(scope='module')
def observations() -> tuple[dict, dict]:
    with EpisodeDataset.load(OFFICIAL, suite='libero_goal', task_id=7) as episode:
        return episode.observation(0), episode.observation(1)


def direct_model(saved: dict, device: torch.device) -> VisualBC:
    model = VisualBC()
    model.load_state_dict(saved['model_state_dict'])
    return model.to(device).eval().requires_grad_(False)


def direct_action(model: VisualBC, observation: dict, device: torch.device) -> np.ndarray:
    inputs = {key: value.unsqueeze(0).to(device)
              for key, value in preprocess_observation(observation).items()}
    with torch.inference_mode():
        return decode_actions(model(**inputs))[0].cpu().numpy().astype(np.float64)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_policy_matches_direct_checkpoint_and_preserves_inputs_and_weights(saved_checkpoint, observations, device) -> None:
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not accessible in this process')
    policy = BCPolicy(CHECKPOINT, device=device)
    reference = direct_model(saved_checkpoint, policy.device)
    observation = observations[0]
    before = {key: value.copy() if isinstance(value, np.ndarray) else value
              for key, value in observation.items()}
    weights = {key: value.detach().cpu().clone() for key, value in policy.model.state_dict().items()}
    expected = direct_action(reference, observation, policy.device)
    action = policy.act(observation)
    assert action.shape == (7,) and action.dtype == np.float64 and np.isfinite(action).all()
    assert np.all(np.abs(action[:6]) <= 1.) and action[6] in (-1., 1.)
    np.testing.assert_allclose(action, expected, rtol=1e-6, atol=1e-7)
    assert not policy.model.training and all(not parameter.requires_grad for parameter in policy.model.parameters())
    assert all(parameter.grad is None for parameter in policy.model.parameters())
    assert policy.checkpoint_metadata == {key: saved_checkpoint[key] for key in ('epoch', 'config', 'metrics')}
    for key, value in policy.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), saved_checkpoint['model_state_dict'][key], rtol=0, atol=0)
        torch.testing.assert_close(value.cpu(), weights[key], rtol=0, atol=0)
    for key in before:
        if isinstance(before[key], np.ndarray):
            np.testing.assert_array_equal(observation[key], before[key])
        else:
            assert observation[key] == before[key]


def test_reset_clears_timing_without_changing_predictions(observations) -> None:
    policy = BCPolicy(CHECKPOINT, device='cpu')
    first = policy.act(observations[0])
    policy.act(observations[1])
    assert len(policy.inference_times_ms) == 2
    assert all(np.isfinite(value) and value >= 0. for value in policy.inference_times_ms)
    policy.reset()
    assert policy.inference_times_ms == []
    np.testing.assert_array_equal(policy.act(observations[0]), first)
    assert len(policy.inference_times_ms) == 1


def test_auto_device_falls_back_to_real_cpu_inference(monkeypatch, observations) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    policy = BCPolicy(CHECKPOINT, device='auto')
    assert policy.device == torch.device('cpu')
    assert np.isfinite(policy.act(observations[0])).all()
    with pytest.raises(RuntimeError, match='CUDA|unavailable'):
        BCPolicy(CHECKPOINT, device='cuda')
    with pytest.raises(ValueError, match='device'):
        BCPolicy(CHECKPOINT, device='invalid')


@pytest.mark.parametrize('damage', ['wrong_rgb_contract', 'missing_weight', 'nonpositive_std', 'nan_weight'])
def test_invalid_checkpoint_copies_are_rejected(saved_checkpoint, tmp_path, damage) -> None:
    """Corrupt temporary copies only; the actual checkpoint is hash-checked unchanged."""
    saved = deepcopy(saved_checkpoint)
    if damage == 'wrong_rgb_contract':
        saved['config']['images'] = 'BGR'
    elif damage == 'missing_weight':
        del saved['model_state_dict']['image_encoder.0.weight']
    elif damage == 'nonpositive_std':
        saved['model_state_dict']['state_std'][0] = 0.
    else:
        saved['model_state_dict']['image_encoder.0.weight'].flatten()[0] = float('nan')
    path = tmp_path / f'{damage}.pt'
    torch.save(saved, path)
    with pytest.raises((ValueError, RuntimeError), match='Checkpoint|checkpoint|finite|std|Missing key'):
        BCPolicy(path, device='cpu')


def test_nonfinite_observation_is_rejected_before_recording_latency(observations) -> None:
    policy = BCPolicy(CHECKPOINT, device='cpu')
    damaged = deepcopy(observations[0])
    damaged['robot_state'][0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        policy.act(damaged)
    assert policy.inference_times_ms == []


class ProtocolEnvironment:
    """One-step wiring fixture with real arrays; it does not simulate BC actions."""

    def __init__(self, initial: dict, successor: dict) -> None:
        self.initial, self.successor = initial, successor
        self.suite, self.task_id, self.task_name = 'libero_goal', 7, 'turn_on_the_stove'
        self.language, self.initial_state_count = initial['language'], 2
        self.resets, self.received_actions, self.mutated_arguments = [], [], []

    def reset(self, *, seed=None, init_state_index=None):
        self.resets.append((seed, init_state_index))
        self.index = init_state_index
        return self.initial, {'success': False, 'simulation_time': 0., 'init_state_index': self.index}

    def step(self, action):
        self.received_actions.append(action.copy())
        self.mutated_arguments.append(action)
        action[:] = 0.  # Runner's observer must still receive the command that was issued.
        info = {'success': False, 'steps': 1, 'simulation_time': .05, 'init_state_index': self.index}
        return self.successor, .25, False, True, info

    def episode_metadata(self):
        return {'suite': self.suite, 'task_id': self.task_id, 'task_name': self.task_name,
                'image_size': 128, 'image_orientation': 'raw_robosuite_opengl',
                'control_mode': 'osc_pose_delta_normalized', 'action_dim': 7, 'control_dt': .05,
                'position_delta_scale_m': .05, 'axis_angle_delta_scale_rad': .5,
                'action_low': [-1.] * 7, 'action_high': [1.] * 7, 'gripper': '-1 open, +1 close'}


def test_python_evaluator_rejects_task_mismatch_before_any_reset(observations) -> None:
    policy = BCPolicy(CHECKPOINT, device='cpu')
    env = ProtocolEnvironment(*observations)
    env.task_id = 0
    with pytest.raises(ValueError, match='incompatible.*task_id'):
        evaluate_task(env, policy=policy, policy_name='bc')
    assert env.resets == []


def test_custom_policy_evaluation_observes_postobs_and_independent_action_copies(observations) -> None:
    policy = BCPolicy(CHECKPOINT, device='cpu')
    expected = policy.act(observations[0])
    policy.reset()
    env = ProtocolEnvironment(*observations)
    events = []
    report = evaluate_task(env, policy=policy, policy_name='bc', episodes=2, seed=12,
                           on_step=lambda obs, action, info: events.append((obs, action, dict(info))))
    assert env.resets == [(12, 0), (13, 1)] and report['policy'] == 'bc'
    assert report['success_rate'] == 0. and report['mean_reward'] == .25 and report['mean_length'] == 1.
    assert len(events) == 2 and len(policy.inference_times_ms) == 1
    for index, (observation, action, info) in enumerate(events):
        assert observation is env.successor and info['steps'] == 1 and info['init_state_index'] == index
        np.testing.assert_array_equal(action, expected)
        np.testing.assert_array_equal(env.received_actions[index], expected)
        assert not np.shares_memory(action, env.mutated_arguments[index])
        result = report['episodes'][index]
        assert result['length'] == 1 and result['truncated'] and not result['terminated']
        assert result['first_success_step'] is None


def test_native_three_step_observations_match_direct_model(saved_checkpoint) -> None:
    """Real headless smoke test; run on the host with EGL/device access."""
    from embodiedlab.envs.libero_env import LiberoEnv

    policy = BCPolicy(CHECKPOINT, device='auto')
    reference = direct_model(saved_checkpoint, policy.device)
    with LiberoEnv(suite='libero_goal', task_id=7, image_size=128,
                   max_episode_steps=3, settle_steps=10) as env:
        observation, _ = env.reset(seed=42, init_state_index=0)
        for step in range(1, 4):
            expected = direct_action(reference, observation, policy.device)
            action = policy.act(observation)
            np.testing.assert_allclose(action, expected, rtol=1e-6, atol=1e-7)
            observation, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward) and terminated == info['success']
            if terminated or truncated:
                break
        assert terminated or truncated
        assert info['steps'] == step and len(policy.inference_times_ms) == step
        if not terminated:
            assert step == 3 and truncated
