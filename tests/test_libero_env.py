"""Real pinned LIBERO integration checks, run with .venv-libero/bin/python."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from embodiedlab.envs.libero_env import LiberoEnv, list_task_ids

SUITES = ('libero_spatial', 'libero_object')
IMAGE_SIZE = 64
NO_OP = np.array([0., 0., 0., 0., 0., 0., -1.])


@pytest.fixture(scope='module', params=SUITES)
def native_env(request):
    """Two genuine upstream tasks; always release their offscreen contexts."""
    env = LiberoEnv(suite=request.param, task_id=0, image_size=IMAGE_SIZE,
                    max_episode_steps=3, init_state_index=0, settle_steps=10)
    try:
        yield env
    finally:
        env.close()


@pytest.mark.parametrize('suite', SUITES)
def test_real_suite_inventory(suite: str) -> None:
    assert list_task_ids(suite) == list(range(10))


@pytest.mark.parametrize('options, message', [
    ({'suite': 'unknown_suite'}, 'Unknown suite'),
    ({'task_id': -1}, 'task_id'),
    ({'task_id': 10}, 'task_id'),
    ({'task_id': True}, 'task_id'),
    ({'image_size': 0}, 'image_size'),
    ({'image_size': 64.0}, 'image_size'),
    ({'settle_steps': -1}, 'settle_steps'),
    ({'settle_steps': 1.5}, 'settle_steps'),
    ({'max_episode_steps': 0}, 'max_episode_steps'),
    ({'init_state_index': -1}, 'init_state_index'),
    ({'init_state_index': True}, 'init_state_index'),
    ({'init_state_index': 1_000_000}, 'init_state_index'),
])
def test_invalid_constructor_options_need_no_rendering(options: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LiberoEnv(**options)


def test_canonical_state_and_two_real_raw_rgb_images(native_env) -> None:
    observation, info = native_env.reset(seed=42, init_state_index=0)
    assert set(observation) == {'robot_state', 'joint_positions', 'agentview_image',
                                'wrist_image', 'language'}
    assert observation['robot_state'].shape == (8,)
    assert observation['joint_positions'].shape == (7,)
    assert np.isfinite(observation['robot_state']).all()
    assert np.isfinite(observation['joint_positions']).all()
    assert observation['language'] == native_env.language and native_env.language.strip()
    # LIBERO chooses its official Panda mounting variant for each task domain.
    expected_robot = {'libero_spatial': 'MountedPanda', 'libero_object': 'OnTheGroundPanda'}
    assert native_env._env.robots[0].robot_model.__class__.__name__ == expected_robot[native_env.suite]
    controller = native_env._env.robots[0].controller
    assert controller.use_delta and controller.control_dim == 6
    np.testing.assert_allclose(controller.output_max, [.05, .05, .05, .5, .5, .5])
    raw = native_env._env.env._get_observations()
    from robosuite.utils.transform_utils import quat2axisangle
    np.testing.assert_allclose(observation['robot_state'][:3], raw['robot0_eef_pos'])
    np.testing.assert_allclose(observation['robot_state'][3:6],
                               quat2axisangle(raw['robot0_eef_quat'].copy()))
    np.testing.assert_allclose(observation['robot_state'][6:], raw['robot0_gripper_qpos'])
    np.testing.assert_array_equal(observation['joint_positions'], raw['robot0_joint_pos'])
    for public_key, raw_key in [('agentview_image', 'agentview_image'),
                                ('wrist_image', 'robot0_eye_in_hand_image')]:
        image = observation[public_key]
        assert image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and image.dtype == np.uint8
        assert np.ptp(image) > 0  # Actual rendered scene, not a blank placeholder.
        np.testing.assert_array_equal(image, raw[raw_key])
        assert not np.shares_memory(image, raw[raw_key])
    assert info['steps'] == 0 and info['simulation_time'] == 0
    assert info['init_state_index'] == 0 and info['task_id'] == 0
    assert native_env.episode_metadata()['image_orientation'] == 'raw_robosuite_opengl'


def test_invalid_actions_do_not_advance_any_native_state(native_env) -> None:
    native_env.reset(seed=42, init_state_index=0)
    before = native_env._env.get_sim_state().copy()
    native_steps = native_env._env.env.timestep
    invalid = [np.zeros(6), np.zeros(8), np.zeros((1, 7)), np.array([]),
               np.array([np.nan, 0, 0, 0, 0, 0, -1]),
               np.array([0, np.inf, 0, 0, 0, 0, -1]),
               np.array([1.1, 0, 0, 0, 0, 0, -1]),
               np.array([0, 0, 0, 0, 0, 0, -1.1])]
    for action in invalid:
        with pytest.raises(ValueError, match='seven|normalized'):
            native_env.step(action)
        np.testing.assert_array_equal(native_env._env.get_sim_state(), before)
        assert native_env._env.env.timestep == native_steps and native_env.steps == 0


def test_timeout_counts_policy_steps_and_preserves_terminal_observation(native_env) -> None:
    observation, info = native_env.reset(seed=42, init_state_index=0)
    assert not info['success']
    initial_native_time = info['native_simulation_time']
    for step in range(1, 4):
        observation, reward, terminated, truncated, info = native_env.step(NO_OP)
        assert np.isfinite(reward)
        assert not terminated and not info['success']
        assert truncated == (step == 3)
        assert info['steps'] == step
        assert info['simulation_time'] == pytest.approx(step / 20)
        assert info['native_simulation_time'] - initial_native_time == pytest.approx(step / 20)
        # A reset would drop the raw timestep back to its ten settle steps.
        assert native_env._env.env.timestep == native_env.settle_steps + step
    raw = native_env._env.env._get_observations()
    np.testing.assert_array_equal(observation['joint_positions'], raw['robot0_joint_pos'])
    np.testing.assert_array_equal(observation['agentview_image'], raw['agentview_image'])
    with pytest.raises(RuntimeError, match='reset'):
        native_env.step(NO_OP)


def test_explicit_seed_and_initial_state_reproduce_state_rgb_and_actions(native_env) -> None:
    index = min(3, native_env.initial_state_count - 1)
    actions = [np.array([.05, 0, 0, 0, 0, 0, -1]), NO_OP.copy(), NO_OP.copy()]

    def trajectory():
        observation, info = native_env.reset(seed=53, init_state_index=index)
        assert info['init_state_index'] == index
        states = [observation]
        endings = []
        for action in actions:
            observation, reward, terminated, truncated, info = native_env.step(action)
            states.append(observation)
            endings.append((reward, terminated, truncated))
        return states, endings

    first, first_endings = trajectory()
    native_env.reset(seed=99, init_state_index=0)
    second, second_endings = trajectory()
    assert first_endings == second_endings
    for left, right in zip(first, second):
        assert left['language'] == right['language']
        for key in ('robot_state', 'joint_positions', 'agentview_image', 'wrist_image'):
            np.testing.assert_array_equal(left[key], right[key])


def test_invalid_reset_inputs_do_not_change_current_episode(native_env) -> None:
    native_env.reset(seed=42, init_state_index=0)
    before = native_env._env.get_sim_state().copy()
    for options in ({'seed': -1}, {'seed': True}, {'seed': 1.5},
                    {'init_state_index': -1}, {'init_state_index': native_env.initial_state_count},
                    {'init_state_index': True}):
        with pytest.raises(ValueError, match='seed|init_state_index'):
            native_env.reset(**options)
        np.testing.assert_array_equal(native_env._env.get_sim_state(), before)


def test_real_native_close_is_called_exactly_once(monkeypatch) -> None:
    env = LiberoEnv(suite='libero_spatial', task_id=0, image_size=IMAGE_SIZE, max_episode_steps=1)
    calls = 0
    try:
        env.reset(seed=42)
        native = env._env
        actual_close = native.close

        def close_once():
            nonlocal calls
            calls += 1
            actual_close()

        monkeypatch.setattr(native, 'close', close_once)
        env.close()
        env.close()
        assert calls == 1 and env._env is None
        with pytest.raises(RuntimeError, match='closed'):
            env.reset(seed=42)
        with pytest.raises(RuntimeError, match='reset'):
            env.step(NO_OP)
    finally:
        env.close()


def test_real_offscreen_environment_without_display(tmp_path) -> None:
    process_env = os.environ.copy()
    process_env.pop('DISPLAY', None)
    process_env.pop('WAYLAND_DISPLAY', None)
    process_env['LIBERO_CONFIG_PATH'] = str(tmp_path / 'libero-config')
    process_env['HF_HUB_OFFLINE'] = '1'  # The pinned local asset link must suffice.
    process_env['PYTHONDONTWRITEBYTECODE'] = '1'
    code = '''import numpy as np
from embodiedlab.envs.libero_env import LiberoEnv
with LiberoEnv(suite="libero_object", task_id=0, image_size=64, max_episode_steps=1) as env:
    obs, info = env.reset(seed=42, init_state_index=0)
    obs, reward, terminated, truncated, info = env.step(np.array([0.,0.,0.,0.,0.,0.,-1.]))
    assert truncated and not terminated and not info["success"]
    assert obs["agentview_image"].shape == (64,64,3)
    assert obs["wrist_image"].dtype == np.uint8
    assert np.ptp(obs["agentview_image"]) > 0
print("real headless LIBERO completed")
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-c', code], cwd=root, env=process_env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert 'real headless LIBERO completed' in result.stdout
