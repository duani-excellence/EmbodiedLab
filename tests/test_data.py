"""HDF5 format units, the actual official source, and real native recording replay."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from embodiedlab.data.episode import EpisodeDataset, EpisodeRecorder
from embodiedlab.data.replay import replay_episode


OFFICIAL = Path(__file__).resolve().parents[1] / 'data/libero/libero_goal/turn_on_the_stove_demo.hdf5'


def format_observation(index: int = 0) -> dict:
    """Array fixture only; these values do not claim a simulated task outcome."""
    pixels = np.full((16, 16, 3), index, dtype=np.uint8)
    return {'robot_state': np.arange(8, dtype=np.float64) * .01,
            'joint_positions': np.zeros(7, dtype=np.float64),
            'agentview_image': pixels, 'wrist_image': pixels.copy(),
            'language': 'turn on the stove'}


@pytest.fixture
def recorded_file(tmp_path) -> Path:
    metadata = {'suite': 'libero_goal', 'task_id': 7, 'task_name': 'turn_on_the_stove',
                'image_size': 16, 'max_episode_steps': 2, 'init_state_index': 0,
                'settle_steps': 10, 'control_dt': .05, 'control_mode': 'osc_pose_delta_normalized'}
    path = tmp_path / 'recorded.hdf5'
    recorder = EpisodeRecorder(path, metadata)
    recorder.start(format_observation(), {'simulation_time': 0.}, seed=42)
    for index in range(2):
        recorder.append(np.zeros(7), format_observation(index + 1), float(index),
                        False, index == 1, {'simulation_time': (index + 1) * .05,
                                           'success': False, 'failure_reason': 'timeout' if index == 1 else ''})
    recorder.close()
    return path


def test_recorded_lazy_read_alignment_and_chunks(recorded_file) -> None:
    with EpisodeDataset.load(recorded_file) as episode:
        assert len(episode) == 2 and episode.task_id == 7
        assert episode.metadata['complete'] is True
        assert episode.metadata['observation_alignment'] == 'pre_action'
        assert episode.metadata['image_orientation'] == 'raw_robosuite_opengl'
        assert episode.actions.shape == (2, 7)
        assert episode.timestamps.shape == (3,)
        np.testing.assert_array_equal(episode.observation(2)['agentview_image'], np.full((16, 16, 3), 2, dtype=np.uint8))
        transitions = list(episode.transitions())
        assert len(transitions) == 2
        assert transitions[0]['observation']['agentview_image'][0, 0, 0] == 0
        assert transitions[0]['next_observation']['agentview_image'][0, 0, 0] == 1
        assert transitions[1]['truncated'] is True
        assert episode.environment_config['init_state_index'] == 0
        with pytest.raises(IndexError): episode.step(2)
        with pytest.raises(ValueError, match='already align'): next(episode.shifted_bc_pairs())
    with h5py.File(recorded_file, 'r') as file:
        camera = file['data/demo_0/obs/agentview_image']
        assert camera.chunks == (1, 16, 16, 3) and camera.compression == 'gzip'


def test_incomplete_recording_and_overwrite_are_rejected(tmp_path, recorded_file) -> None:
    recorder = EpisodeRecorder(tmp_path / 'incomplete.hdf5', {'control_dt': .05})
    recorder.start(format_observation(), {'simulation_time': 0.}, seed=42)
    recorder.close()
    with pytest.raises(ValueError, match='incomplete'):
        EpisodeDataset.load(tmp_path / 'incomplete.hdf5')
    with pytest.raises(FileExistsError):
        EpisodeRecorder(recorded_file).start(format_observation(), {'simulation_time': 0.}, seed=42)


@pytest.mark.parametrize('damage, message', [
    ('nan_action', 'finite'), ('out_of_range_action', 'Normalized'),
    ('nan_state', 'finite'), ('short_image', 'observations'),
    ('bad_time', 'increasing'), ('wrong_interval', 'control_dt'),
    ('bad_version', 'format'), ('bad_success', 'Success'),
    ('false_success', 'Success'), ('bad_flag_type', 'boolean'),
])
def test_invalid_recorded_file_is_rejected(recorded_file, damage, message) -> None:
    with h5py.File(recorded_file, 'r+') as file:
        group = file['data/demo_0']
        if damage == 'nan_action': group['actions'][0, 0] = np.nan
        elif damage == 'out_of_range_action': group['actions'][0, 0] = 1.01
        elif damage == 'nan_state': group['obs/robot_state'][0, 0] = np.nan
        elif damage == 'short_image': group['obs/agentview_image'].resize(2, axis=0)
        elif damage == 'bad_time': group['timestamps'][1] = 0.
        elif damage == 'wrong_interval': group['timestamps'][1] = .03
        elif damage == 'false_success':
            group['terminated'][-1] = True
            group['truncated'][-1] = False
        elif damage == 'bad_flag_type':
            values = group['terminated'][()].astype(np.uint8)
            del group['terminated']
            group.create_dataset('terminated', data=values)
        else:
            metadata = json.loads(group.attrs['metadata_json'])
            if damage == 'bad_version': metadata['format_version'] = 1.0
            else: metadata['success'] = True
            group.attrs['metadata_json'] = json.dumps(metadata)
    with pytest.raises(ValueError, match=message):
        EpisodeDataset.load(recorded_file)


def test_actual_official_demo_shapes_raw_pixels_and_shifted_pairs() -> None:
    if not OFFICIAL.exists():
        pytest.skip('Run the documented single-task download first')
    with EpisodeDataset.load(OFFICIAL, task_id=7, suite='libero_goal') as episode:
        assert len(episode) == 80
        assert len(EpisodeDataset.list_episodes(OFFICIAL)) == 50
        assert episode.metadata['observation_alignment'] == 'post_action'
        assert episode.language == 'turn on the stove'
        assert episode.task_id == 7 and episode.initial_state.shape == (79,)
        observation = episode.observation(0)
        with h5py.File(OFFICIAL, 'r') as file:
            group = file['data/demo_0']
            np.testing.assert_array_equal(observation['robot_state'], np.r_[group['obs/ee_states'][0], group['obs/gripper_states'][0]])
            np.testing.assert_array_equal(observation['agentview_image'], group['obs/agentview_rgb'][0])
            assert group['robot_states'].shape == (80, 9)
            # Verified source model qpos order begins with robot0_joint1..7:
            # stored obs[t] matches the next pre-action simulator state exactly.
            np.testing.assert_array_equal(group['obs/joint_states'][:-1], group['states'][1:, 1:8])
            assert np.max(np.abs(group['obs/joint_states'][()] - group['states'][:, 1:8])) > .01
        pairs = list(episode.shifted_bc_pairs())
        assert len(pairs) == 79
        np.testing.assert_array_equal(pairs[0][1], episode.actions[1])
        np.testing.assert_array_equal(pairs[0][0]['robot_state'], observation['robot_state'])
        with pytest.raises(ValueError, match='post-action'): next(episode.transitions())
        with pytest.raises(ValueError, match='source-state'): replay_episode(None, episode)


def test_validation_never_reads_whole_camera_arrays(monkeypatch) -> None:
    if not OFFICIAL.exists():
        pytest.skip('Run the documented single-task download first')
    original = h5py.Dataset.__getitem__

    def checked_getitem(dataset, selection):
        if dataset.name.endswith(('_rgb', '_image')) and selection == ():
            raise AssertionError('Whole-episode camera read')
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, '__getitem__', checked_getitem)
    with EpisodeDataset.load(OFFICIAL, task_id=7) as episode:
        assert episode.observation(0)['agentview_image'].shape == (128, 128, 3)


def test_real_native_recording_and_action_replay(tmp_path) -> None:
    from embodiedlab.envs.libero_env import LiberoEnv

    path = tmp_path / 'actual_native.hdf5'
    with LiberoEnv(suite='libero_goal', task_id=7, image_size=32,
                   max_episode_steps=2, settle_steps=2) as env:
        observation, info = env.reset(seed=42, init_state_index=0)
        recorder = EpisodeRecorder(path, env.episode_metadata())
        recorder.start(observation, info, seed=42)
        while True:
            action = np.array([0., 0., 0., 0., 0., 0., -1.])
            observation, reward, terminated, truncated, info = env.step(action)
            recorder.append(action, observation, reward, terminated, truncated, info)
            if terminated or truncated: break
        recorder.close()
    with EpisodeDataset.load(path) as episode:
        assert len(episode) == 2
        assert episode.metadata['success'] is False
        with LiberoEnv(**episode.environment_config) as env:
            result = replay_episode(env, episode)
        assert result.matches and result.length_match
        assert result.max_joint_position_deviation < 1e-6
        assert result.max_rgb_pixel_deviation == 0
        assert result.success_match and result.terminated_match and result.truncated_match
