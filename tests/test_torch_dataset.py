"""Pinned expert-data alignment and the shared offline/online Torch conversion."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from embodiedlab.data.torch_dataset import (
    LiberoTorchDataset,
    preprocess_observation,
    split_episodes,
)


OFFICIAL = Path(__file__).resolve().parents[1] / 'data/libero/libero_goal/turn_on_the_stove_demo.hdf5'
CAMERAS = {'agentview_image': 'agentview_rgb', 'wrist_image': 'eye_in_hand_rgb'}
SAMPLE_KEYS = {*CAMERAS, 'robot_state', 'action'}


def assert_source_pair(sample: dict, group: h5py.Group, action_index: int) -> None:
    """Compare a sample directly with its original pre-action frame and command."""
    assert set(sample) == SAMPLE_KEYS
    observation_index = action_index - 1
    expected_state = np.r_[group['obs/ee_states'][observation_index],
                           group['obs/gripper_states'][observation_index]].astype(np.float32)
    np.testing.assert_array_equal(sample['robot_state'].numpy(), expected_state)
    np.testing.assert_array_equal(sample['action'].numpy(), group['actions'][action_index].astype(np.float32))
    for public_key, stored_key in CAMERAS.items():
        tensor = sample[public_key]
        source = group[f'obs/{stored_key}'][observation_index]
        assert tensor.shape == (3, *source.shape[:2])
        # Convert back to HWC for the comparison, preserving every pixel and channel.
        np.testing.assert_array_equal(tensor.numpy().transpose(1, 2, 0),
                                      source.astype(np.float32) / 255.)
        assert 0. <= tensor.min().item() <= tensor.max().item() <= 1.
    for value in sample.values():
        assert isinstance(value, torch.Tensor) and value.dtype == torch.float32
        assert value.device.type == 'cpu' and torch.isfinite(value).all()


def test_real_all_episode_length_and_metadata() -> None:
    with LiberoTorchDataset(OFFICIAL, suite='libero_goal', task_id=7) as dataset:
        assert dataset.episodes == tuple(f'demo_{index}' for index in range(50))
        assert len(dataset) == 4410
        assert dataset.episode_lengths[0] == 80 and dataset.episode_lengths[-1] == 90
        assert dataset.sample_counts == tuple(length - 1 for length in dataset.episode_lengths)
        assert dataset.language == 'turn on the stove'
        assert dataset.metadata['observation_alignment'] == 'post_action'
        assert dataset.metadata['provenance']['repo_id'] == 'yifengzhu-hf/LIBERO-datasets'
        assert dataset.locate(0) == ('demo_0', 1)
        assert dataset.locate(len(dataset) - 1) == ('demo_49', 89)
        with h5py.File(OFFICIAL, 'r') as source:
            assert_source_pair(dataset[0], source['data/demo_0'], 1)
            assert_source_pair(dataset[len(dataset) - 1], source['data/demo_49'], 89)


def test_real_first_last_and_episode_boundary_are_shifted_exactly() -> None:
    with LiberoTorchDataset(OFFICIAL, episodes=('demo_0', 'demo_1')) as dataset:
        assert len(dataset) == 174
        assert dataset.episode_lengths == (80, 96)
        assert dataset.sample_counts == (79, 95)
        # These four positions catch accidental cross-episode shifting and both ends.
        positions = [(0, 'demo_0', 1), (78, 'demo_0', 79),
                     (79, 'demo_1', 1), (173, 'demo_1', 95)]
        with h5py.File(OFFICIAL, 'r') as source:
            for index, name, action_index in positions:
                assert dataset.locate(index) == (name, action_index)
                assert_source_pair(dataset[index], source[f'data/{name}'], action_index)


def test_real_rgb_stays_lazy_and_samples_batch_normally(monkeypatch) -> None:
    original = h5py.Dataset.__getitem__
    camera_reads = []

    def guarded_read(dataset, selection):
        if dataset.name.endswith(('_rgb', '_image')):
            first_selection = selection[0] if isinstance(selection, tuple) and selection else selection
            full_slice = isinstance(first_selection, slice) and first_selection == slice(None)
            if selection is Ellipsis or selection == () or full_slice:
                raise AssertionError('A Torch sample must not load an entire RGB episode')
            camera_reads.append(dataset.name)
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded_read)
    with LiberoTorchDataset(OFFICIAL, episodes=('demo_0', 'demo_1')) as dataset:
        assert not camera_reads  # Initialization may read metadata and small state arrays.
        batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
        assert set(batch) == SAMPLE_KEYS
        assert batch['robot_state'].shape == (4, 8) and batch['action'].shape == (4, 7)
        assert batch['agentview_image'].shape == (4, 3, 128, 128)
        assert batch['wrist_image'].shape == (4, 3, 128, 128)
        assert all(value.dtype == torch.float32 for value in batch.values())
        assert camera_reads
        with h5py.File(OFFICIAL, 'r') as source:
            first_sample = {key: value[0] for key, value in batch.items()}
            assert_source_pair(first_sample, source['data/demo_0'], 1)


def test_preprocessing_keeps_pixels_direction_channels_and_inputs_unchanged() -> None:
    pixels = np.array([[[0, 11, 255], [23, 47, 91], [181, 201, 221]],
                       [[249, 87, 31], [64, 128, 192], [2, 7, 19]]], dtype=np.uint8)
    observation = {'agentview_image': pixels.copy(), 'wrist_image': pixels[:, ::-1].copy(),
                   'robot_state': np.arange(8, dtype=np.float64) / 10.,
                   'joint_positions': np.zeros(7), 'language': 'turn on the stove'}
    before = {key: value.copy() for key, value in observation.items() if isinstance(value, np.ndarray)}
    result = preprocess_observation(observation)
    assert set(result) == {*CAMERAS, 'robot_state'}
    for key in CAMERAS:
        assert result[key].shape == (3, 2, 3) and result[key].dtype == torch.float32
        np.testing.assert_array_equal(result[key].numpy().transpose(1, 2, 0),
                                      before[key].astype(np.float32) / 255.)
    np.testing.assert_array_equal(result['robot_state'].numpy(), before['robot_state'].astype(np.float32))
    assert result['robot_state'].dtype == torch.float32
    for key, value in before.items():
        np.testing.assert_array_equal(observation[key], value)
    # Converted arrays also belong to the caller; editing them cannot alter the input.
    result['agentview_image'][0, 0, 0] = 1.
    result['robot_state'][0] = -1.
    np.testing.assert_array_equal(observation['agentview_image'], before['agentview_image'])
    np.testing.assert_array_equal(observation['robot_state'], before['robot_state'])


def test_split_uses_whole_episodes_and_is_reproducible() -> None:
    names = tuple(f'demo_{index}' for index in range(50))
    train, validation = split_episodes(names, seed=42, train_fraction=.8)
    assert isinstance(train, tuple) and isinstance(validation, tuple)
    assert len(train) == 40 and len(validation) == 10
    assert len(set(train)) == 40 and len(set(validation)) == 10
    assert set(train).isdisjoint(validation) and set(train) | set(validation) == set(names)
    assert (train, validation) == split_episodes(names, seed=42, train_fraction=.8)
    assert (train, validation) != split_episodes(names, seed=43, train_fraction=.8)


@pytest.mark.parametrize('names, options', [
    ((), {}), (('demo_0',), {}), (('demo_0', 'demo_0'), {}),
    (('demo_0', 'demo_1'), {'train_fraction': 0.}),
    (('demo_0', 'demo_1'), {'train_fraction': 1.}),
    (('demo_0', 'demo_1'), {'train_fraction': np.nan}),
    (('demo_0', 'demo_1'), {'seed': -1}),
])
def test_invalid_split_settings_are_rejected(names, options) -> None:
    with pytest.raises(ValueError):
        split_episodes(names, **options)


@pytest.mark.parametrize('index', [-1, 79, 80, 1.5, True])
def test_invalid_indices_are_rejected(index) -> None:
    with LiberoTorchDataset(OFFICIAL, episodes=('demo_0',)) as dataset:
        with pytest.raises((IndexError, TypeError, ValueError)):
            dataset[index]
        with pytest.raises((IndexError, TypeError, ValueError)):
            dataset.locate(index)


def test_worker_context_is_explicitly_rejected() -> None:
    with LiberoTorchDataset(OFFICIAL, episodes=('demo_0',)) as dataset:
        iterator = iter(DataLoader(dataset, batch_size=1, num_workers=1))
        try:
            with pytest.raises(RuntimeError, match='num_workers|worker'):
                next(iterator)
        finally:
            # Torch's multiprocessing iterator has no public close() API.
            iterator._shutdown_workers()


def test_context_manager_and_constructor_errors_release_real_hdf5_handles(monkeypatch, tmp_path) -> None:
    original = h5py.File
    handles = []

    def track_file(*args, **kwargs):
        handle = original(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(h5py, 'File', track_file)
    with pytest.raises(RuntimeError, match='caller failed'):
        with LiberoTorchDataset(OFFICIAL, episodes=('demo_0', 'demo_1')) as dataset:
            dataset[0]
            dataset[79]
            raise RuntimeError('caller failed')
    assert handles and all(not handle.id.valid for handle in handles)
    dataset.close()  # Repeated close must also be safe.
    with pytest.raises(RuntimeError, match='closed'):
        dataset[0]
    with pytest.raises(ValueError):
        LiberoTorchDataset(OFFICIAL, episodes=('demo_0', 'demo_missing'))
    assert all(not handle.id.valid for handle in handles)
    unverified = tmp_path / 'unverified.hdf5'
    write_tiny_official_format(unverified, 3, provenance=False)
    with pytest.raises(ValueError, match='unverified|source'):
        LiberoTorchDataset(unverified)
    assert all(not handle.id.valid for handle in handles)


def write_tiny_official_format(path: Path, length: int, *, provenance: bool = True) -> None:
    """Small format/error fixture only; it is never used as a real expert-data result."""
    with h5py.File(OFFICIAL, 'r') as source, h5py.File(path, 'w') as file:
        data = file.create_group('data')
        for key, value in source['data'].attrs.items():
            data.attrs[key] = value
        group = data.create_group('demo_0')
        group.create_dataset('actions', data=np.zeros((length, 7), dtype=np.float32))
        group.create_dataset('rewards', data=np.zeros(length, dtype=np.float32))
        group.create_dataset('dones', data=np.zeros(length, dtype=np.uint8))
        group.create_dataset('states', data=np.zeros((length, 79), dtype=np.float64))
        obs = group.create_group('obs')
        for key, dimension in [('ee_states', 6), ('gripper_states', 2), ('joint_states', 7)]:
            obs.create_dataset(key, data=np.zeros((length, dimension), dtype=np.float64))
        for key in CAMERAS.values():
            obs.create_dataset(key, data=np.zeros((length, 128, 128, 3), dtype=np.uint8))
    if provenance:
        manifest = json.loads(OFFICIAL.with_suffix('.source.json').read_text())
        path.with_suffix('.source.json').write_text(json.dumps(manifest))


def test_episode_without_an_available_shifted_pair_is_rejected(tmp_path) -> None:
    path = tmp_path / 'one_step.hdf5'
    write_tiny_official_format(path, 1)
    with pytest.raises(ValueError, match='two|2|short|sample|step'):
        LiberoTorchDataset(path)


@pytest.mark.parametrize('damage', ['missing_provenance', 'wrong_revision', 'wrong_image_convention', 'broken_format'])
def test_unknown_source_or_wrong_format_is_rejected(tmp_path, damage) -> None:
    path = tmp_path / f'{damage}.hdf5'
    write_tiny_official_format(path, 3, provenance=damage != 'missing_provenance')
    if damage == 'wrong_revision':
        manifest = json.loads(path.with_suffix('.source.json').read_text())
        manifest['revision'] = 'unverified'
        path.with_suffix('.source.json').write_text(json.dumps(manifest))
    elif damage == 'wrong_image_convention':
        with h5py.File(path, 'r+') as file:
            file['data'].attrs['macros_image_convention'] = 'opencv'
    elif damage == 'broken_format':
        with h5py.File(path, 'r+') as file:
            del file['data/demo_0/obs/ee_states']
    message = 'ee_states' if damage == 'broken_format' else 'unverified|source'
    with pytest.raises(ValueError, match=message):
        LiberoTorchDataset(path)
