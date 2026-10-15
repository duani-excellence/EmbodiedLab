"""Verified source identity, task contracts, and shifted lazy samples."""
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
import torch

from embodiedlab.data.torch_dataset import LiberoTorchDataset


DATA = Path(__file__).resolve().parents[1] / 'data/libero/libero_goal'
STOVE = DATA / 'turn_on_the_stove_demo.hdf5'
SOURCES = [
    pytest.param('turn_on_the_stove', 7, 'turn on the stove',
                 '387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5', id='stove'),
    pytest.param('open_the_middle_drawer_of_the_cabinet', 0,
                 'open the middle drawer of the cabinet',
                 '20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737', id='drawer'),
]
CAMERAS = {'agentview_image': 'agentview_rgb', 'wrist_image': 'eye_in_hand_rgb'}
CONTRACT = {
    'observation_alignment': 'post_action',
    'supervision_alignment': 'obs[t-1] -> action[t], t=1..T-1',
    'image_orientation': 'raw_robosuite_opengl',
    'image_shape': [3, 128, 128],
    'state_layout': ['eef_pos_world_m:3', 'axis_angle_rad:3', 'gripper_qpos_m:2'],
    'state_dim': 8,
    'action_dim': 7,
    'action_semantics': 'osc_pose_delta_normalized',
    'position_delta_scale_m': .05,
    'axis_angle_delta_scale_rad': .5,
    'gripper_values': [-1, 1],
}


@pytest.mark.parametrize('name,task_id,language,digest', SOURCES)
def test_real_source_infers_default_task_and_training_contract(name, task_id, language, digest) -> None:
    path = DATA / f'{name}_demo.hdf5'
    with LiberoTorchDataset(path) as dataset:
        assert dataset.metadata['suite'] == 'libero_goal'
        assert dataset.metadata['task_id'] == task_id
        assert dataset.metadata['task_name'] == name
        assert dataset.language == language
        source = dataset.metadata['verified_source']
        expected = {
            'repo_id': 'yifengzhu-hf/LIBERO-datasets',
            'revision': '147fa5eb607bb696dd922c3518084a90954cd2f3',
            'path': f'libero_goal/{name}_demo.hdf5',
            'sha256': digest,
            'size': path.stat().st_size,
            'suite': 'libero_goal',
            'task_id': task_id,
            'task_order_index': 0,
            'task_name': name,
            'language': language,
        }
        assert {key: source[key] for key in expected} == expected
        assert source['audit_evidence']
        assert source['observed_gripper_values'] == ([-1, 1] if task_id == 7 else [-1])
        assert dataset.metadata['training_contract'] == CONTRACT
        assert len(dataset.episodes) == 50
        assert dataset.sample_counts == tuple(length - 1 for length in dataset.episode_lengths)
        assert len(dataset) == sum(dataset.sample_counts)


@pytest.mark.parametrize('name,task_id,language,digest', SOURCES)
@pytest.mark.parametrize('wrong_field', ['suite', 'task_id'])
def test_explicit_task_must_match_the_verified_source(name, task_id, language, digest, wrong_field) -> None:
    options = {'suite': 'libero_goal', 'task_id': task_id}
    options[wrong_field] = 'libero_spatial' if wrong_field == 'suite' else (0 if task_id == 7 else 7)
    with pytest.raises(ValueError, match='suite|task|source|match'):
        LiberoTorchDataset(DATA / f'{name}_demo.hdf5', episodes=('demo_0',), **options)


@pytest.mark.parametrize('name,task_id,language,digest', SOURCES)
def test_both_tasks_keep_shifted_episode_boundaries_and_lazy_rgb(name, task_id, language, digest, monkeypatch) -> None:
    path = DATA / f'{name}_demo.hdf5'
    reads = []
    original = h5py.Dataset.__getitem__

    def read_one_frame(array, selection):
        if array.name.endswith(('/agentview_rgb', '/eye_in_hand_rgb')):
            # Initialization and numeric validation must not decode camera arrays.
            assert isinstance(selection, (int, np.integer)), 'RGB must be read one frame at a time'
            reads.append((array.name, int(selection)))
        return original(array, selection)

    monkeypatch.setattr(h5py.Dataset, '__getitem__', read_one_frame)
    with LiberoTorchDataset(path, episodes=('demo_0', 'demo_1')) as dataset:
        assert reads == []
        first_count, second_count = dataset.sample_counts
        positions = [(0, 'demo_0', 1), (first_count - 1, 'demo_0', first_count),
                     (first_count, 'demo_1', 1), (len(dataset) - 1, 'demo_1', second_count)]
        with h5py.File(path, 'r') as file:
            for position, episode, action_index in positions:
                assert dataset.locate(position) == (episode, action_index)
                sample = dataset[position]
                group = file[f'data/{episode}']
                observation_index = action_index - 1
                expected_state = np.r_[group['obs/ee_states'][observation_index],
                                       group['obs/gripper_states'][observation_index]].astype(np.float32)
                np.testing.assert_array_equal(sample['robot_state'].numpy(), expected_state)
                np.testing.assert_array_equal(sample['action'].numpy(), group['actions'][action_index].astype(np.float32))
                assert set(sample) == {*CAMERAS, 'robot_state', 'action'}
                for key, stored in CAMERAS.items():
                    source = group[f'obs/{stored}'][observation_index]
                    expected = source.transpose(2, 0, 1).astype(np.float32) / np.float32(255.)
                    np.testing.assert_array_equal(sample[key].numpy(), expected)
                    assert sample[key].shape == (3, 128, 128) and sample[key].is_contiguous()
                assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in sample.values())
        assert len(reads) == 16  # Four samples and four independent direct-source comparisons.


def write_small_official_shaped_file(path: Path) -> dict:
    """Create invalid source bytes with valid small HDF structure for rejection tests."""
    with h5py.File(STOVE, 'r') as source, h5py.File(path, 'w') as file:
        data = file.create_group('data')
        for key, value in source['data'].attrs.items():
            data.attrs[key] = value
        group = data.create_group('demo_0')
        group.create_dataset('actions', data=np.array([[0.] * 6 + [-1.], [0.] * 6 + [1.]]))
        group.create_dataset('rewards', data=np.zeros(2))
        group.create_dataset('dones', data=np.array([0, 1], dtype=np.uint8))
        group.create_dataset('states', data=np.zeros((2, 79)))
        obs = group.create_group('obs')
        for key, dimension in [('ee_states', 6), ('gripper_states', 2), ('joint_states', 7)]:
            obs.create_dataset(key, data=np.zeros((2, dimension)))
        for key in CAMERAS.values():
            obs.create_dataset(key, data=np.zeros((2, 128, 128, 3), dtype=np.uint8))
    provenance = json.loads(STOVE.with_suffix('.source.json').read_text())
    path.with_suffix('.source.json').write_text(json.dumps(provenance))
    return provenance


def test_copied_valid_sidecar_cannot_authenticate_a_small_fake_file(tmp_path) -> None:
    path = tmp_path / 'fake.hdf5'
    source = write_small_official_shaped_file(path)
    assert path.stat().st_size < source['size']
    with pytest.raises(ValueError, match='size|fingerprint|sha|source'):
        LiberoTorchDataset(path)


def test_same_size_fake_requires_the_real_file_sha256(tmp_path) -> None:
    path = tmp_path / 'same_size_fake.hdf5'
    source = write_small_official_shaped_file(path)
    # Sparse padding avoids copying or changing the original 447 MB source file.
    with path.open('r+b') as file:
        file.truncate(source['size'])
    assert path.stat().st_size == source['size']
    with pytest.raises(ValueError, match='SHA|sha|digest|fingerprint|source'):
        LiberoTorchDataset(path)


def test_sidecar_sha_and_size_cannot_be_forged_to_approve_fake_bytes(tmp_path) -> None:
    path = tmp_path / 'forged_sidecar.hdf5'
    source = write_small_official_shaped_file(path)
    source['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    source['size'] = path.stat().st_size
    path.with_suffix('.source.json').write_text(json.dumps(source))
    with pytest.raises(ValueError, match='verified|source|SHA|sha|fingerprint|size'):
        LiberoTorchDataset(path)


def test_unknown_identity_is_rejected_even_for_the_real_approved_file_bytes(tmp_path) -> None:
    path = tmp_path / 'unknown_identity.hdf5'
    # Authentic contents isolate identity rejection from size, hash and schema errors.
    shutil.copyfile(STOVE, path)
    source = json.loads(STOVE.with_suffix('.source.json').read_text())
    source['repo_id'] = 'unknown-source/not-a-verified-dataset'
    path.with_suffix('.source.json').write_text(json.dumps(source))
    with pytest.raises(ValueError, match='verified|unknown|source'):
        LiberoTorchDataset(path)
