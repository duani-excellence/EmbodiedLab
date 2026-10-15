"""Task identity and versioned I/O checks using the unchanged real E5 weights."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from embodiedlab.data.episode import EpisodeDataset
from embodiedlab.data.torch_dataset import LiberoTorchDataset, split_episodes
from embodiedlab.policies.bc import BCPolicy
from embodiedlab.policies.checkpoint import (
    CHECKPOINT_VERSION, LEGACY_SOURCE, make_contract, read_contract,
)


ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / 'training_results/phase5b3_train_seed42/best.pt'
OFFICIAL = ROOT / 'data/libero/libero_goal/turn_on_the_stove_demo.hdf5'


@pytest.fixture(scope='module', autouse=True)
def torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='module')
def legacy() -> dict:
    if not E5.is_file():
        pytest.skip('Optional legacy E5 checkpoint is not distributed; '
                    'restore training_results/phase5b3_train_seed42/best.pt for legacy compatibility tests')
    fingerprint = hashlib.sha256(E5.read_bytes()).hexdigest()
    yield torch.load(E5, map_location='cpu', weights_only=True)
    assert hashlib.sha256(E5.read_bytes()).hexdigest() == fingerprint


@pytest.fixture(scope='module')
def dataset_metadata() -> dict:
    with LiberoTorchDataset(OFFICIAL, episodes=('demo_0',)) as dataset:
        return deepcopy(dataset.metadata)


@pytest.fixture(scope='module')
def current(legacy, dataset_metadata) -> dict:
    saved = deepcopy(legacy)
    saved['format_version'] = CHECKPOINT_VERSION
    saved['contract'] = make_contract(
        dataset_metadata, train_episodes=saved['config']['train_episodes'],
        validation_episodes=saved['config']['validation_episodes'], split_seed=42)
    return saved


@pytest.fixture(scope='module')
def current_path(current, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp('checkpoint_contract') / 'converted_e5.pt'
    torch.save(current, path)
    return path


@pytest.fixture(scope='module')
def observation() -> dict:
    with EpisodeDataset.load(OFFICIAL, suite='libero_goal', task_id=7) as episode:
        return episode.observation(0)


def test_contract_owns_metadata_and_records_full_split(current, dataset_metadata) -> None:
    contract = read_contract(current)
    assert contract['task'] == {'suite': 'libero_goal', 'task_id': 7, 'task_name': 'turn_on_the_stove'}
    assert contract['source']['sha256'] == LEGACY_SOURCE['sha256']
    assert contract['training']['image_shape'] == [3, 128, 128]
    assert contract['training']['gripper_classes'] == {'0': -1, '1': 1}
    assert len(contract['split']['train_episodes']) == 40
    assert len(contract['split']['validation_episodes']) == 10
    assert not set(contract['split']['train_episodes']) & set(contract['split']['validation_episodes'])
    contract['source']['task_name'] = 'modified'
    contract['training']['image_shape'][1] = 1
    assert dataset_metadata['verified_source']['task_name'] == 'turn_on_the_stove'
    assert current['contract']['training']['image_shape'] == [3, 128, 128]


@pytest.mark.parametrize('task_id,name', [
    (7, 'turn_on_the_stove'), (0, 'open_the_middle_drawer_of_the_cabinet'),
])
def test_contract_uses_each_real_verified_dataset_without_task_defaults(task_id, name) -> None:
    path = OFFICIAL.with_name(name + '_demo.hdf5')
    train, validation = split_episodes(EpisodeDataset.list_episodes(path), seed=42)
    with LiberoTorchDataset(path, episodes=('demo_0',)) as dataset:
        contract = make_contract(dataset.metadata, train_episodes=train,
                                 validation_episodes=validation)
    assert contract['task'] == {'suite': 'libero_goal', 'task_id': task_id, 'task_name': name}
    assert contract['source']['sha256'] == dataset.metadata['provenance']['sha256']
    if task_id == 0:
        assert contract['source']['stored_task_name'] == 'open_the_middle_layer_of_the_drawer'
        assert contract['source']['observed_gripper_values'] == [-1]
        assert contract['training']['gripper_values'] == [-1, 1]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_version_two_loads_original_buffers_and_predictions(current_path, legacy, observation, device) -> None:
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not accessible in this process')
    with pytest.warns(UserWarning, match='Legacy checkpoint task identity explicitly mapped'):
        old = BCPolicy(E5, device=device)
    new = BCPolicy(current_path, device=device)
    assert (new.suite, new.task_id, new.task_name, new.image_size) == (
        'libero_goal', 7, 'turn_on_the_stove', 128)
    assert 'legacy_identity_source' not in new.contract
    assert old.contract['legacy_identity_source'] == 'pinned_provenance_and_dataset_sha256'
    for key, value in new.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), legacy['model_state_dict'][key], rtol=0, atol=0)
    np.testing.assert_array_equal(new.act(observation), old.act(observation))
    assert not new.model.training and not any(p.requires_grad for p in new.model.parameters())


@pytest.mark.parametrize('damage', ['unknown_source', 'missing_provenance', 'conflicting_sha'])
def test_legacy_identity_cannot_be_guessed_from_filename(legacy, tmp_path, damage) -> None:
    saved = deepcopy(legacy)
    if damage == 'unknown_source':
        saved['config']['provenance']['path'] = 'libero_goal/another_task_demo.hdf5'
    elif damage == 'missing_provenance':
        del saved['config']['provenance']
    else:
        saved['config']['dataset_sha256'] = '0' * 64
    path = tmp_path / 'turn_on_the_stove_best.pt'
    torch.save(saved, path)
    with pytest.raises(ValueError, match='Checkpoint.*(identity|provenance)'):
        BCPolicy(path, device='cpu')


@pytest.mark.parametrize('damage,match', [
    ('task_identity', 'identity'), ('hash', 'sha256'), ('preprocessing', 'image_preprocessing'),
    ('overlap', 'overlap'), ('architecture', 'architecture'), ('config_provenance', 'provenance'),
    ('config_split', 'train_episodes'), ('config_contract', 'config contract'),
])
def test_inconsistent_or_unsupported_contract_is_rejected(current, damage, match) -> None:
    saved = deepcopy(current)
    if damage == 'task_identity':
        saved['contract']['task']['task_id'] = 0
    elif damage == 'hash':
        saved['contract']['source']['sha256'] = 'not-a-hash'
    elif damage == 'preprocessing':
        saved['contract']['training']['image_preprocessing'] = 'flip_vertical'
    elif damage == 'overlap':
        saved['contract']['split']['validation_episodes'].append(
            saved['contract']['split']['train_episodes'][0])
    elif damage == 'architecture':
        saved['contract']['architecture'] = 'unknown_architecture'
    elif damage == 'config_provenance':
        saved['config']['provenance']['revision'] = '0' * 40
    elif damage == 'config_split':
        saved['config']['train_episodes'].reverse()
    else:
        saved['config']['contract'] = {**saved['contract'], 'architecture': 'contradiction'}
    with pytest.raises(ValueError, match=match):
        read_contract(saved)


def test_known_stove_hash_cannot_be_relabelled_as_another_task(dataset_metadata, legacy) -> None:
    metadata = deepcopy(dataset_metadata)
    source = metadata['verified_source']
    source.update(task_id=0, task_name='open_the_middle_drawer_of_the_cabinet',
                  path='libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5')
    with pytest.raises(ValueError, match='SHA256 identifies the stove task'):
        make_contract(metadata, train_episodes=legacy['config']['train_episodes'],
                      validation_episodes=legacy['config']['validation_episodes'])


class MetadataEnvironment:
    """Only the public metadata protocol is needed; compatibility does not reset."""

    def __init__(self) -> None:
        self.metadata = {'suite': 'libero_goal', 'task_id': 7, 'task_name': 'turn_on_the_stove',
                         'image_size': 128, 'image_orientation': 'raw_robosuite_opengl',
                         'control_mode': 'osc_pose_delta_normalized', 'action_dim': 7,
                         'control_dt': .05, 'position_delta_scale_m': .05,
                         'axis_angle_delta_scale_rad': .5, 'action_low': [-1.] * 7,
                         'action_high': [1.] * 7, 'gripper': '-1 open, +1 close'}

    def episode_metadata(self) -> dict:
        return self.metadata


def test_compatibility_checks_identity_images_and_physical_action_semantics(current_path) -> None:
    policy = BCPolicy(current_path, device='cpu')
    env = MetadataEnvironment()
    original = deepcopy(env.metadata)
    policy.assert_compatible(env)
    assert env.metadata == original
    for key, wrong in [('task_id', 0), ('task_name', 'another_task'), ('image_size', 64),
                       ('image_orientation', 'flipped'), ('position_delta_scale_m', .1),
                       ('control_dt', .1), ('gripper', '+1 open, -1 close')]:
        env.metadata = {**original, key: wrong}
        with pytest.raises(ValueError, match=key):
            policy.assert_compatible(env)
    assert policy.inference_times_ms == []


def test_malformed_checkpoint_and_ambiguous_split_fail_clearly(current, dataset_metadata) -> None:
    for saved in ([], {**current, 'format_version': True}, {**current, 'contract': None},
                  {**current, 'config': None}):
        with pytest.raises(ValueError, match='Checkpoint'):
            read_contract(saved)
    with pytest.raises(ValueError, match='not a string'):
        make_contract(dataset_metadata, train_episodes='demo_0', validation_episodes=['demo_1'])
