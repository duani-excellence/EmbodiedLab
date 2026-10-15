"""Versioned, explicit task and I/O metadata for the unchanged visual_bc_v1."""
from copy import deepcopy
import re
from typing import Sequence
import warnings


CHECKPOINT_VERSION = 2
IMAGE_PREPROCESSING = 'uint8_RGB_HWC -> float32_RGB_CHW / 255; no spatial transform'
TRAINING_CONTRACT = {
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

# Only this historical source was used before checkpoints stored task identity.
# All identity fields and both SHA256 copies must agree; local filenames do not
# establish identity. New tasks must use the explicit version-2 contract.
LEGACY_SOURCE = {
    'repo_id': 'yifengzhu-hf/LIBERO-datasets',
    'revision': '147fa5eb607bb696dd922c3518084a90954cd2f3',
    'path': 'libero_goal/turn_on_the_stove_demo.hdf5',
    'sha256': '387fc10747696b80dea6ed8d7f2beaa162bf92ae11750b241073cbd33aac73d5',
    'size': 447509922,
}
SOURCE_FINGERPRINT_KEYS = ('repo_id', 'revision', 'path', 'sha256', 'size')
LEGACY_TASK = {'suite': 'libero_goal', 'task_id': 7, 'task_name': 'turn_on_the_stove'}


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f'Checkpoint {name} must be a dictionary')
    return value


def validate_contract(contract: dict) -> dict:
    """Validate supported task/source/I/O/split metadata and return an owned copy."""
    contract = deepcopy(_mapping(contract, 'contract'))
    if type(contract.get('contract_version')) is not int or contract['contract_version'] != 1:
        raise ValueError('Checkpoint contract_version must be 1')
    if contract.get('architecture') != 'visual_bc_v1':
        raise ValueError('Checkpoint architecture must be visual_bc_v1')
    task = _mapping(contract.get('task'), 'task')
    source = _mapping(contract.get('source'), 'source')
    for key in ('suite', 'task_name'):
        if not isinstance(task.get(key), str) or not task[key]:
            raise ValueError(f'Checkpoint task.{key} must be a nonempty string')
    if type(task.get('task_id')) is not int or task['task_id'] < 0:
        raise ValueError('Checkpoint task.task_id must be a nonnegative integer')
    if (type(source.get('task_id')) is not int
            or any(source.get(key) != task[key] for key in ('suite', 'task_id', 'task_name'))):
        raise ValueError('Checkpoint task identity disagrees with its verified source')
    for key in ('repo_id', 'path', 'language', 'audit_evidence'):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f'Checkpoint source.{key} must be a nonempty string')
    if not isinstance(source.get('revision'), str) or not re.fullmatch(r'[0-9a-f]{40}', source['revision']):
        raise ValueError('Checkpoint source.revision must be a pinned 40-character commit')
    if not isinstance(source.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', source['sha256']):
        raise ValueError('Checkpoint source.sha256 must be a 64-character SHA256')
    if (source['sha256'] == LEGACY_SOURCE['sha256']
            and any(task[key] != value for key, value in LEGACY_TASK.items())):
        raise ValueError('Checkpoint source SHA256 identifies the stove task; task identity conflicts')
    if type(source.get('size')) is not int or source['size'] <= 0:
        raise ValueError('Checkpoint source.size must be a positive integer')
    if type(source.get('task_order_index')) is not int or source['task_order_index'] < 0:
        raise ValueError('Checkpoint source.task_order_index must be a nonnegative integer')
    training = _mapping(contract.get('training'), 'training')
    expected = {**TRAINING_CONTRACT, 'image_preprocessing': IMAGE_PREPROCESSING,
                'gripper_classes': {'0': -1, '1': 1}, 'control_dt': .05}
    for key, value in expected.items():
        if training.get(key) != value:
            raise ValueError(f'Checkpoint training.{key} is unsupported: expected {value!r}')
    split = _mapping(contract.get('split'), 'split')
    if type(split.get('seed')) is not int or split['seed'] < 0:
        raise ValueError('Checkpoint split.seed must be a nonnegative integer')
    for key in ('train_episodes', 'validation_episodes'):
        names = split.get(key)
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError(f'Checkpoint split.{key} must contain unique nonempty episode names')
    if set(split['train_episodes']) & set(split['validation_episodes']):
        raise ValueError('Checkpoint train/validation episode splits overlap')
    return contract


def make_contract(metadata: dict, *, train_episodes: Sequence[str],
                  validation_episodes: Sequence[str], split_seed: int = 42) -> dict:
    """Build checkpoint metadata from a verified dataset and whole-episode split."""
    metadata = _mapping(metadata, 'dataset metadata')
    if isinstance(train_episodes, (str, bytes)) or isinstance(validation_episodes, (str, bytes)):
        raise ValueError('Checkpoint episode splits must be sequences of names, not a string')
    source = deepcopy(_mapping(metadata.get('verified_source'), 'verified_source'))
    training = deepcopy(_mapping(metadata.get('training_contract'), 'training_contract'))
    if metadata.get('control_freq', 20) != 20:
        raise ValueError('Checkpoint dataset control_freq must be 20 Hz')
    training.update(image_preprocessing=IMAGE_PREPROCESSING,
                    gripper_classes={'0': -1, '1': 1}, control_dt=.05)
    return validate_contract({
        'contract_version': 1, 'architecture': 'visual_bc_v1',
        'task': {key: source.get(key) for key in ('suite', 'task_id', 'task_name')},
        'source': source, 'training': training,
        'split': {'seed': split_seed, 'train_episodes': list(train_episodes),
                  'validation_episodes': list(validation_episodes)},
    })


def read_contract(saved: dict) -> dict:
    """Read version 2, or explicitly identify the one verified legacy source."""
    saved = _mapping(saved, 'payload')
    config = _mapping(saved.get('config'), 'config')
    version = saved.get('format_version')
    if type(version) is not int or version not in (1, CHECKPOINT_VERSION):
        raise ValueError('Checkpoint format_version must be 1 or 2')
    if version == CHECKPOINT_VERSION:
        contract = validate_contract(saved.get('contract'))
        # Config is retained for experiment reporting, so contradictory copies
        # must fail rather than letting reports and inference name different data.
        source = contract['source']
        if 'contract' in config and config['contract'] != contract:
            raise ValueError('Checkpoint config contract disagrees with top-level contract')
        if config.get('dataset_sha256', source['sha256']) != source['sha256']:
            raise ValueError('Checkpoint config dataset_sha256 disagrees with contract')
        provenance = _mapping(config.get('provenance', source), 'config.provenance')
        if any(provenance.get(key) != source[key] for key in SOURCE_FINGERPRINT_KEYS):
            raise ValueError('Checkpoint config provenance disagrees with contract')
        for key, value in contract['task'].items():
            if key in config and config[key] != value:
                raise ValueError(f'Checkpoint config {key} disagrees with contract')
        conventions = {'architecture': 'visual_bc_v1', 'alignment': 'obs[t-1] -> action[t]',
                       'images': 'unchanged RGB/CHW float32 [0,1]', 'gripper_classes': [-1, 1]}
        for key, value in conventions.items():
            if key in config and config[key] != value:
                raise ValueError(f'Checkpoint config {key} disagrees with contract')
        for key in ('train_episodes', 'validation_episodes'):
            if key in config and config[key] != contract['split'][key]:
                raise ValueError(f'Checkpoint config {key} disagrees with contract')
        if config.get('split_seed') is not None and config['split_seed'] != contract['split']['seed']:
            raise ValueError('Checkpoint config split_seed disagrees with contract')
        return contract
    expected = {'architecture': 'visual_bc_v1', 'alignment': 'obs[t-1] -> action[t]',
                'images': 'unchanged RGB/CHW float32 [0,1]', 'gripper_classes': [-1, 1]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Checkpoint legacy model, alignment, RGB or gripper convention is unsupported')
    provenance = _mapping(config.get('provenance'), 'legacy provenance')
    if (any(provenance.get(key) != value for key, value in LEGACY_SOURCE.items())
            or config.get('dataset_sha256') != LEGACY_SOURCE['sha256']):
        raise ValueError('Checkpoint legacy task identity is unknown; an explicit version-2 contract is required')
    source = {**LEGACY_SOURCE, **LEGACY_TASK, 'task_order_index': 0,
              'language': 'turn on the stove',
              'audit_evidence': 'Phase 5B verified legacy provenance and dataset SHA256'}
    split_seed = config.get('split_seed')
    contract = make_contract({'verified_source': source, 'training_contract': TRAINING_CONTRACT},
                             train_episodes=config.get('train_episodes', ()),
                             validation_episodes=config.get('validation_episodes', ()),
                             split_seed=config.get('seed') if split_seed is None else split_seed)
    contract['legacy_identity_source'] = 'pinned_provenance_and_dataset_sha256'
    warnings.warn('Legacy checkpoint task identity explicitly mapped from pinned provenance '
                  'and dataset SHA256 to libero_goal/7/turn_on_the_stove; see policy.contract',
                  UserWarning, stacklevel=2)
    return contract
