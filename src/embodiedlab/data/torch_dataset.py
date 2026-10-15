"""Lazy PyTorch supervision from explicitly audited official LIBERO sources."""
from bisect import bisect_right
from copy import deepcopy
import hashlib
from importlib.resources import files
from itertools import accumulate
import json
import operator
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from embodiedlab.data.episode import EpisodeDataset, IMAGE_KEYS, _validate_observation
from embodiedlab.interfaces import Observation

def _verified_source(path: Path, metadata: dict, *, suite: str | None,
                     task_id: int | None) -> tuple[dict, dict]:
    """Authenticate real file bytes once, independently of its editable sidecar."""
    manifest = json.loads(files('embodiedlab.data').joinpath('verified_sources.json').read_text())
    provenance = metadata.get('provenance', {})
    if not isinstance(provenance, dict):
        raise ValueError('Unverified source sidecar must contain a JSON object')
    identity = ('repo_id', 'revision', 'path')
    source = next((item for item in manifest['sources']
                   if all(provenance.get(key) == item[key] for key in identity)), None)
    if source is None or metadata['source_format'] != 'libero_official_hdf5':
        raise ValueError('Unverified source: use an audited official file with its pinned .source.json')
    if ((suite is not None and suite != source['suite'])
            or (task_id is not None and (type(task_id) is not int or task_id != source['task_id']))):
        raise ValueError('Requested suite/task does not match the verified source')
    if any(provenance.get(key) != source[key] for key in ('sha256', 'size')):
        raise ValueError('Unverified source sidecar SHA256/size does not match the audited fingerprint')
    if path.stat().st_size != source['size']:
        raise ValueError('Unverified source file size does not match the audited fingerprint')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != source['sha256']:
        raise ValueError('Unverified source file SHA256 does not match the audited fingerprint')
    return source, manifest['training_contract']


def _validate_source_episode(reader: EpisodeDataset, source: dict) -> None:
    """Check the audited schema and control semantics without decoding RGB."""
    metadata = reader.metadata
    if (metadata['task_name'] != source['stored_task_name']
            or metadata['language'] != source['language']
            or metadata['source_image_convention'] != 'opengl'
            or metadata['observation_alignment'] != 'post_action'):
        raise ValueError('Source task, language, camera convention or alignment is unverified')
    settings = metadata['env_args']['env_kwargs']
    controller = settings['controller_configs']
    signature = {'type': 'OSC_POSE', 'control_delta': True, 'input_min': -1, 'input_max': 1,
                 'output_min': [-.05, -.05, -.05, -.5, -.5, -.5],
                 'output_max': [.05, .05, .05, .5, .5, .5]}
    if (settings.get('robots') != ['Panda'] or settings.get('control_freq') != 20
            or any(controller.get(key) != value for key, value in signature.items())):
        raise ValueError('Source controller does not match the audited normalized OSC contract')
    for key in ('agentview_rgb', 'eye_in_hand_rgb'):
        if reader._obs[key].shape[1:] != (128, 128, 3):
            raise ValueError(f'Source {key} must have audited HWC shape (128,128,3)')
    if not set(np.unique(reader.actions[:, 6])).issubset(source['observed_gripper_values']):
        raise ValueError('Source gripper values do not match its audited sample support')
    # Historical env_args aliases are resolved only by an exact audited source SHA.
    metadata['source_task_name'] = metadata['task_name']
    metadata.update({key: source[key] for key in ('suite', 'task_id', 'task_order_index', 'task_name')})


def preprocess_observation(observation: Observation) -> dict[str, torch.Tensor]:
    """Convert offline or live raw RGB/state identically, without rotating pixels.

    Images: HWC uint8 RGB -> contiguous CHW float32 in [0,1].
    State: eef xyz (m), axis-angle (rad), gripper qpos (m), shape (8,).
    Language remains dataset metadata; joint positions are not model inputs here.
    """
    _validate_observation(observation)
    result = {}
    for key in IMAGE_KEYS:
        # NumPy keeps small per-frame operations independent of Torch's thread pool.
        pixels = observation[key].astype(np.float32) / np.float32(255.)
        result[key] = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
    state = np.array(observation['robot_state'], dtype=np.float32, copy=True)
    if not np.isfinite(state).all():
        raise ValueError('robot_state must remain finite after float32 conversion')
    result['robot_state'] = torch.from_numpy(state)
    return result


def split_episodes(episodes: Sequence[str], *, seed: int = 42,
                   train_fraction: float = .8) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return a reproducible, disjoint train/validation split of whole episodes."""
    names = tuple(episodes)
    if len(names) < 2 or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError('Provide at least two unique, nonempty episode names')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not np.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError('train_fraction must be between zero and one')
    count = int(len(names) * train_fraction)
    if not 0 < count < len(names):
        raise ValueError('Both episode splits must be nonempty')
    order = np.random.default_rng(seed).permutation(len(names))
    return tuple(names[i] for i in order[:count]), tuple(names[i] for i in order[count:])


class LiberoTorchDataset(Dataset[dict[str, torch.Tensor]]):
    """T-1 supervision pairs per episode: obs[t-1] -> action[t], t=1..T-1.

    Existing EpisodeDataset readers validate small numeric arrays once and keep
    RGB lazy. Each sample reads two camera frames, one state and one action.
    Use num_workers=0 and close() or a with block to release all HDF5 handles.
    """

    def __init__(self, path: str | Path, episodes: Sequence[str] | None = None, *,
                 suite: str | None = None, task_id: int | None = None) -> None:
        self.path = Path(path)
        self._readers: dict[str, EpisodeDataset] = {}
        self._closed = False
        available = EpisodeDataset.list_episodes(self.path)
        self.episodes = available if episodes is None else tuple(episodes)
        if (not self.episodes or len(set(self.episodes)) != len(self.episodes)
                or any(name not in available for name in self.episodes)):
            raise ValueError('Select a nonempty set of unique episodes present in the file')
        lengths = []
        try:
            for name in self.episodes:
                reader = EpisodeDataset.load(self.path, name)
                self._readers[name] = reader
                if len(reader) < 2:
                    raise ValueError(f'{name} needs at least two actions for shifted supervision')
                if len(self._readers) == 1:
                    source, contract = _verified_source(self.path, reader.metadata, suite=suite, task_id=task_id)
                _validate_source_episode(reader, source)
                lengths.append(len(reader))
            self.metadata = deepcopy(self._readers[self.episodes[0]].metadata)
            self.metadata['verified_source'] = deepcopy(source)
            self.metadata['training_contract'] = deepcopy(contract)
            self.language = self.metadata['language']
            self.episode_lengths = tuple(lengths)
            self.sample_counts = tuple(length - 1 for length in lengths)
            self._ends = tuple(accumulate(self.sample_counts))
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        """Number of aligned samples across the selected complete episodes."""
        return self._ends[-1]

    def locate(self, index: int) -> tuple[str, int]:
        """Map a global sample index to its episode and supervised action index t."""
        if isinstance(index, (bool, np.bool_)):
            raise TypeError('Sample index must be an integer')
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(f'Sample index must be in [0,{len(self)})')
        position = bisect_right(self._ends, index)
        start = 0 if position == 0 else self._ends[position - 1]
        return self.episodes[position], index - start + 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Read and preprocess exactly obs[t-1] and the original action[t]."""
        if self._closed:
            raise RuntimeError('Dataset is closed')
        if get_worker_info() is not None:
            raise RuntimeError('HDF5 multiprocessing is not supported yet; use num_workers=0')
        name, timestep = self.locate(index)
        reader = self._readers[name]
        sample = preprocess_observation(reader.observation(timestep - 1))
        # No extra scaling: the source is already normalized OSC delta + gripper.
        sample['action'] = torch.from_numpy(np.asarray(reader.actions[timestep], dtype=np.float32))
        return sample

    def close(self) -> None:
        """Release episode readers; repeated close is safe."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._closed = True

    def __enter__(self) -> 'LiberoTorchDataset':
        if self._closed:
            raise RuntimeError('Dataset is closed')
        return self

    def __exit__(self, *_args) -> None:
        self.close()
