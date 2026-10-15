"""Lazy official LIBERO demos and streaming EmbodiedLab HDF5 recordings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

NUMERIC_SHAPES = {'robot_state': (8,), 'joint_positions': (7,)}
IMAGE_KEYS = ('agentview_image', 'wrist_image')
OBSERVATION_KEYS = (*NUMERIC_SHAPES, *IMAGE_KEYS, 'language')


def _text(value: object) -> str:
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _validate_observation(observation: dict) -> None:
    """Validate only the current LIBERO state, two RGB cameras and language."""
    if set(observation) != set(OBSERVATION_KEYS):
        raise ValueError(f'Observation must contain exactly {OBSERVATION_KEYS}')
    for key, shape in NUMERIC_SHAPES.items():
        value = np.asarray(observation[key])
        if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ValueError(f'{key} must contain finite floating-point values with shape {shape}')
    for key in IMAGE_KEYS:
        value = np.asarray(observation[key])
        if value.ndim != 3 or value.shape[-1] != 3 or 0 in value.shape or value.dtype != np.uint8:
            raise ValueError(f'{key} must be nonempty HWC RGB uint8')
    if not isinstance(observation['language'], str) or not observation['language']:
        raise ValueError('language must be a nonempty string')


class EpisodeDataset:
    """One HDF5 episode; RGB is fetched per timestep, never loaded in bulk.

    Official files have T post-action observations for T actions. Our recordings
    have T+1 observations: obs[t] precedes action[t]. Pixels retain raw robosuite
    orientation in both formats; visual preview rotation is a separate concern.
    """

    def __init__(self, path: str | Path, episode: str = 'demo_0', *,
                 suite: str | None = None, task_id: int | None = None) -> None:
        self.path = Path(path)
        self.episode_name = episode
        self._file = None
        try:
            self._file = h5py.File(self.path, 'r')
            self._group = self._file[f'data/{episode}']
            self._obs = self._group['obs']
            self._recorded = 'metadata_json' in self._group.attrs
            if self._recorded:
                self.metadata = json.loads(_text(self._group.attrs['metadata_json']))
                if (not isinstance(self.metadata, dict)
                        or self.metadata.get('source_format') != 'embodiedlab_libero_hdf5'
                        or type(self.metadata.get('format_version')) is not int
                        or self.metadata['format_version'] != 1):
                    raise ValueError('Unsupported EmbodiedLab HDF5 format')
                if self.metadata.get('complete') is not True:
                    raise ValueError('Recording is incomplete; the episode did not end')
            else:
                data = self._file['data']
                problem = json.loads(_text(data.attrs['problem_info']))
                env_args = json.loads(_text(data.attrs['env_args']))
                self.metadata = {
                    'source_format': 'libero_official_hdf5', 'observation_alignment': 'post_action',
                    'image_orientation': 'raw_robosuite_opengl',
                    'source_image_convention': _text(data.attrs.get('macros_image_convention', 'unknown')),
                    'language': problem['language_instruction'], 'suite': suite or self.path.parent.name,
                    # Official files omit suite task ordering; unknown is preferable to a guessed ID.
                    'task_id': task_id, 'task_name': Path(env_args['bddl_file']).stem,
                    'control_freq': env_args['env_kwargs']['control_freq'], 'env_args': env_args,
                    'official_robot_states': 'gripper2 + eef_pos3 + quaternion_xyzw4 (9)',
                    'end_flag_semantics': 'official curated dones, not re-evaluated native success',
                }
                source = self.path.with_suffix('.source.json')
                if source.exists():
                    self.metadata['provenance'] = json.loads(source.read_text())
            self.validate()
        except (OSError, KeyError, TypeError, ValueError) as error:
            self.close()
            raise ValueError(f'Cannot load episode {self.path}/{episode}: {error}') from error

    @classmethod
    def load(cls, path: str | Path, episode: str = 'demo_0', *,
             suite: str | None = None, task_id: int | None = None) -> 'EpisodeDataset':
        """Open a validated lazy episode; close it or use a with block."""
        return cls(path, episode, suite=suite, task_id=task_id)

    @staticmethod
    def list_episodes(path: str | Path) -> tuple[str, ...]:
        """List demo names without reading any camera arrays."""
        with h5py.File(path, 'r') as file:
            return tuple(sorted(file['data'], key=lambda name: int(name.split('_')[-1])))

    def __len__(self) -> int:
        return self._group['actions'].shape[0]

    @property
    def actions(self) -> h5py.Dataset:
        """Lazily readable (T, 7) normalized OSC delta commands."""
        return self._group['actions']

    @property
    def rewards(self) -> h5py.Dataset:
        """Per-action rewards in their actual recorded dtype."""
        return self._group['rewards']

    @property
    def terminated(self) -> h5py.Dataset:
        """Native recording flags; official curated dones are a separate field."""
        if not self._recorded:
            raise ValueError('Official demos contain curated dones, not native terminated flags')
        return self._group['terminated']

    @property
    def truncated(self) -> h5py.Dataset:
        """Native recording timeout flags; official demos do not record them."""
        if not self._recorded:
            raise ValueError('Official demos do not record truncation flags')
        return self._group['truncated']

    @property
    def timestamps(self) -> h5py.Dataset:
        """Actual relative simulation timestamps in our recordings, shape (T+1,)."""
        if not self._recorded:
            raise ValueError('Official demos do not contain aligned observation timestamps')
        return self._group['timestamps']

    @property
    def language(self) -> str:
        """Natural-language task instruction shared by all frames."""
        return self.metadata['language']

    @property
    def task_id(self) -> int | None:
        """Explicit suite task index; official files may require an external ID."""
        return self.metadata.get('task_id')

    @property
    def environment_config(self) -> dict:
        """Return reset/constructor settings only for our own recordings."""
        if not self._recorded:
            raise ValueError('Official demos require independent source-model/state diagnostics')
        keys = ('suite', 'task_id', 'image_size', 'max_episode_steps', 'init_state_index', 'settle_steps')
        return {key: self.metadata[key] for key in keys}

    @property
    def initial_state(self) -> np.ndarray:
        """Reset-only simulator state; never include it in policy observations."""
        if self._recorded:
            raise ValueError('Our recordings use seed + official init_state_index, not flattened state injection')
        return np.asarray(self._group['states'][0], dtype=np.float64)

    @property
    def model_xml(self) -> str | None:
        """Source model for explicit official-demo validation, when present."""
        value = self._group.attrs.get('model_file')
        return _text(value) if value is not None else None

    def validate(self) -> None:
        """Check shapes and small numeric arrays; RGB arrays remain lazy."""
        if self.actions.ndim != 2 or self.actions.shape[1] != 7 or self.actions.shape[0] < 1:
            raise ValueError('actions must have nonempty shape (T, 7)')
        length = len(self)
        actions = self.actions[()]
        if self.actions.dtype.kind != 'f' or not np.isfinite(actions).all():
            raise ValueError('actions must contain finite floating-point OSC deltas')
        if np.any(actions < -1.) or np.any(actions > 1.):
            raise ValueError('Normalized OSC actions must be in [-1,1]')
        if not isinstance(self.language, str) or not self.language:
            raise ValueError('language must be a nonempty string')
        count = length + 1 if self._recorded else length
        shapes = NUMERIC_SHAPES if self._recorded else {'ee_states': (6,), 'gripper_states': (2,), 'joint_states': (7,)}
        for key, shape in shapes.items():
            value = self._obs[key]
            if value.shape != (count, *shape) or value.dtype.kind != 'f' or not np.isfinite(value[()]).all():
                raise ValueError(f'obs/{key} must contain finite values with shape {(count, *shape)}')
        cameras = IMAGE_KEYS if self._recorded else ('agentview_rgb', 'eye_in_hand_rgb')
        for key in cameras:
            image = self._obs[key]
            if image.ndim != 4 or image.shape[0] != count or image.shape[-1] != 3 or 0 in image.shape or image.dtype != np.uint8:
                raise ValueError(f'obs/{key} must be (observations, H, W, 3) uint8')
        rewards = self._group['rewards']
        if rewards.shape != (length,) or rewards.dtype.kind not in 'fiu' or not np.isfinite(rewards[()]).all():
            raise ValueError('rewards must contain T finite values')
        for key in (('terminated', 'truncated') if self._recorded else ('dones',)):
            value = self._group[key]
            if (value.shape != (length,) or not np.isin(value[()], [0, 1]).all()
                    or (self._recorded and value.dtype != np.bool_)):
                raise ValueError(f'{key} must contain T boolean end flags')
        if not self._recorded:
            states = self._group['states']
            if states.ndim != 2 or states.shape[0] != length or not np.isfinite(states[()]).all():
                raise ValueError('Official simulator states must contain T finite flattened states')
        if self._recorded:
            if type(self.metadata.get('seed')) is not int or self.metadata['seed'] < 0:
                raise ValueError('seed must be a nonnegative integer')
            if type(self.metadata.get('success')) is not bool:
                raise ValueError('success must be a boolean')
            if self.metadata.get('image_orientation') != 'raw_robosuite_opengl':
                raise ValueError('Recorded image_orientation must be raw_robosuite_opengl')
            if self.metadata.get('observation_alignment') != 'pre_action':
                raise ValueError('Recorded observation_alignment must be pre_action')
            max_steps = self.metadata.get('max_episode_steps')
            if type(max_steps) is not int or max_steps < 1 or length > max_steps:
                raise ValueError('max_episode_steps must be positive and at least the recorded length')
            terminated, truncated = self._group['terminated'][()], self._group['truncated'][()]
            done = terminated | truncated
            if np.any(terminated & truncated) or done[:-1].any() or not done[-1]:
                raise ValueError('Only the last transition may end, with exactly one end flag')
            if self.metadata['success'] != bool(terminated[-1]):
                raise ValueError('Success must equal the final native terminated flag')
            if truncated[-1] and length != max_steps:
                raise ValueError('Truncation must occur at max_episode_steps')
            times = self._group['timestamps'][()]
            if times.shape != (count,) or not np.isfinite(times).all() or abs(times[0]) > 1e-9 or np.any(np.diff(times) <= 0):
                raise ValueError('timestamps must contain T+1 finite increasing times')
            control_dt = self.metadata.get('control_dt')
            if (type(control_dt) not in (int, float) or not np.isfinite(control_dt) or control_dt <= 0
                    or not np.allclose(np.diff(times), control_dt, rtol=0, atol=1e-9)):
                raise ValueError('Timestamp intervals must match finite positive control_dt')

    def observation(self, index: int) -> dict:
        """Read one frame: pre-action for recordings, post-action for official demos."""
        count = len(self) + 1 if self._recorded else len(self)
        if not 0 <= index < count:
            raise IndexError(f'Observation index must be in [0, {count})')
        if self._recorded:
            observation = {key: self._obs[key][index] for key in (*NUMERIC_SHAPES, *IMAGE_KEYS)}
        else:
            robot_state = np.concatenate([self._obs['ee_states'][index], self._obs['gripper_states'][index]])
            observation = {'robot_state': robot_state,
                           'joint_positions': self._obs['joint_states'][index],
                           'agentview_image': self._obs['agentview_rgb'][index],
                           'wrist_image': self._obs['eye_in_hand_rgb'][index]}
        observation['language'] = self.language
        _validate_observation(observation)
        return observation

    def step(self, index: int) -> tuple[dict, np.ndarray, float]:
        """Read stored obs[index], action[index], reward[index]; consult alignment metadata."""
        if not 0 <= index < len(self):
            raise IndexError('Step index must be in [0, T)')
        return self.observation(index), self.actions[index], float(self._group['rewards'][index])

    def transitions(self, start: int = 0, stop: int | None = None) -> Iterator[dict]:
        """Yield aligned transitions only for our T+1 recordings."""
        if not self._recorded:
            raise ValueError('Official observations are post-action; use shifted_bc_pairs() explicitly')
        stop = len(self) if stop is None else stop
        if not 0 <= start <= stop <= len(self):
            raise IndexError('Transition range must be in [0, T]')
        for index in range(start, stop):
            obs, action, reward = self.step(index)
            yield {'observation': obs, 'action': action, 'reward': reward,
                   'next_observation': self.observation(index + 1),
                   'terminated': bool(self._group['terminated'][index]),
                   'truncated': bool(self._group['truncated'][index])}

    def shifted_bc_pairs(self) -> Iterator[tuple[dict, np.ndarray]]:
        """Official only: obs[t-1] -> action[t], t=1..T-1; drop the unavailable first pair."""
        if self._recorded:
            raise ValueError('Our recordings already align obs[t] with action[t]; use transitions()')
        for index in range(1, len(self)):
            yield self.observation(index - 1), self.actions[index]

    def close(self) -> None:
        """Release the read handle; repeated close is safe."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> 'EpisodeDataset':
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class EpisodeRecorder:
    """Stream observations to HDF5; each RGB frame is its own gzip chunk."""

    def __init__(self, path: str | Path, environment_metadata: dict | None = None) -> None:
        self.path = Path(path)
        if self.path.suffix not in ('.h5', '.hdf5'):
            raise ValueError('Recording path must end in .h5 or .hdf5')
        self._file = None
        self._ended = False
        self.set_environment_metadata(environment_metadata or {})

    def set_environment_metadata(self, metadata: dict) -> None:
        """Copy the actual post-reset environment configuration before start()."""
        self._environment_metadata = json.loads(json.dumps(metadata, allow_nan=False))

    def start(self, observation: dict, info: dict, *, seed: int) -> None:
        """Open an exclusive recording and write initial pre-action obs[0]."""
        if self._file is not None:
            raise RuntimeError('Recorder is already recording')
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        _validate_observation(observation)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, 'x')
        self._group = self._file.create_group('data').create_group('demo_0')
        self._obs = self._group.create_group('obs')
        self._metadata = {**self._environment_metadata, 'source_format': 'embodiedlab_libero_hdf5',
                          'format_version': 1, 'seed': seed, 'complete': False,
                          'observation_alignment': 'pre_action', 'image_orientation': 'raw_robosuite_opengl',
                          'language': observation['language']}
        for key in (*NUMERIC_SHAPES, *IMAGE_KEYS):
            value = np.asarray(observation[key])
            options = {'compression': 'gzip', 'compression_opts': 1} if key in IMAGE_KEYS else {}
            self._obs.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                     chunks=(1, *value.shape), dtype=value.dtype, **options)
        for key, shape, dtype in [('actions', (7,), np.float64), ('rewards', (), np.float64),
                                  ('terminated', (), np.bool_), ('truncated', (), np.bool_), ('timestamps', (), np.float64)]:
            self._group.create_dataset(key, shape=(0, *shape), maxshape=(None, *shape), chunks=True, dtype=dtype)
        self._write_observation(observation, float(info['simulation_time']))
        self._ended = False

    @staticmethod
    def _append_array(dataset: h5py.Dataset, value: object) -> None:
        index = dataset.shape[0]
        dataset.resize(index + 1, axis=0)
        dataset[index] = value

    def _write_observation(self, observation: dict, timestamp: float) -> None:
        for key in (*NUMERIC_SHAPES, *IMAGE_KEYS):
            self._append_array(self._obs[key], observation[key])
        self._append_array(self._group['timestamps'], timestamp)

    def append(self, action: np.ndarray, observation: dict, reward: float,
               terminated: bool, truncated: bool, info: dict) -> None:
        """Append the exact issued normalized OSC command and its successor obs."""
        if self._file is None or self._ended:
            raise RuntimeError('Start a new recording before append()')
        _validate_observation(observation)
        for key in (*NUMERIC_SHAPES, *IMAGE_KEYS):
            if np.asarray(observation[key]).shape != self._obs[key].shape[1:]:
                raise ValueError(f'{key} shape changed during the episode')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all() or not np.isfinite(reward):
            raise ValueError('action must contain seven finite OSC deltas and reward must be finite')
        if observation['language'] != self._metadata['language']:
            raise ValueError('Episode language changed')
        for key, value in [('actions', action), ('rewards', reward), ('terminated', terminated), ('truncated', truncated)]:
            self._append_array(self._group[key], value)
        self._write_observation(observation, float(info['simulation_time']))
        self._metadata['success'] = bool(info['success'])
        self._metadata['failure_reason'] = str(info.get('failure_reason', ''))
        self._ended = bool(terminated or truncated)

    def close(self) -> None:
        """Finalize completed runs; interrupted files remain explicitly incomplete."""
        if self._file is not None:
            self._metadata['complete'] = self._ended
            self._group.attrs['metadata_json'] = json.dumps(self._metadata, allow_nan=False)
            self._file.close()
            self._file = None

    def episode(self) -> EpisodeDataset:
        """Close writer and open the completed recording for lazy reading."""
        self.close()
        return EpisodeDataset.load(self.path)
