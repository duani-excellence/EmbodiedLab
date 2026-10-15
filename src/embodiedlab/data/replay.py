"""Replay our recorded OSC commands through public environment methods."""
from dataclasses import dataclass

import numpy as np

from embodiedlab.data.episode import EpisodeDataset, IMAGE_KEYS, _validate_observation
from embodiedlab.interfaces import Environment


@dataclass(frozen=True)
class ReplayResult:
    """Actual state, pixel, reward, time and outcome deviations; no target is invented."""
    recorded_length: int
    replay_length: int
    max_robot_state_deviation: float
    max_joint_position_deviation: float
    max_rgb_pixel_deviation: int
    max_reward_deviation: float
    max_timestamp_deviation: float
    language_match: bool
    terminated_match: bool
    truncated_match: bool
    success_match: bool
    matches: bool

    @property
    def length_match(self) -> bool:
        """Whether every recorded control action was executed."""
        return self.recorded_length == self.replay_length


def replay_episode(env: Environment, episode: EpisodeDataset, *, atol: float = 1e-6,
                   pixel_atol: int = 0) -> ReplayResult:
    """Reset an own T+1 recording and re-execute actions; never write simulator state.

    Official demos have a different observation alignment and reset/model source.
    Their independent diagnostics must not be reported as exact recording replay.
    """
    episode.validate()
    if episode.metadata.get('observation_alignment') != 'pre_action':
        raise ValueError('Official post-action demos require separate source-state/model diagnostics')
    if not np.isfinite(atol) or atol < 0 or type(pixel_atol) is not int or pixel_atol < 0:
        raise ValueError('Replay tolerances must be finite and nonnegative')
    current = env.episode_metadata()
    for key in ('suite', 'task_id', 'task_name', 'image_size', 'settle_steps',
                'control_mode', 'image_orientation'):
        if current[key] != episode.metadata[key]:
            raise ValueError(f'Replay {key} differs from the recorded environment')
    observation, info = env.reset(seed=episode.metadata['seed'],
                                  init_state_index=episode.metadata['init_state_index'])
    max_state = max_joint = max_reward = max_time = 0.0
    max_pixel = 0
    language_match = terminated_match = truncated_match = True

    def compare(index: int) -> None:
        nonlocal max_state, max_joint, max_pixel, max_time, language_match
        _validate_observation(observation)
        recorded = episode.observation(index)
        language_match &= observation['language'] == recorded['language']
        max_state = max(max_state, float(np.max(np.abs(observation['robot_state'] - recorded['robot_state']))))
        max_joint = max(max_joint, float(np.max(np.abs(observation['joint_positions'] - recorded['joint_positions']))))
        for key in IMAGE_KEYS:
            if observation[key].shape != recorded[key].shape:
                raise ValueError(f'Replay {key} shape differs from the recording')
            # uint8 subtraction would wrap; signed pixels preserve actual deviation.
            pixel_delta = observation[key].astype(np.int16) - recorded[key].astype(np.int16)
            max_pixel = max(max_pixel, int(np.max(np.abs(pixel_delta))))
        timestamp = float(info['simulation_time'])
        if not np.isfinite(timestamp):
            raise ValueError('Replay simulation_time must be finite')
        max_time = max(max_time, abs(timestamp - float(episode.timestamps[index])))

    compare(0)
    length = 0
    for index in range(len(episode)):
        observation, reward, terminated, truncated, info = env.step(episode.actions[index])
        length += 1
        compare(index + 1)
        if not np.isfinite(reward):
            raise ValueError('Replay reward must be finite')
        max_reward = max(max_reward, abs(float(reward) - float(episode.rewards[index])))
        terminated_match &= terminated == bool(episode.terminated[index])
        truncated_match &= truncated == bool(episode.truncated[index])
        if terminated or truncated:
            break
    terminated_match &= length == len(episode)
    truncated_match &= length == len(episode)
    success_match = bool(info['success']) == episode.metadata['success']
    matches = (length == len(episode) and language_match and terminated_match and truncated_match and success_match
               and max_state <= atol and max_joint <= atol and max_pixel <= pixel_atol
               and max_reward <= atol and max_time <= atol)
    return ReplayResult(len(episode), length, max_state, max_joint, max_pixel, max_reward, max_time,
                        bool(language_match), bool(terminated_match), bool(truncated_match),
                        bool(success_match), bool(matches))
