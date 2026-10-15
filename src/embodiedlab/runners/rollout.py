"""A public-interface episode loop, independent of simulator and policy details."""
from dataclasses import dataclass
import secrets
from typing import TYPE_CHECKING, Callable

import numpy as np

from embodiedlab.interfaces import Action, Environment, Info, Observation, Policy

StepObserver = Callable[[Observation, Action, Info], None]

if TYPE_CHECKING:
    from embodiedlab.data.episode import EpisodeRecorder


@dataclass(frozen=True)
class EpisodeResult:
    """Executed reward sum, action count, native success and ending flags."""

    reward: float
    length: int
    success: bool
    terminated: bool
    truncated: bool
    simulation_time: float | None = None
    init_state_index: int | None = None


def run_episode(env: Environment, policy: Policy, *, seed: int | None = None,
                init_state_index: int | None = None,
                recorder: 'EpisodeRecorder | None' = None,
                on_step: StepObserver | None = None) -> EpisodeResult:
    """Run an episode; an optional observer receives post-action obs/action/info.

    The read-only observer is for diagnostics, never part of the policy input.
    """
    if recorder is not None and seed is None:
        seed = secrets.randbits(32)
    reset_options = {'seed': seed}
    if init_state_index is not None:
        reset_options['init_state_index'] = init_state_index
    try:
        observation, info = env.reset(**reset_options)
        policy.reset()
        if recorder is not None:
            recorder.set_environment_metadata(env.episode_metadata())
            recorder.start(observation, info, seed=seed)
        total_reward = 0.0
        length = 0
        while True:
            action = policy.act(observation)
            # Preserve exactly the policy action passed to step, before mutation.
            recorded_action = (np.asarray(action).copy()
                               if recorder is not None or on_step is not None else None)
            observation, reward, terminated, truncated, info = env.step(action)
            if recorder is not None:
                recorder.append(recorded_action, observation, reward, terminated, truncated, info)
            if on_step is not None:
                on_step(observation, recorded_action, info)
            total_reward += float(reward)
            length += 1
            if terminated or truncated:
                return EpisodeResult(
                    reward=total_reward, length=length,
                    success=bool(info.get('success', False)),
                    terminated=bool(terminated), truncated=bool(truncated),
                    simulation_time=(float(info['simulation_time'])
                                     if 'simulation_time' in info else None),
                    init_state_index=info.get('init_state_index', init_state_index),
                )
    finally:
        if recorder is not None:
            recorder.close()
