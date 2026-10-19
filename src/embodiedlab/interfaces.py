"""Small, structural interfaces used by the policy and runner."""
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Action = NDArray[np.float64]
Observation = dict[str, np.ndarray | str]
Info = dict[str, float | int | bool | str]


class Environment(Protocol):
    """Gymnasium-style calls; info includes task metrics and success."""

    def reset(self, *, seed: int | None = None,
              init_state_index: int | None = None) -> tuple[Observation, Info]:
        """Start an episode and optionally seed its initialization."""
        ...

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, Info]:
        """Advance one control cycle and return observation and episode status."""
        ...

    def episode_metadata(self) -> dict:
        """Describe the task, initialization and action/observation conventions."""
        ...


class Policy(Protocol):
    """Observation-to-action interface, without simulator access."""

    def reset(self) -> None:
        """Clear any policy state at the start of an episode."""
        ...

    def act(self, observation: Observation) -> Action:
        """Return the task's documented action vector (LIBERO uses seven OSC values)."""
        ...
