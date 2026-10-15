"""Minimal LIBERO baselines; seven normalized Cartesian/gripper commands."""
import numpy as np

from embodiedlab.interfaces import Action, Observation


class ZeroPolicy:
    """Return zero translation, rotation and gripper commands."""

    def reset(self) -> None:
        """This policy has no episode state."""

    def act(self, observation: Observation) -> Action:
        """Return an independent seven-dimensional zero action."""
        return np.zeros(7, dtype=np.float64)


class RandomPolicy:
    """Sample each Cartesian/gripper command uniformly from [-1, 1]."""

    def __init__(self, seed: int = 0) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError('Policy seed must be a nonnegative integer')
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        """Restart this policy's local RNG for reproducible action sequences."""
        self.rng = np.random.default_rng(self.seed)

    def act(self, observation: Observation) -> Action:
        """Return seven normalized end-effector delta/gripper commands."""
        return self.rng.uniform(-1.0, 1.0, size=7)
