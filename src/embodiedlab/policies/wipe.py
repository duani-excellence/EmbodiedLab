"""Two traditional Wipe baselines sharing native OSC and object-state feedback."""
import numpy as np
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.transform_utils import quat2mat


class WipePolicy:
    """Touch-localized fixed depth or wrist-force admittance, without simulator access.

    Both use the same nearest-remaining-marker planner, approach and orientation.
    These are explicit oracle-state controller baselines, not learned experts.
    """

    def __init__(self, mode: str = 'touch_position', *, target_force_N: float = 8.) -> None:
        if mode not in ('touch_position', 'force_admittance'):
            raise ValueError('Unknown Wipe policy')
        self.mode, self.target_force_N = mode, target_force_N
        self.reset()

    def reset(self) -> None:
        """Clear the contact localization, force bias and phase at each reset."""
        self.phase, self.steps = 'hover', 0
        self.hover_xy = None
        self.bias_z = self.force_estimate_N = 0.
        self.z_reference = 1.

    def act(self, observation: dict) -> np.ndarray:
        """Return six normalized world pose deltas; force feedback adjusts only Z."""
        pos = observation['eef_pos']
        rotation = quat2mat(observation['eef_quat'])
        desired_rotation = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        rot_error = orientation_error(desired_rotation, rotation)
        remaining = observation['marker_xy'][~observation['marker_wiped']]
        if len(remaining) == 0:
            remaining = observation['marker_xy']
        target_xy = remaining[np.argmin(np.linalg.norm(remaining - pos[:2], axis=1))]
        if self.hover_xy is None:
            self.hover_xy = target_xy.copy()
        # MuJoCo's wrist sensor reports the child-on-parent wrench. Rotate its
        # site-frame force to world in the environment, then subtract a measured
        # unloaded bias: upward contact corresponds to bias_z - wrist_world_z.
        if self.phase == 'hover':
            self.bias_z = float(observation['wrist_force_world_N'][2])
        force = self.bias_z - float(observation['wrist_force_world_N'][2])
        self.force_estimate_N += .25 * (force - self.force_estimate_N)

        if self.phase == 'hover':
            target_xy, self.z_reference = self.hover_xy, 1.
            if (self.steps > 20 and np.linalg.norm(pos[:2] - target_xy) < .012
                    and abs(pos[2] - 1.) < .01 and np.linalg.norm(rot_error) < .03):
                self.phase, self.z_reference = 'approach', float(pos[2])
        elif self.phase == 'approach':
            target_xy = self.hover_xy
            self.z_reference = max(.86, self.z_reference - .0015)
            if self.force_estimate_N > 1.:
                self.phase = 'wipe'
                # Strong position baseline also localizes the unknown table by
                # first touch; it is not deliberately tied to nominal height.
                self.z_reference = float(pos[2]) - .0035
        elif self.mode == 'force_admittance':
            # dz/dt = gain * (measured - desired force). Positive error lifts
            # the virtual equilibrium; integration preserves a static load.
            change = .001 * .05 * (self.force_estimate_N - self.target_force_N)
            self.z_reference = float(np.clip(self.z_reference + np.clip(change, -.001, .001), .86, .92))

        delta = np.r_[np.clip(1.5 * (target_xy - pos[:2]), -.015, .015),
                      np.clip(self.z_reference - pos[2], -.005, .005)]
        self.steps += 1
        return np.r_[delta / .05, np.clip(rot_error / .5, -.2, .2)].astype(np.float64)
