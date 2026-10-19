"""Native robosuite Wipe with read-only, physics-rate contact measurements."""
from copy import deepcopy
import hashlib
import importlib.metadata as metadata

import mujoco
import numpy as np


class WipeEnv:
    """Panda/WipingGripper; six normalized world-frame OSC pose deltas.

    Position scales by 0.05 m, rotation by 0.5 rad. There is no gripper action.
    Policies see native marker XY/flags (oracle object state), pose and wrist
    force. Ground-truth table height/contact pairs stay in diagnostic info.
    """

    def __init__(self, *, scenario: str = 'nominal', horizon: int = 1000,
                 render: bool = False) -> None:
        if scenario not in ('nominal', 'height_variation', 'upward_pulse'):
            raise ValueError('Unknown Wipe scenario')
        if type(horizon) is not int or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        self.scenario, self.horizon, self.render = scenario, horizon, render
        self._env = None
        self._original_step = None
        self._closed = False
        self._ended = True
        self.control_dt = .05

    def reset(self, *, seed: int | None = None,
              init_state_index: int | None = None) -> tuple[dict, dict]:
        """Rebuild the native seeded scene; Wipe has no official init-state IDs."""
        if self._closed:
            raise RuntimeError('Environment is closed')
        if init_state_index is not None:
            raise ValueError('Wipe uses seeds, not LIBERO initial-state indices')
        if type(seed) is not int or seed < 0:
            raise ValueError('Wipe reset requires a nonnegative integer seed')
        if self._env is not None:
            if self._original_step is not None:
                self._env.sim.step = self._original_step
            self._env.close()
            self._env = self._original_step = None
        import robosuite
        from robosuite.controllers import load_controller_config
        from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG

        # v1.4 samples during construction as well as reset, using NumPy's RNG.
        np.random.seed(seed)
        config = deepcopy(DEFAULT_WIPE_CONFIG)
        config['use_condensed_obj_obs'] = False  # Official explicit-marker observations.
        if self.scenario == 'height_variation':
            config['table_height_std'] = .01  # Native min(0, Normal(0, 1 cm)).
        self.controller_config = load_controller_config(default_controller='OSC_POSE')
        self._env = robosuite.make('Wipe', robots='Panda', task_config=config,
            controller_configs=self.controller_config, horizon=self.horizon,
            use_camera_obs=False, has_offscreen_renderer=self.render, has_renderer=False)
        raw = self._env.reset()
        self.seed, self.steps = seed, 0
        self.first_contact_step = self.pulse_start_step = None
        self._ended = False
        self._wiper_ids = {self._env.sim.model.geom_name2id(n)
                          for n in self._env.robots[0].gripper.contact_geoms}
        self._table_id = self._env.sim.model.geom_name2id('table_collision')
        self._tool_body = self._env.sim.model.body_name2id('gripper0_wiping_gripper')
        self._ft_site = self._env.sim.model.site_name2id('gripper0_ft_frame')
        self._original_step = self._env.sim.step
        self._env.sim.step = self._physics_step
        self.force_substeps: list[float] = []
        obs = self._observation(raw)
        self.initial_observation = obs
        state = np.r_[self._env.sim.data.qpos, self._env.sim.data.qvel,
                      obs['marker_xy'].ravel(), self._env.table_offset]
        self.initial_state_sha256 = hashlib.sha256(state.tobytes()).hexdigest()
        self.initial_frame = self.frame() if self.render else None
        return obs, {'success': False, 'steps': 0, 'simulation_time': 0.}

    def _physics_step(self) -> None:
        self._original_step()
        sim, normal_force = self._env.sim, 0.
        # Sample after every native 2 ms step. Read contacts without forwarding
        # the model or changing solver/controller state.
        for index, contact in enumerate(sim.data.contact[:sim.data.ncon]):
            if ((contact.geom1 in self._wiper_ids and contact.geom2 == self._table_id)
                    or (contact.geom2 in self._wiper_ids and contact.geom1 == self._table_id)):
                force = np.zeros(6)
                mujoco.mj_contactForce(sim.model._model, sim.data._data, index, force)
                normal_force += abs(float(force[0]))
        self.force_substeps.append(normal_force)

    def _observation(self, raw: dict) -> dict:
        rotation = self._env.sim.data.site_xmat[self._ft_site].reshape(3, 3)
        return {'eef_pos': raw['robot0_eef_pos'].copy(),
                'eef_quat': raw['robot0_eef_quat'].copy(),
                'marker_xy': np.array([raw[f'marker{i}_pos'][:2] for i in range(self._env.num_markers)]),
                'marker_wiped': np.array([raw[f'marker{i}_wiped'] for i in range(self._env.num_markers)], dtype=bool),
                'wrist_force_world_N': rotation @ self._env.robots[0].ee_force}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute native dynamics; distinguish native safety endings and timeout."""
        if self._env is None or self._ended or self._closed:
            raise RuntimeError('Call reset before stepping an unfinished episode')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Wipe action must be six finite normalized OSC deltas in [-1,1]')
        pulse = (self.scenario == 'upward_pulse' and self.pulse_start_step is not None
                 and self.pulse_start_step <= self.steps < self.pulse_start_step + 10)
        # A declared 12 N, 0.5 s external upward load at the tool CoM. No state
        # teleportation, model edit, controller change or disturbance to LIBERO.
        self._env.sim.data.xfrc_applied[self._tool_body, 2] = 12. if pulse else 0.
        self.force_substeps = []
        raw, reward, done, _ = self._env.step(action)
        self.steps += 1
        force = np.asarray(self.force_substeps)
        if self.first_contact_step is None and np.any(force > 1.):
            self.first_contact_step = self.steps
            if self.scenario == 'upward_pulse':
                self.pulse_start_step = self.steps + 30
        success = bool(self._env._check_success())
        collision = bool(self._env.check_contact(self._env.robots[0].robot_model))
        joint_limit = bool(self._env.robots[0].check_q_limits())
        terminated = success or collision or joint_limit
        truncated = self.steps >= self.horizon and not terminated
        if bool(done) != (terminated or truncated):
            raise RuntimeError('Unexpected native Wipe termination')
        self._ended = terminated or truncated
        info = {'success': success, 'steps': self.steps,
                'native_reward': float(reward),
                'simulation_time': self.steps * self.control_dt,
                'native_wiped_count': len(self._env.wiped_markers),
                'wipe_fraction': len(self._env.wiped_markers) / self._env.num_markers,
                'contact_fraction': float(np.mean(force > 1.)),
                'normal_mean_N': float(np.mean(force)), 'normal_peak_N': float(np.max(force)),
                'pulse_active': bool(pulse),
                'failure_reason': ('arm_collision' if collision else 'joint_limit' if joint_limit
                                   else 'time_limit' if truncated else '')}
        return self._observation(raw), float(reward), terminated, truncated, info

    def episode_metadata(self) -> dict:
        """Actual native configuration; diagnostic height is never policy input."""
        return {'env_name': 'robosuite.Wipe', 'robot': 'Panda', 'gripper': 'WipingGripper',
                'scenario': self.scenario, 'seed': self.seed, 'max_episode_steps': self.horizon,
                'control_dt': self.control_dt, 'physics_dt': self._env.model_timestep,
                'action_dim': 6, 'controller': self.controller_config,
                'task_config': self._env.task_config,
                'actual_table_z_m': float(self._env.table_offset[2]),
                'initial_state_sha256': self.initial_state_sha256,
                'policy_observation': 'native oracle marker XY/flags, eef pose, wrist force; no table height or contact forces',
                'versions': {n: metadata.version(n) for n in ('robosuite', 'mujoco', 'numpy')},
                'success': 'unmodified native Wipe._check_success',
                'marker_flags': 'native cached observable, sampled before reward updates; use native_wiped_count for same-step erasure events',
                'contact_measurement': 'sum of tool-table normal forces, every native physics step',
                'pulse': '12 N +worldZ at tool CoM, 10 control steps; starts 30 steps after first >1 N contact'}

    def frame(self) -> np.ndarray:
        """Render a diagnostic RGB frame; no pixels enter these state baselines."""
        return self._env.sim.render(width=512, height=512, camera_name='agentview')[::-1].copy()

    def close(self) -> None:
        """Release only this environment; old LIBERO environments are untouched."""
        if self._env is not None:
            if self._original_step is not None:
                self._env.sim.step = self._original_step
            self._env.close()
            self._env = None
            self._original_step = None
        self._closed = self._ended = True
