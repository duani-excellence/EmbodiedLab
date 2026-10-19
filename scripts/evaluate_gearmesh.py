"""Evaluate the official recurrent PPO on native Isaac Lab Factory GearMesh."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
from time import perf_counter

from isaaclab.app import AppLauncher

TASK = 'Isaac-Factory-GearMesh-Direct-v0'
parser = argparse.ArgumentParser(description=__doc__)
# AppLauncher inspects existing arguments; register it before required options
# so --help does not fail its preliminary parse.
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--contacts', action='store_true', help='Read filtered native normal forces at 15Hz')
args = parser.parse_args()
if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
    parser.error('Episode seeds must be distinct and nonnegative.')
args.checkpoint = args.checkpoint.resolve(strict=True)
args.output_dir.mkdir(parents=True, exist_ok=False)
launcher = AppLauncher(args)
app = launcher.app

# Omniverse modules must be imported after AppLauncher starts Isaac Sim.
import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_tasks.direct.factory import factory_utils
from isaacsim.core.simulation_manager import SimulationManager
from embodiedlab.runners.rollout import run_episode


def write_json(path: Path, value: dict) -> None:
    """Save measured values; reject non-finite JSON numbers."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path: Path) -> str:
    """Hash a checkpoint or source file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def contact_view(env):
    """Use exact native rigid-body paths; enable reporting already present in USD."""
    sensors = list(env._held_asset.root_physx_view.link_paths[0])
    filters = [path for asset in (env._fixed_asset, env._small_gear_asset, env._large_gear_asset)
               for path in asset.root_physx_view.link_paths[0]]
    gear_count = len(filters)
    robot_paths = env._robot.root_physx_view.link_paths[0]
    filters += [robot_paths[env.left_finger_body_idx], robot_paths[env.right_finger_body_idx]]
    if not sensors or not gear_count:
        raise RuntimeError('Native gear articulation contains no rigid bodies to measure.')
    view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
        sensors, filter_patterns=[filters] * len(sensors))
    if view.sensor_count != len(sensors) or view.filter_count != len(filters):
        raise RuntimeError('PhysX contact view did not resolve the exact native gear/finger bodies.')
    return view, {'sensor_paths': sensors, 'filter_paths': filters, 'gear_filter_count': gear_count}


class SingleFactoryEnv:
    """Expose one native environment to the existing, simulator-agnostic Runner."""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.native = wrapped.unwrapped
        self.contact, self.contact_metadata = contact_view(self.native) if args.contacts else (None, None)
        self.original_rewards = self.native._get_rewards
        self.native._get_rewards = self.capture_rewards
        self.latest = {}
        self.latest_info = {}

    def snapshot(self) -> tuple[dict, dict]:
        """Read native state; diagnostics are never passed to the PPO network."""
        env = self.native
        held, _ = factory_utils.get_held_base_pose(
            env.held_pos, env.held_quat, env.cfg_task.name,
            env.cfg_task.fixed_asset_cfg, 1, env.device)
        target, _ = factory_utils.get_target_held_base_pose(
            env.fixed_pos, env.fixed_quat, env.cfg_task.name,
            env.cfg_task.fixed_asset_cfg, 1, env.device)
        observation = {name: tensor[0].detach().cpu().numpy().copy()
                       for name, tensor in env._get_observations().items()}
        for name, tensor in {'joint_pos': env.joint_pos, 'joint_vel': env.joint_vel,
                             'fingertip_pos': env.fingertip_midpoint_pos,
                             'fingertip_quat': env.fingertip_midpoint_quat,
                             'held_pos': env.held_pos, 'held_quat': env.held_quat,
                             'fixed_pos': env.fixed_pos, 'fixed_quat': env.fixed_quat,
                             'held_base_pos': held, 'target_base_pos': target}.items():
            observation[name] = tensor[0].detach().cpu().numpy().copy()
        if self.contact is not None:
            for name, tensor in {
                'pair_normal_force_w_N': self.contact.get_contact_force_matrix(dt=env.physics_dt),
                'net_normal_force_w_N': self.contact.get_net_contact_forces(dt=env.physics_dt)}.items():
                observation[name] = tensor.detach().cpu().numpy().copy()
                if not np.isfinite(observation[name]).all():
                    raise RuntimeError(f'PhysX reported non-finite {name}; no substitute values written.')
            if observation['pair_normal_force_w_N'].shape != (self.contact.sensor_count, self.contact.filter_count, 3):
                raise RuntimeError('PhysX returned an unexpected contact-force matrix shape.')
        step = int(env.episode_length_buf[0].item())
        info = {'success': bool(env._get_curr_successes(env.cfg_task.success_threshold)[0]),
                'simulation_time': step * env.step_dt,
                'xy_error_m': float(torch.linalg.vector_norm((held - target)[0, :2])),
                'z_error_m': float((held - target)[0, 2]),
                'native_terminated': bool(env.reset_terminated[0]),
                'native_truncated': bool(env.reset_time_outs[0])}
        return observation, info

    def capture_rewards(self):
        """Read the terminal state BEFORE native DirectRLEnv auto-resets it."""
        reward = self.original_rewards()
        self.latest, self.latest_info = self.snapshot()
        self.latest_info['native_reward'] = float(reward[0])
        return reward

    def reset(self, *, seed=None, init_state_index=None):
        """Use official randomized resets, reproducible by explicit seed."""
        if init_state_index is not None:
            raise ValueError('Factory uses seeded native resets, not LIBERO initial-state indices.')
        self.native.seed(seed)
        self.wrapped.reset()
        self.latest, self.latest_info = self.snapshot()
        self.initial_observation = self.latest
        return self.latest, {'simulation_time': 0.0, 'success': self.latest_info['success']}

    def step(self, action):
        """Native wrapper clips six actions; success and time limit stay separate."""
        action = np.asarray(action)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError('Factory action must be a finite six-dimensional vector.')
        tensor = torch.as_tensor(action, dtype=torch.float32, device=self.native.device)[None]
        _, reward, dones, _ = self.wrapped.step(tensor)
        # Factory labels timeout as both flags. Preserve raw flags in info, but
        # expose the known time limit as truncated; success is an independent metric.
        return self.latest, float(reward[0]), False, bool(dones[0]), self.latest_info.copy()

    def close(self) -> None:
        """Remove the read-only measurement hook and release the native scene."""
        self.native._get_rewards = self.original_rewards
        self.wrapped.close()


class NativePPOPolicy:
    """Call the restored upstream player with actor observations only."""

    def __init__(self, player, device):
        self.player, self.device = player, device

    def reset(self) -> None:
        self.player.reset()
        self.first_action = True
        self.latency_s = []

    def act(self, observation: dict) -> np.ndarray:
        obs = torch.as_tensor(observation['policy'], device=self.device)[None]
        if self.first_action:
            self.player.get_batch_size(obs, 1)
            if self.player.is_rnn:
                self.player.init_rnn()
            self.first_action = False
        torch.cuda.synchronize()
        started = perf_counter()
        with torch.inference_mode():
            action = self.player.get_action(self.player.obs_to_torch(obs),
                                           is_deterministic=self.player.is_deterministic)
        torch.cuda.synchronize()
        self.latency_s.append(perf_counter() - started)
        return action[0].detach().cpu().numpy().astype(np.float64)


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device or 'cuda:0', num_envs=1)
    cfg.seed = args.seeds[0]
    agent_cfg = load_cfg_from_registry(TASK, 'rl_games_cfg_entry_point')
    params = agent_cfg['params']
    params['seed'] = args.seeds[0]
    params['config']['device'] = cfg.sim.device
    params['config']['num_actors'] = 1
    params['load_checkpoint'], params['load_path'] = True, str(args.checkpoint)
    native = gym.make(TASK, cfg=cfg)
    env = None
    results = []
    started = perf_counter()
    try:
        settings = params.get('env', {})
        wrapped = RlGamesVecEnvWrapper(native, cfg.sim.device,
            settings.get('clip_observations', float('inf')), settings.get('clip_actions', 1.0))
        vecenv.register('IsaacRlgWrapper', lambda name, count, **kw: RlGamesGpuEnv(name, count, **kw))
        env_configurations.register('rlgpu', {'vecenv_type': 'IsaacRlgWrapper',
                                            'env_creator': lambda **kw: wrapped})
        runner = Runner()
        runner.load(agent_cfg)
        player = runner.create_player()
        player.restore(str(args.checkpoint))
        policy = NativePPOPolicy(player, cfg.sim.device)
        env = SingleFactoryEnv(wrapped)
        write_json(args.output_dir / 'protocol.json', {
            'task': TASK, 'seeds': args.seeds, 'checkpoint_sha256': sha256(args.checkpoint),
            'entry_sha256': sha256(Path(__file__)),
            'versions': {name: version(name) for name in ('isaaclab', 'isaacsim', 'rl-games', 'torch')},
            'source_sha256': sha256(Path(inspect.getfile(type(native.unwrapped)))),
            'task_cfg_sha256': sha256(Path(inspect.getfile(type(cfg)))),
            'agent_config': agent_cfg, 'physics_dt': env.native.physics_dt,
            'control_dt': env.native.step_dt, 'configured_max_steps': env.native.max_episode_length,
            'actor_dim': cfg.observation_space, 'critic_dim': cfg.state_space,
            'position_delta_scale_m': cfg.ctrl.pos_action_threshold,
            'rotation_delta_scale_rad': cfg.ctrl.rot_action_threshold,
            'native_action_ema_factor': cfg.ctrl.ema_factor,
            'policy_stochastic': not player.is_deterministic,
            'contact_force_measurement': {'enabled': args.contacts, 'paths': env.contact_metadata,
                'sampling': 'last physics substep of each control; 15Hz, not120Hz peaks',
                'quantity': 'aggregated world normal-force vectors; tangential friction excluded'},
            'success': 'native final and ever-success reported separately; no early stop'})
        for seed in args.seeds:
            observations, actions, rows = [], [], []

            def observe(observation, action, info):
                observations.append(observation)
                actions.append(action)
                rows.append(info)
                if len(rows) % 50 == 0:
                    print(f"seed={seed} step={len(rows)} success={info['success']} "
                          f"xy_mm={1000 * info['xy_error_m']:.3f} "
                          f"z_mm={1000 * info['z_error_m']:.3f}", flush=True)

            begin = perf_counter()
            result = run_episode(env, policy, seed=seed, on_step=observe)
            observations = [env.initial_observation, *observations]
            trace = {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}
            np.savez_compressed(args.output_dir / f'episode_seed{seed}.npz', **trace,
                actions=np.array(actions), clipped_actions=np.clip(actions, -1., 1.),
                success=np.array([row['success'] for row in rows]),
                rewards=np.array([row['native_reward'] for row in rows]),
                native_terminated=np.array([row['native_terminated'] for row in rows]),
                native_truncated=np.array([row['native_truncated'] for row in rows]),
                xy_error_m=np.array([row['xy_error_m'] for row in rows]),
                z_error_m=np.array([row['z_error_m'] for row in rows]),
                timestamps=np.arange(len(actions) + 1) * env.native.step_dt,
                inference_seconds=np.array(policy.latency_s))
            successes = [i + 1 for i, row in enumerate(rows) if row['success']]
            report = {'seed': seed, **asdict(result), 'ever_success': bool(successes),
                      'first_success_step': successes[0] if successes else None,
                      'final_xy_error_m': rows[-1]['xy_error_m'],
                      'final_z_error_m': rows[-1]['z_error_m'],
                      'inference_mean_s': float(np.mean(policy.latency_s)),
                      'wall_seconds': perf_counter() - begin}
            if args.contacts:
                norms = np.linalg.norm(trace['pair_normal_force_w_N'][1:], axis=-1)
                gear_norms = norms[:, :, :env.contact_metadata['gear_filter_count']]
                report['sampled_gear_normal_vector_peak_N'] = float(gear_norms.max())
                report['sampled_gripper_normal_vector_peak_N'] = float(norms[:, :, -2:].max())
                report['sampled_gear_contact_fraction_gt0_1N'] = float((gear_norms > .1).any(axis=(1, 2)).mean())
            results.append(report)
            write_json(args.output_dir / 'results.json', {'status': 'running', 'episodes': results})
            print(json.dumps(report), flush=True)
        write_json(args.output_dir / 'results.json', {'status': 'complete', 'episodes': results,
            'success_rate': float(np.mean([r['success'] for r in results])),
            'ever_success_rate': float(np.mean([r['ever_success'] for r in results])),
            'wall_seconds': perf_counter() - started})
    except Exception as error:
        write_json(args.output_dir / 'error.json', {'status': 'incomplete', 'episodes': results,
                                                   'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        env.close() if env is not None else native.close()


try:
    main()
finally:
    app.close()
