"""Compare two state-feedback controllers on native robosuite Wipe."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from embodiedlab.benchmark import wilson_interval
from embodiedlab.envs.wipe_env import WipeEnv
from embodiedlab.policies.wipe import WipePolicy
from embodiedlab.runners.rollout import run_episode

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: dict) -> None:
    """Write measured results as JSON; non-finite metrics are never accepted."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def recovery_time(contact: np.ndarray, start: int, dt: float) -> float | None:
    """Time to the first three consecutive qualifying post-disturbance steps."""
    for index in range(start, len(contact) - 2):
        if contact[index:index + 3].all():
            return float((index - start + 1) * dt)
    return None


def measure_episode(env: WipeEnv, policy: WipePolicy, seed: int,
                    directory: Path, video: bool) -> dict:
    """Use the existing Runner; store task-specific numeric traces without RGB NPZ."""
    directory.mkdir(parents=True, exist_ok=False)
    observations, actions, rows, phases, substep_forces = [], [], [], [], []
    force_estimates, z_references = [], []
    writer = None
    if video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(directory / 'episode.mp4', fps=20,
                                    codec='libx264', pixelformat='yuv420p')

    def observe(observation: dict, action: np.ndarray, info: dict) -> None:
        if writer is not None:
            if not rows:
                writer.append_data(env.initial_frame)
            writer.append_data(env.frame())
        observations.append(observation)
        actions.append(action)
        rows.append(info)
        phases.append(policy.phase)
        substep_forces.append(env.force_substeps.copy())
        force_estimates.append(policy.force_estimate_N)
        z_references.append(policy.z_reference)

    started = perf_counter()
    try:
        result = run_episode(env, policy, seed=seed, on_step=observe)
    except Exception as error:
        write_json(directory / 'error.json', {'status': 'incomplete', 'success': None,
                   'completed_steps': len(rows), 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        if writer is not None:
            writer.close()
    wall = perf_counter() - started
    observations = [env.initial_observation, *observations]
    force = np.asarray(substep_forces)
    wiping = np.array(phases) == 'wipe'
    active = np.array([r['pulse_active'] for r in rows])
    contact = np.array([r['contact_fraction'] for r in rows])
    # Native marker observables are cached before reward() changes wiped_markers.
    # Preserve those actual policy inputs, but count erase events from post-reward
    # native info rather than shifting the cached observation heuristically.
    counts = np.r_[0, [r['native_wiped_count'] for r in rows]]
    gained = np.diff(counts)
    measured = force[wiping]
    pulse_indices = np.flatnonzero(active)
    loss = bool(len(pulse_indices) and np.any(contact[active] < .2))
    recovery = pressure_recovery = None
    if len(pulse_indices):
        end = int(pulse_indices[-1]) + 1
        if loss:
            recovery = recovery_time(contact >= .8, end, env.control_dt)
        means = force.mean(axis=1)
        pressure_recovery = recovery_time(np.abs(means - policy.target_force_N) <= 2., end, env.control_dt)
    metrics = {'native_success': result.success, 'wipe_fraction': rows[-1]['wipe_fraction'],
        'failure_reason': rows[-1]['failure_reason'],
        'normal_peak_N': float(force.max()),
        'wiping_contact_fraction_gt1N': float(np.mean(measured > 1.)) if measured.size else None,
        'wiping_mean_normal_N': float(measured.mean()) if measured.size else None,
        'wiping_force_rmse_N': float(np.sqrt(np.mean((measured - policy.target_force_N) ** 2))) if measured.size else None,
        'physics_fraction_gt60N': float(np.mean(force > 60.)),
        'markers_cleared_without_gt1N_contact_in_control_interval': int(gained[contact == 0].sum()),
        'pulse_applied': bool(len(pulse_indices)), 'contact_lost_during_pulse': loss,
        'contact_recovery_s': recovery, 'force_recovery_to_8plusminus2N_s': pressure_recovery,
        'first_contact_step': env.first_contact_step,
        'pulse_start_step_zero_based': env.pulse_start_step,
        'wall_seconds': wall, 'wiping_control_steps': int(wiping.sum())}
    np.savez_compressed(directory / 'trajectory.npz', actions=np.array(actions),
        eef_pos=np.array([o['eef_pos'] for o in observations]),
        eef_quat=np.array([o['eef_quat'] for o in observations]),
        wrist_force_world_N=np.array([o['wrist_force_world_N'] for o in observations]),
        marker_xy=observations[0]['marker_xy'],
        marker_wiped=np.array([o['marker_wiped'] for o in observations]),
        native_wiped_count=counts,
        normal_force_substeps_N=force, phases=np.array(phases), pulse_active=active,
        policy_force_estimate_N=np.array(force_estimates), z_reference_m=np.array(z_references),
        rewards=np.array([r['native_reward'] for r in rows]),
        timestamps=np.arange(len(rows) + 1) * env.control_dt)
    report = {'seed': seed, 'policy': policy.mode, 'scenario': env.scenario,
              'status': 'complete', **asdict(result), 'metrics': metrics,
              'environment': env.episode_metadata(),
              'record_path': str(directory / 'trajectory.npz')}
    write_json(directory / 'result.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policies', nargs='+', choices=('touch_position', 'force_admittance'),
                        default=['touch_position', 'force_admittance'])
    parser.add_argument('--scenarios', nargs='+', choices=('nominal', 'height_variation', 'upward_pulse'),
                        default=['nominal'])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--videos', action='store_true', help='Record only the first seed in each policy/scenario')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if (len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds)
            or len(set(args.policies)) != len(args.policies)
            or len(set(args.scenarios)) != len(args.scenarios)):
        parser.error('Use unique policies/scenarios and nonnegative unique seeds')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = ['src/embodiedlab/envs/wipe_env.py', 'src/embodiedlab/policies/wipe.py',
               'scripts/evaluate_wipe.py', 'src/embodiedlab/runners/rollout.py']
    protocol = {'task': 'native robosuite 1.4 Wipe/Panda/WipingGripper',
        'code_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'seeds': args.seeds, 'policies': args.policies, 'scenarios': args.scenarios,
        'horizon': args.horizon, 'target_force_N': 8.,
        'controller': 'unmodified native fixed-impedance OSC_POSE, kp150/damping_ratio1',
        'policy_scope': 'traditional oracle-state baselines, not pretrained models',
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        'height_variation': 'native min(0,Normal(0,0.01m)); hidden from both policies',
        'pulse': '12N upward, 0.5s at tool CoM, 30 steps after first >1N contact',
        'force': '25 native 2ms contact samples per 20Hz control step',
        'terminated': 'native success OR arm collision OR joint limit',
        'truncated': 'native horizon, unless already terminated',
        'records': 'task-specific NPZ; T actions, T+1 poses/forces/cached flags/native wiped counts; no pickle or RGB arrays',
        'wipe_event_alignment': 'post-reward native counts; do not difference the delayed marker observable'}
    write_json(args.output_dir / 'protocol.json', protocol)
    all_results, summaries = [], []
    started = perf_counter()
    for scenario in args.scenarios:
        for seed in args.seeds:
            hashes = []
            for mode in args.policies:
                video = args.videos and seed == args.seeds[0]
                env, policy = WipeEnv(scenario=scenario, horizon=args.horizon, render=video), WipePolicy(mode)
                try:
                    directory = args.output_dir / scenario / mode / f'seed_{seed}'
                    result = measure_episode(env, policy, seed, directory, video)
                    hashes.append(result['environment']['initial_state_sha256'])
                    all_results.append(result)
                    print(scenario, mode, seed, result['success'], result['length'],
                          json.dumps(result['metrics']), flush=True)
                    write_json(args.output_dir / 'results.json', {'status': 'running', 'episodes': all_results})
                finally:
                    env.close()
            if len(set(hashes)) != 1:
                raise RuntimeError('Policies were not evaluated from the same seeded native state')
    for scenario in args.scenarios:
        for mode in args.policies:
            cases = [r for r in all_results if r['scenario'] == scenario and r['policy'] == mode]
            successes = sum(r['success'] for r in cases)
            summary = {'scenario': scenario, 'policy': mode, 'successes': successes,
                       'episodes': len(cases), 'success_rate': successes / len(cases),
                       'wilson95': wilson_interval(successes, len(cases)),
                       'mean_length': float(np.mean([r['length'] for r in cases]))}
            for metric in ('wipe_fraction', 'wiping_contact_fraction_gt1N', 'wiping_mean_normal_N',
                           'wiping_force_rmse_N', 'normal_peak_N'):
                values = [r['metrics'][metric] for r in cases if r['metrics'][metric] is not None]
                summary[metric] = float(np.mean(values)) if values else None
            summaries.append(summary)
    write_json(args.output_dir / 'results.json', {'status': 'complete',
        'wall_seconds': perf_counter() - started, 'summaries': summaries, 'episodes': all_results})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
