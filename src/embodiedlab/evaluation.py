"""Explicit LIBERO task evaluation with immutable per-run JSON reports."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Sequence
from uuid import uuid4

import numpy as np

from embodiedlab.policies.simple import RandomPolicy, ZeroPolicy
from embodiedlab.interfaces import Policy
from embodiedlab.runners.rollout import StepObserver, run_episode

if TYPE_CHECKING:
    from embodiedlab.envs.libero_env import LiberoEnv


def evaluate_task(env: 'LiberoEnv', *, policy_name: str = 'zero', episodes: int = 1,
                  seed: int = 0, record_dir: Path | None = None,
                  policy: Policy | None = None, on_step: StepObserver | None = None,
                  initial_state_indices: Sequence[int] | None = None) -> dict:
    """Evaluate official states 0..episodes-1 with seeds seed..seed+episodes-1.

    Explicit indices preserve the requested order; seed increments by episode
    ordinal, not by state index. A supplied policy is reset and reused each
    episode; policy_name labels it.
    Otherwise construct the existing zero/random interface-check policies.
    """
    if type(episodes) is not int or not 1 <= episodes <= env.initial_state_count:
        raise ValueError(f'episodes must be between 1 and {env.initial_state_count}')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    indices = list(range(episodes)) if initial_state_indices is None else list(initial_state_indices)
    if (len(indices) != episodes
            or any(type(index) is not int or not 0 <= index < env.initial_state_count for index in indices)
            or len(set(indices)) != episodes):
        raise ValueError('initial_state_indices must contain episodes unique valid official state indices')
    if policy is None and policy_name not in ('zero', 'random'):
        raise ValueError('policy_name must be zero or random')
    # Policies with a saved task/I/O contract can reject a mismatched environment
    # even when callers use the Python evaluator instead of the CLI.
    check_compatibility = getattr(policy, 'assert_compatible', None)
    if check_compatibility is not None:
        check_compatibility(env)
    if record_dir is not None:
        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
    episodes_result = []
    for index, state_index in enumerate(indices):
        episode_seed = seed + index
        episode_policy = policy if policy is not None else (
            RandomPolicy(episode_seed) if policy_name == 'random' else ZeroPolicy())
        recorder = None
        record_path = None
        if record_dir is not None:
            from embodiedlab.data.episode import EpisodeRecorder
            record_path = record_dir / f'episode_{index:03d}.hdf5'
            recorder = EpisodeRecorder(record_path)
        started = perf_counter()
        result = run_episode(env, episode_policy, seed=episode_seed,
                             init_state_index=state_index, recorder=recorder, on_step=on_step)
        episodes_result.append({
            'episode_index': index, 'seed': episode_seed,
            **asdict(result),
            'first_success_step': result.length if result.success else None,
            'wall_seconds': perf_counter() - started,
            'record_path': str(record_path.resolve()) if record_path is not None else None,
        })
    return {
        'suite': env.suite, 'task_id': env.task_id, 'task_name': env.task_name,
        'language': env.language, 'policy': policy_name,
        'settings': env.episode_metadata(),
        'episodes': episodes_result,
        'success_rate': float(np.mean([r['success'] for r in episodes_result])),
        'mean_reward': float(np.mean([r['reward'] for r in episodes_result])),
        'mean_length': float(np.mean([r['length'] for r in episodes_result])),
    }


def _write_json(path: Path, payload: dict) -> None:
    """Create one report exclusively; never replace a previous experiment."""
    with path.open('x', encoding='utf-8') as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def evaluate_suites(*, suites: Sequence[str], task_ids: Sequence[int] | None = (0,),
                    episodes: int = 1, seed: int = 0, policy_name: str = 'zero',
                    output_dir: Path = Path('evaluation_results'),
                    record_dir: Path | None = None, image_size: int = 128,
                    max_episode_steps: int | None = None,
                    settle_steps: int = 10) -> tuple[dict, Path]:
    """Evaluate selected tasks, closing each environment, and save settings/results.

    ``task_ids=None`` explicitly evaluates every task of each requested suite.
    Default task selection is only task 0, suitable for an interface smoke test.
    """
    if not suites or len(set(suites)) != len(suites):
        raise ValueError('suites must be nonempty and contain no duplicates')
    if task_ids is not None and (not task_ids or len(set(task_ids)) != len(task_ids)
                               or any(type(i) is not int or i < 0 for i in task_ids)):
        raise ValueError('task_ids must be nonnegative integers without duplicates')
    if type(episodes) is not int or episodes < 1:
        raise ValueError('episodes must be a positive integer')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if policy_name not in ('zero', 'random'):
        raise ValueError('policy_name must be zero or random')

    from embodiedlab.envs.libero_env import LiberoEnv, list_task_ids

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    records_run = None
    if record_dir is not None:
        records_run = Path(record_dir) / run_id
        records_run.mkdir(parents=True, exist_ok=False)
    results = []
    task_settings = []
    for suite in suites:
        available = list_task_ids(suite)
        selected = available if task_ids is None else list(task_ids)
        for task_id in selected:
            if task_id not in available:
                raise ValueError(f'Task {task_id} does not exist in {suite}')
            env = LiberoEnv(suite=suite, task_id=task_id, image_size=image_size,
                            max_episode_steps=max_episode_steps, settle_steps=settle_steps)
            try:
                task_records = records_run / suite / f'task_{task_id:03d}' if records_run is not None else None
                result = evaluate_task(env, policy_name=policy_name, episodes=episodes,
                                       seed=seed, record_dir=task_records)
                task_settings.append(result.pop('settings'))
                results.append(result)
            finally:
                env.close()
    settings = {
        'run_id': run_id, 'suites': list(suites),
        'task_ids': list(task_ids) if task_ids is not None else 'all',
        'episodes_per_task': episodes, 'seed': seed, 'policy': policy_name,
        'image_size': image_size, 'max_episode_steps_override': max_episode_steps,
        'settle_steps': settle_steps, 'tasks': task_settings,
    }
    all_episodes = [episode for task in results for episode in task['episodes']]
    report = {
        'run_id': run_id, 'tasks': results,
        'episode_count': len(all_episodes),
        'success_rate': float(np.mean([episode['success'] for episode in all_episodes])),
    }
    _write_json(run_dir / 'settings.json', settings)
    _write_json(run_dir / 'results.json', report)
    return report, run_dir


def task_diagnostics(env: 'LiberoEnv') -> dict:
    """Read object joints and robot/external contacts at a control-step endpoint."""
    import mujoco

    model, data = env._env.sim.model, env._env.sim.data
    joints = {}
    for index in range(model.njnt):
        name = model.joint_id2name(index)
        if name and not name.startswith(('robot', 'gripper')):
            start = model.jnt_qposadr[index]
            width = {0: 7, 1: 4, 2: 1, 3: 1}[int(model.jnt_type[index])]
            joints[name] = data.qpos[start:start + width].tolist()
    robot = env._env.robots[0]
    robot_ids = {model.geom_name2id(name) for name in
                 (*robot.robot_model.contact_geoms, *robot.gripper.contact_geoms)}
    contacts = []
    for index, contact in enumerate(data.contact[:data.ncon]):
        if (contact.geom1 in robot_ids) != (contact.geom2 in robot_ids):
            force = np.zeros(6)
            mujoco.mj_contactForce(model._model, data._data, index, force)
            contacts.append({'geoms': [model.geom_id2name(i) or str(i)
                                      for i in (contact.geom1, contact.geom2)],
                             'normal_force_N': float(abs(force[0]))})
    domain = env._env.env
    # Evaluate the exact native goal atoms for diagnostics only. This neither
    # defines new success criteria nor supplies privileged state to a policy.
    predicates = [{'predicate': list(goal),
                   'satisfied': bool(domain._eval_predicate(goal))}
                  for goal in domain.parsed_problem['goal_state']]
    return {'object_joint_positions': joints, 'robot_external_contacts': contacts,
            'native_goal_predicates': predicates}


def save_preview(record_path: Path, target: Path) -> None:
    """Six paired frames; rotate only display 180 degrees, never recorded pixels."""
    from PIL import Image, ImageDraw
    from embodiedlab.data.episode import EpisodeDataset

    with EpisodeDataset(record_path) as episode:
        steps = sorted(set(np.linspace(0, len(episode), 6, dtype=int).tolist()))
        canvas = Image.new('RGB', (256 * len(steps), 552), 'white')
        draw = ImageDraw.Draw(canvas)
        for column, step in enumerate(steps):
            observation = episode.observation(step)
            draw.text((column * 256 + 5, 4), f'step {step}', fill='black')
            for row, key in enumerate(('agentview_image', 'wrist_image')):
                pixels = observation[key][::-1, ::-1]
                canvas.paste(Image.fromarray(pixels).resize((256, 256)),
                             (column * 256, 24 + row * 264))
        canvas.save(target)
