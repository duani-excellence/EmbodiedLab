"""Read frozen formal challenge results; never run a model or modify evidence.

Usage: python tools/analyze_challenge.py --manifest benchmark_manifest.json
    --openpi-dir evaluation_results/phase6b_formal_openpi
    --openvla-dir evaluation_results/phase6b_formal_openvla --output-dir NEW_DIR

Missing models/tasks have no observed denominator. Only tasks with every planned
episode complete enter the fixed-manifest macro. Optional videos read one paired
frame at a time; display rotation never changes the HDF5 recording.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from embodiedlab.benchmark import summarize_predicates, summarize_task
from embodiedlab.data.episode import EpisodeDataset

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('openpi', 'openvla_oft')
LATENCY_FIELDS = {'rpc': ('requests', 'rpc_ms'),
                  'synchronized_sample': ('requests', 'synchronized_sample_ms'),
                  'act': ('steps', 'act_ms'), 'envstep': ('steps', 'env_step_ms')}
NOTES = [
    'Success is the recorded unmodified native environment success; incomplete/errors '
    'and unattempted episodes are excluded from the observed success denominator.',
    'Fixed-manifest macro is reported only when all planned tasks and episodes are complete. '
    'Observed-task macro may cover partial tasks and is not the full benchmark.',
    'Wilson 95% intervals describe individual task proportions. No significance test or '
    'overall model superiority is inferred from these small samples.',
    'Predicate first satisfaction/loss refers to observed post-control-step rows, '
    'not an inferred grasp stage or an unobserved simulator transition. Reset predicates '
    'are not logged; a close predicate true at the first observation and later false can '
    'be normal opening during a drawer/microwave task.',
    'Latency percentiles are recomputed from actual diagnostic samples, including '
    'available rows in incomplete attempts; no averaging of episode percentiles. '
    'Act/envstep samples cover recorded executed control steps; an act call failing '
    'before a step may have no step timing. RPC includes transfer '
    'and conversion; act includes RPC only when replanning; envstep excludes observers '
    'and recording. synchronized_sample uses only explicitly recorded synchronized timing.',
    'Initial-state validation checks the exact frozen protocol/task (including state '
    'hash references), episode seed/index, and completed HDF5 metadata. Recordings do '
    'not contain the official reset-state vector, so this is not a new hash verification '
    'of the original initial-state asset.',
]


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


def require_equal(actual, expected, context: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f'{context}: expected {expected!r}, observed {actual!r}')


def validate_manifest(plan: dict) -> None:
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Manifest must contain nonempty tasks')
    seen = set()
    for task in tasks:
        key = (task['suite'], task['task_id'])
        if (not isinstance(key[0], str) or not key[0] or '/' in key[0]
                or type(key[1]) is not int or key[1] < 0 or key in seen):
            raise ValueError('Manifest task identities must be unique and valid')
        seen.add(key)
        formal = task['formal']
        indices, seeds = formal['initial_state_indices'], formal['seeds']
        if (not isinstance(indices, list) or not isinstance(seeds, list) or not indices
                or len(indices) != len(seeds) or len(set(indices)) != len(indices)
                or any(type(item) is not int or item < 0 for item in indices + seeds)):
            raise ValueError('Formal initial-state indices/seeds must be nonnegative aligned lists')
        if type(task['max_episode_steps']) is not int or task['max_episode_steps'] < 1:
            raise ValueError('Manifest horizon must be a positive integer')
        goals = task['native_goal_predicates']
        if (not isinstance(goals, list) or not goals
                or any(not isinstance(goal, list) or not goal
                       or any(not isinstance(part, str) or not part for part in goal)
                       for goal in goals) or len({tuple(goal) for goal in goals}) != len(goals)):
            raise ValueError('Manifest native predicates must be unique nonempty string lists')
    if 'task_order' in plan:
        require_equal(plan['task_order'], [[task['suite'], task['task_id']] for task in tasks],
                      'Manifest task_order')


def latency_summary(values: list[float], expected_count: int) -> dict:
    """Aggregate measured samples, exposing absent measurements explicitly."""
    result = {'count': len(values), 'expected_count': expected_count,
              'missing_count': expected_count - len(values),
              'first_ms': None, 'mean_ms': None, 'p50_ms': None,
              'p95_ms': None, 'max_ms': None}
    if values:
        result.update(first_ms=float(values[0]), mean_ms=float(np.mean(values)),
                      p50_ms=float(np.median(values)),
                      p95_ms=float(np.percentile(values, 95)), max_ms=float(max(values)))
    return result


def measured_values(rows: list[dict], field: str, context: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{context}: {field} must be finite and nonnegative or null')
        values.append(float(value))
    return values


def validate_protocol(protocol: dict, task: dict, model: str, digest: str,
                      model_plan: dict) -> None:
    benchmark = protocol.get('benchmark')
    if not isinstance(benchmark, dict):
        raise ValueError('Formal results need a frozen benchmark protocol')
    for field, expected in [('sha256', digest), ('phase', 'formal'), ('task', task)]:
        require_equal(benchmark.get(field), expected, f'Protocol benchmark {field}')
    for field, expected in [('policy', model), ('suite', task['suite']),
                            ('task_id', task['task_id']), ('language', task['language']),
                            ('initial_state_indices', task['formal']['initial_state_indices']),
                            ('seeds', task['formal']['seeds']),
                            ('max_episode_steps', task['max_episode_steps'])]:
        require_equal(protocol.get(field), expected, f'Protocol {field}')
    for field in ('replan_steps', 'action_horizon'):
        if field in model_plan:
            require_equal(protocol.get(field), model_plan[field], f'Protocol {field}')


def validate_record(path: Path, episode: dict, task: dict, environment: dict) -> dict:
    """EpisodeDataset validates T+1 alignment without loading any RGB sequence."""
    with EpisodeDataset(path) as recorded:
        metadata = recorded.metadata
        for field, expected in [('suite', task['suite']), ('task_id', task['task_id']),
                                ('language', task['language']), ('seed', episode['seed']),
                                ('init_state_index', episode['init_state_index']),
                                ('max_episode_steps', task['max_episode_steps']),
                                ('success', episode['success'])]:
            require_equal(metadata.get(field), expected, f'{path}: metadata {field}')
        for field in ('settle_steps', 'control_dt'):
            if field in environment:
                require_equal(metadata.get(field), environment[field], f'{path}: metadata {field}')
        require_equal(len(recorded), episode['length'], f'{path}: T')
        require_equal(bool(recorded.terminated[-1]), episode['success'], f'{path}: terminated')
        require_equal(bool(recorded.truncated[-1]), episode['truncated'], f'{path}: truncated')
        return {'validated': True, 'transitions': len(recorded),
                'observations': len(recorded) + 1, 'rgb_loading': 'lazy, no sequence read'}


def analyze_episode(episode: dict, ordinal: int, task: dict, directory: Path,
                    environment: dict) -> tuple[dict, dict[str, list[float]], dict[str, int]]:
    """Validate one manifest-aligned episode and summarize only logged evidence."""
    planned = task['formal']
    for field, expected in [('episode_index', ordinal),
                            ('init_state_index', planned['initial_state_indices'][ordinal]),
                            ('seed', planned['seeds'][ordinal])]:
        require_equal(episode.get(field), expected, f'{directory}: episode {ordinal} {field}')
    summarize_task([episode])  # Shared strict success / denominator semantics.
    complete = episode['status'] == 'complete'
    if not complete and episode.get('success') is not None:
        raise ValueError('Incomplete episodes must have null success')
    if complete:
        if not 1 <= episode['length'] <= task['max_episode_steps']:
            raise ValueError('Complete episode length outside the frozen horizon')
        require_equal(episode.get('terminated'), episode['success'], 'Episode terminated')
        require_equal(episode.get('truncated'), not episode['success'], 'Episode truncated')
        if episode['truncated'] and episode['length'] != task['max_episode_steps']:
            raise ValueError('Timeout must end at the frozen horizon')
    diagnostics_path = directory / f'episode_{ordinal:03d}_diagnostics.json'
    diagnostics = read_json(diagnostics_path)
    steps, requests = diagnostics.get('steps'), diagnostics.get('requests')
    if not isinstance(steps, list) or not isinstance(requests, list):
        raise ValueError(f'{diagnostics_path}: steps and requests must be lists')
    expected_length = episode['length'] if complete else episode.get('completed_steps', len(steps))
    require_equal(len(steps), expected_length, 'Diagnostic step count')
    require_equal([row.get('step') for row in steps], list(range(1, len(steps) + 1)),
                  'Diagnostics must log each post-control step')
    expected_goals = {tuple(goal) for goal in task['native_goal_predicates']}
    predicate_progress = summarize_predicates(steps)
    if steps and {tuple(goal['predicate']) for goal in predicate_progress} != expected_goals:
        raise ValueError('Diagnostic native predicates differ from the manifest')
    successful_steps = []
    for row in steps:
        success = row.get('native_success')
        if type(success) is not bool:
            raise ValueError('Diagnostic native_success must be boolean')
        require_equal(success, all(goal['satisfied'] for goal in row['native_goal_predicates']),
                      'Native success / native goal conjunction')
        if success:
            successful_steps.append(row['step'])
    if complete:
        require_equal(bool(successful_steps), episode['success'], 'Diagnostic outcome')
        require_equal(successful_steps[0] if successful_steps else None,
                      episode.get('first_success_step'), 'Diagnostic first success')
        if successful_steps and successful_steps != [episode['length']]:
            raise ValueError('Runner must stop at the first native success')
    require_equal(episode.get('inference_requests'), len(requests), 'Inference request count')
    request_steps = [row.get('request_step') for row in requests]
    if (any(type(step) is not int or not 0 <= step <= len(steps) for step in request_steps)
            or any(left >= right for left, right in zip(request_steps, request_steps[1:]))):
        raise ValueError('Request steps must be increasing control-step indices')
    samples, expected_counts = {}, {}
    for name, (source, field) in LATENCY_FIELDS.items():
        rows = diagnostics[source]
        samples[name] = measured_values(rows, field, str(diagnostics_path))
        expected_counts[name] = len(rows)
    output = {key: episode.get(key) for key in (
        'episode_index', 'init_state_index', 'seed', 'status', 'success', 'length',
        'first_success_step', 'terminated', 'truncated', 'error')}
    output.update(inference_requests=len(requests), completed_steps=len(steps),
                  diagnostics_path=str(diagnostics_path.resolve()),
                  native_predicate_progress=predicate_progress,
                  latency={name: latency_summary(samples[name], expected_counts[name])
                           for name in LATENCY_FIELDS})
    final_unsatisfied = [goal['predicate'] for goal in predicate_progress if not goal['final_satisfied']]
    lost = [goal['predicate'] for goal in predicate_progress if goal['ever_lost_after_satisfied']]
    output['evidence'] = {
        'classification': ('success' if complete and episode['success'] else
                           'timeout_with_final_unsatisfied_native_atoms' if complete else
                           'incomplete_error_excluded'),
        'final_unsatisfied_native_atoms': final_unsatisfied,
        'observed_atoms_ever_lost_after_satisfaction': lost,
        'close_predicate_caveat': any(goal['predicate'][0] == 'close'
            and goal['first_satisfied_step'] == 1 and goal['ever_lost_after_satisfied']
            for goal in predicate_progress),
    }
    if complete:
        record = episode.get('record_path')
        if not isinstance(record, str) or not record:
            raise ValueError('Complete episodes must name their actual HDF5 record')
        record_path = Path(record)
        if not record_path.is_absolute():
            record_path = directory / record_path
        output['record_path'] = str(record_path.resolve())
        output['record_validation'] = validate_record(record_path, episode, task, environment)
    else:
        output['record_validation'] = {'validated': False, 'reason': 'incomplete episode'}
    return output, samples, expected_counts


def analyze_challenge(manifest_path: Path, openpi_dir: Path | None = None,
                      openvla_dir: Path | None = None) -> dict:
    """Build a read-only report; absent inputs never create fictitious failures."""
    manifest_path = Path(manifest_path)
    plan = read_json(manifest_path)
    validate_manifest(plan)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report = {'schema_version': 1, 'benchmark_id': plan['benchmark_id'], 'phase': 'formal',
              'manifest_path': str(manifest_path.resolve()), 'manifest_sha256': digest,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'notes': NOTES,
              'models': {}, 'paired_initial_states': [], 'videos': []}
    for model, supplied in zip(MODELS, (openpi_dir, openvla_dir)):
        directory = Path(supplied) if supplied is not None else None
        available = directory is not None and directory.is_dir()
        if directory is not None and directory.exists() and not directory.is_dir():
            raise ValueError(f'Model input is not a directory: {directory}')
        model_report = {'status': 'not_run',
                        'input_dir': str(directory.resolve()) if directory else None,
                        'input_directory_present': available, 'tasks': []}
        all_samples = {name: [] for name in LATENCY_FIELDS}
        all_expected = {name: 0 for name in LATENCY_FIELDS}
        for task in plan['tasks']:
            count = len(task['formal']['initial_state_indices'])
            summary = {'suite': task['suite'], 'task_id': task['task_id'],
                       'language': task['language'], 'requested': count, 'status': 'not_run',
                       'results_path': None, 'episodes': [], **summarize_task([]),
                       'unattempted': count}
            files = sorted((directory / task['suite'] / f'task_{task["task_id"]}').glob('*/results.json')) if available else []
            if len(files) > 1:
                raise ValueError(f'{model} {task["suite"]}/{task["task_id"]}: multiple results; '
                                 'refusing outcome-based run selection')
            samples = {name: [] for name in LATENCY_FIELDS}
            expected = {name: 0 for name in LATENCY_FIELDS}
            if files:
                path = files[0]
                result = read_json(path)
                for field, value in [('suite', task['suite']), ('task_id', task['task_id']),
                                     ('requested', count)]:
                    require_equal(result.get(field), value, f'{path}: {field}')
                validate_protocol(read_json(path.parent / 'protocol.json'), task, model, digest,
                                  plan.get('models', {}).get(model, {}))
                episodes = result.get('episodes')
                if not isinstance(episodes, list) or len(episodes) > count:
                    raise ValueError(f'{path}: episode count exceeds the frozen plan')
                for ordinal, episode in enumerate(episodes):
                    row, episode_samples, episode_expected = analyze_episode(
                        episode, ordinal, task, path.parent, plan.get('environment', {}))
                    summary['episodes'].append(row)
                    for name in LATENCY_FIELDS:
                        samples[name].extend(episode_samples[name])
                        expected[name] += episode_expected[name]
                statistics = summarize_task(episodes)
                for field, value in [('attempted', len(episodes)), ('completed', statistics['completed']),
                                     ('incomplete', statistics['incomplete']),
                                     ('unattempted', count - len(episodes)),
                                     ('success_rate_denominator', statistics['completed'])]:
                    require_equal(result.get(field), value, f'{path}: {field}')
                done = statistics['completed'] == count
                if result.get('status') not in ('complete', 'incomplete'):
                    raise ValueError(f'{path}: run status must be complete or incomplete')
                if result['status'] == 'complete' and not done:
                    raise ValueError(f'{path}: complete run is missing completed episodes')
                # Cleanup/close can fail after every episode is complete. Keep those
                # observed outcomes, expose the run error, and withhold the full macro.
                summary.update(statistics,
                               status='complete' if done and result['status'] == 'complete' else 'partial',
                               source_run_status=result['status'], source_run_error=result.get('error'),
                               results_path=str(path.resolve()), unattempted=count - len(episodes))
            summary['inference_requests'] = sum(row['inference_requests'] for row in summary['episodes'])
            summary['latency'] = {name: latency_summary(samples[name], expected[name]) for name in LATENCY_FIELDS}
            for name in LATENCY_FIELDS:
                all_samples[name].extend(samples[name])
                all_expected[name] += expected[name]
            model_report['tasks'].append(summary)
        tasks = model_report['tasks']
        complete_tasks = [task for task in tasks if task['status'] == 'complete']
        observed_tasks = [task for task in tasks if task['completed']]
        has_results = any(task['results_path'] is not None for task in tasks)
        model_report['status'] = ('complete' if len(complete_tasks) == len(tasks) else
                                  'partial' if has_results else 'not_run')
        model_report['macro'] = {
            'planned_tasks': len(tasks), 'tasks_with_results': sum(task['results_path'] is not None for task in tasks),
            'tasks_with_completed_episodes': len(observed_tasks), 'fully_completed_tasks': len(complete_tasks),
            'planned_episodes': sum(task['requested'] for task in tasks),
            'completed_episodes': sum(task['completed'] for task in tasks),
            'incomplete_episodes': sum(task['incomplete'] for task in tasks),
            'unattempted_episodes': sum(task['unattempted'] for task in tasks),
            'successes': sum(task['successes'] for task in tasks),
            'per_task_denominators': [{'suite': task['suite'], 'task_id': task['task_id'],
                                      'completed': task['completed'], 'planned': task['requested']}
                                     for task in tasks],
            'fixed_manifest_task_macro_success_rate': (
                sum(task['success_rate'] for task in tasks) / len(tasks)
                if len(complete_tasks) == len(tasks) else None),
            'observed_task_macro_success_rate': (
                sum(task['success_rate'] for task in observed_tasks) / len(observed_tasks)
                if observed_tasks else None),
        }
        model_report['inference_requests'] = sum(task['inference_requests'] for task in tasks)
        model_report['latency'] = {name: latency_summary(all_samples[name], all_expected[name]) for name in LATENCY_FIELDS}
        report['models'][model] = model_report
    for task_number, task in enumerate(plan['tasks']):
        lookups = {model: {row['init_state_index']: row for row in report['models'][model]['tasks'][task_number]['episodes']}
                   for model in MODELS}
        for state, seed in zip(task['formal']['initial_state_indices'], task['formal']['seeds']):
            paired = {'suite': task['suite'], 'task_id': task['task_id'],
                      'init_state_index': state, 'seed': seed, 'comparison': 'unpaired'}
            outcomes = {}
            for model in MODELS:
                row = lookups[model].get(state)
                outcome = ('success' if row['success'] else 'timeout') if row and row['status'] == 'complete' else (
                    'incomplete' if row else 'not_run')
                paired[model] = {'outcome': outcome, 'length': row.get('length') if row else None,
                                 'first_success_step': row.get('first_success_step') if row else None,
                                 'inference_requests': row['inference_requests'] if row else None}
                outcomes[model] = outcome
            if all(outcome in ('success', 'timeout') for outcome in outcomes.values()):
                a, b = (outcomes[model] == 'success' for model in MODELS)
                paired['comparison'] = ('both_success' if a and b else 'both_timeout' if not a and not b
                                         else 'openpi_only_success' if a else 'openvla_oft_only_success')
            report['paired_initial_states'].append(paired)
    return report


def make_videos(report: dict, output_dir: Path) -> list[dict]:
    """Select the first manifest-order success and timeout per task/model only."""
    selected = []
    for model, data in report['models'].items():
        for task in data['tasks']:
            for outcome in ('success', 'timeout'):
                row = next((row for row in task['episodes'] if row['status'] == 'complete'
                            and row['success'] == (outcome == 'success')), None)
                if row:
                    selected.append((model, task, outcome, row))
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        from PIL import Image
    except (ImportError, RuntimeError, OSError) as error:
        return [{'status': 'skipped', 'reason': f'Video dependencies/ffmpeg unavailable: {error}',
                 'selected_episodes': len(selected)}]
    videos = []
    for model, task, outcome, row in selected:
        name = f'{model}_{task["suite"]}_task_{task["task_id"]}_{outcome}_init_{row["init_state_index"]}.mp4'
        target = output_dir / 'videos' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        item = {'model': model, 'suite': task['suite'], 'task_id': task['task_id'],
                'outcome': outcome, 'init_state_index': row['init_state_index'],
                'record_path': row['record_path'], 'path': str(target.resolve()),
                'display': 'agentview left, wrist right; rotate180; 512x256; includes T+1 frames'}
        try:
            with EpisodeDataset(row['record_path']) as episode:
                fps = 1. / episode.metadata['control_dt']
                with imageio.get_writer(str(target), format='FFMPEG', fps=fps,
                                        codec='libx264', macro_block_size=None) as writer:
                    for index in range(len(episode) + 1):
                        observation = episode.observation(index)
                        frames = [np.asarray(Image.fromarray(observation[key][::-1, ::-1]).resize((256, 256)))
                                  for key in ('agentview_image', 'wrist_image')]
                        writer.append_data(np.concatenate(frames, axis=1))
            item.update(status='complete', frames=row['length'] + 1, fps=fps)
        except (ImportError, RuntimeError, OSError, ValueError) as error:
            target.unlink(missing_ok=True)
            item.update(status='skipped', reason=f'Video encoder unavailable/failed: {error}')
        videos.append(item)
    return videos


def markdown_report(report: dict) -> str:
    percent = lambda value: '—' if value is None else f'{value:.1%}'
    number = lambda value: '—' if value is None else f'{value:.2f}'
    lines = [f'# {report["benchmark_id"]} — formal analysis', '',
             'Observed denominators are explicit. Missing/incomplete runs are not failures.', '',
             '| Model | Task | Status | Success / completed | Planned | Wilson 95% | Mean length | Mean first success | Requests |',
             '|---|---|---|---:|---:|---|---:|---:|---:|']
    for model, data in report['models'].items():
        for task in data['tasks']:
            interval = task['wilson_95']
            ci = f'{percent(interval[0])}–{percent(interval[1])}' if interval else '—'
            lines.append(f'| {model} | {task["suite"]}/{task["task_id"]} | {task["status"]} | '
                         f'{task["successes"]}/{task["completed"]} | {task["requested"]} | {ci} | '
                         f'{number(task["mean_length"])} | {number(task["mean_first_success_step"])} | {task["inference_requests"]} |')
    for model, data in report['models'].items():
        macro = data['macro']
        lines.extend(['', f'**{model}: {data["status"]}**. Fully complete tasks '
                      f'{macro["fully_completed_tasks"]}/{macro["planned_tasks"]}; '
                      f'completed episodes {macro["completed_episodes"]}/{macro["planned_episodes"]}; '
                      f'incomplete {macro["incomplete_episodes"]}; unattempted {macro["unattempted_episodes"]}. '
                      f'Fixed-manifest task macro: {percent(macro["fixed_manifest_task_macro_success_rate"])}. '
                      f'Observed-task macro: {percent(macro["observed_task_macro_success_rate"])} '
                      f'({macro["tasks_with_completed_episodes"]}/{macro["planned_tasks"]} tasks; may include partial tasks).', ''])
    lines.extend(['', '## Actual latency samples (ms)', '',
                  '| Model | Measurement | N / expected | Mean | P50 | P95 | Max |',
                  '|---|---|---:|---:|---:|---:|---:|'])
    for model, data in report['models'].items():
        for name, values in data['latency'].items():
            lines.append(f'| {model} | {name} | {values["count"]}/{values["expected_count"]} | '
                         + ' | '.join(number(values[key]) for key in ('mean_ms', 'p50_ms', 'p95_ms', 'max_ms')) + ' |')
    failures = []
    for model, data in report['models'].items():
        for task in data['tasks']:
            patterns = {}
            for row in task['episodes']:
                if row['status'] == 'complete' and not row['success']:
                    key = tuple(tuple(atom) for atom in row['evidence']['final_unsatisfied_native_atoms'])
                    patterns[key] = patterns.get(key, 0) + 1
            for atoms, count in patterns.items():
                failures.append(f'| {model} | {task["suite"]}/{task["task_id"]} | {count} | '
                                + '; '.join(' '.join(atom) for atom in atoms) + ' |')
    if failures:
        lines.extend(['', '## Timeout evidence', '',
                      '| Model | Task | Episodes | Final unsatisfied native atoms |',
                      '|---|---|---:|---|', *failures,
                      '', 'Per-episode first observed satisfaction and later loss are in summary.json.'])
    lines.extend(['', '## Paired initial states', '',
                  '| Task | Init | Seed | OpenPI | OFT | Observed pair |',
                  '|---|---:|---:|---|---|---|'])
    for row in report['paired_initial_states']:
        lines.append(f'| {row["suite"]}/{row["task_id"]} | {row["init_state_index"]} | {row["seed"]} | '
                     f'{row["openpi"]["outcome"]} | {row["openvla_oft"]["outcome"]} | {row["comparison"]} |')
    lines.extend(['', '## Evidence limits', ''])
    lines.extend(f'- {note}' for note in report['notes'])
    if report['videos']:
        lines.extend(['', '## Selected display videos', ''])
        for video in report['videos']:
            lines.append(f'- [{Path(video["path"]).name}]({video["path"]})' if video['status'] == 'complete'
                         else f'- Skipped: {video["reason"]}')
    return '\n'.join(lines) + '\n'


def write_report(report: dict, output_dir: Path, *, videos: bool = False) -> None:
    """Create a new report directory exclusively, preserving all source files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report['videos'] = make_videos(report, output_dir) if videos else []
    with (output_dir / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with (output_dir / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write(markdown_report(report))
    with (output_dir / 'paired_initial_states.csv').open('x', encoding='utf-8', newline='') as stream:
        fields = ['suite', 'task_id', 'init_state_index', 'seed', 'openpi_outcome', 'openpi_length',
                  'openpi_first_success_step', 'openpi_inference_requests', 'openvla_oft_outcome',
                  'openvla_oft_length', 'openvla_oft_first_success_step', 'openvla_oft_inference_requests', 'comparison']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report['paired_initial_states']:
            flat = {key: row[key] for key in ('suite', 'task_id', 'init_state_index', 'seed', 'comparison')}
            for model in MODELS:
                flat.update({f'{model}_{key}': value for key, value in row[model].items()})
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'benchmark_manifest.json')
    parser.add_argument('--openpi-dir', type=Path)
    parser.add_argument('--openvla-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory; existing outputs are never overwritten')
    parser.add_argument('--videos', action='store_true',
                        help='At most first success + first timeout per task/model; streaming display only')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f'Output directory already exists: {args.output_dir}')
    try:
        report = analyze_challenge(args.manifest, args.openpi_dir, args.openvla_dir)
        write_report(report, args.output_dir, videos=args.videos)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f'Analysis saved: {args.output_dir.resolve() / "summary.json"}')
    for model, data in report['models'].items():
        macro = data['macro']
        print(f'{model}: {data["status"]}, completed={macro["completed_episodes"]}/'
              f'{macro["planned_episodes"]}, full tasks={macro["fully_completed_tasks"]}/'
              f'{macro["planned_tasks"]}')


if __name__ == '__main__':
    main()
