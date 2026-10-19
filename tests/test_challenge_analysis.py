"""SYNTHETIC I/O/statistics fixtures only; no VLA, physics, or model-score evidence."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from embodiedlab.data.episode import EpisodeRecorder


SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'analyze_challenge.py'
SPEC = importlib.util.spec_from_file_location('challenge_analysis', SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


@pytest.fixture
def synthetic_plan(tmp_path):
    """Deliberately tiny invented tasks; never pretend to be benchmark evidence."""
    tasks = []
    for task_id in range(2):
        tasks.append({'suite': 'SYNTHETIC_TEST_ONLY', 'task_id': task_id,
                      'language': 'synthetic validation fixture, no physics',
                      'max_episode_steps': 3,
                      'native_goal_predicates': [['close', 'synthetic_drawer'],
                                                 ['in', 'synthetic_object', 'synthetic_drawer']],
                      'formal': {'initial_state_indices': [0, 1], 'seeds': [42, 43]}})
    plan = {'benchmark_id': 'SYNTHETIC_STATISTICS_TEST_ONLY', 'tasks': tasks,
            'environment': {'control_dt': 0.05, 'settle_steps': 10},
            'models': {'openpi': {'replan_steps': 5, 'action_horizon': 10},
                       'openvla_oft': {'replan_steps': 8, 'action_horizon': 8}}}
    path = tmp_path / 'synthetic_manifest.json'
    save(path, plan)
    return path, plan


def synthetic_record(path, task, ordinal, success, length):
    observation = {'robot_state': np.zeros(8, dtype=np.float64),
                   'joint_positions': np.zeros(7, dtype=np.float64),
                   'agentview_image': np.zeros((2, 2, 3), dtype=np.uint8),
                   'wrist_image': np.zeros((2, 2, 3), dtype=np.uint8),
                   'language': task['language']}
    metadata = {'suite': task['suite'], 'task_id': task['task_id'],
                'init_state_index': ordinal, 'max_episode_steps': 3,
                'settle_steps': 10, 'control_dt': 0.05}
    recorder = EpisodeRecorder(path, metadata)
    recorder.start(observation, {'simulation_time': 0.}, seed=42 + ordinal)
    for step in range(1, length + 1):
        final = step == length
        recorder.append(np.zeros(7), observation, float(final and success),
                        final and success, final and not success,
                        {'simulation_time': step * 0.05, 'success': final and success})
    recorder.close()


def synthetic_task_run(root, manifest, plan, task_number=0, outcomes=(True, False),
                       model='openpi', partial=False):
    task = plan['tasks'][task_number]
    run = root / task['suite'] / f'task_{task["task_id"]}' / 'synthetic_timestamp'
    protocol = {'policy': model, 'suite': task['suite'], 'task_id': task['task_id'],
                'language': task['language'], 'max_episode_steps': 3,
                'initial_state_indices': [0, 1], 'seeds': [42, 43],
                **plan['models'][model],
                'benchmark': {'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                              'phase': 'formal', 'task': deepcopy(task)}}
    save(run / 'protocol.json', protocol)
    episodes = []
    for ordinal, success in enumerate(outcomes):
        length = 2 if success else 3
        record = run / f'synthetic_{ordinal}.hdf5'
        synthetic_record(record, task, ordinal, success, length)
        steps = []
        for step in range(1, length + 1):
            # Drawer starts closed at first observation, then opens on timeout.
            close = success or step == 1
            inside = success and step == length
            steps.append({'step': step, 'native_success': close and inside,
                          'act_ms': step * 10., 'env_step_ms': step * 1.,
                          'native_goal_predicates': [
                              {'predicate': task['native_goal_predicates'][0], 'satisfied': close},
                              {'predicate': task['native_goal_predicates'][1], 'satisfied': inside}]})
        # Deliberately unequal request counts and delays test sample aggregation.
        requests = [{'request_step': step, 'rpc_ms': 10. if ordinal == 0 else 100.,
                     'synchronized_sample_ms': 5. if ordinal == 0 else 50.}
                    for step in range(1 if ordinal == 0 else 2)]
        row = {'episode_index': ordinal, 'init_state_index': ordinal, 'seed': 42 + ordinal,
               'status': 'complete', 'success': success, 'length': length,
               'first_success_step': length if success else None,
               'terminated': success, 'truncated': not success,
               'inference_requests': len(requests), 'record_path': str(record)}
        save(run / f'episode_{ordinal:03d}_diagnostics.json', {'steps': steps, 'requests': requests})
        episodes.append(row)
    if partial:
        episodes.append({'episode_index': len(episodes), 'init_state_index': len(episodes),
                         'seed': 42 + len(episodes), 'status': 'incomplete', 'success': None,
                         'completed_steps': 0, 'inference_requests': 1,
                         'error': {'type': 'SYNTHETIC_TEST_ERROR'}})
        save(run / f'episode_{len(episodes)-1:03d}_diagnostics.json',
             {'steps': [], 'requests': [{'request_step': 0, 'rpc_ms': None}]})
    complete = sum(row['status'] == 'complete' for row in episodes)
    save(run / 'results.json', {'suite': task['suite'], 'task_id': task['task_id'],
                              'requested': 2, 'attempted': len(episodes), 'completed': complete,
                              'incomplete': len(episodes) - complete,
                              'unattempted': 2 - len(episodes),
                              'success_rate_denominator': complete,
                              'status': 'complete' if complete == 2 else 'incomplete',
                              'episodes': episodes})
    return run


def test_synthetic_partial_denominators_latency_predicates_and_missing_model(tmp_path, synthetic_plan):
    manifest, plan = synthetic_plan
    root = tmp_path / 'synthetic_openpi'
    run = synthetic_task_run(root, manifest, plan)
    report = analysis.analyze_challenge(manifest, root, tmp_path / 'missing_model')
    model = report['models']['openpi']
    assert model['status'] == 'partial'
    assert model['macro']['fixed_manifest_task_macro_success_rate'] is None
    assert model['macro']['observed_task_macro_success_rate'] == .5
    assert model['macro']['fully_completed_tasks'] == 1
    assert model['macro']['completed_episodes'] == 2
    assert model['macro']['unattempted_episodes'] == 2
    task = model['tasks'][0]
    assert task['successes'] == 1 and task['success_rate_denominator'] == 2
    assert task['wilson_95'] == pytest.approx((.09453, .90547), abs=1e-5)
    assert task['mean_length'] == 2.5 and task['mean_first_success_step'] == 2.
    assert task['inference_requests'] == 3
    rpc = model['latency']['rpc']
    assert rpc['count'] == 3 and rpc['mean_ms'] == 70.
    assert rpc['p50_ms'] == 100. and rpc['p95_ms'] == 100.
    assert model['latency']['synchronized_sample']['mean_ms'] == 35.
    assert model['latency']['act']['count'] == 5
    timeout = task['episodes'][1]
    assert timeout['evidence']['classification'] == 'timeout_with_final_unsatisfied_native_atoms'
    assert timeout['evidence']['close_predicate_caveat'] is True
    assert timeout['native_predicate_progress'][0]['first_satisfied_step'] == 1
    assert timeout['native_predicate_progress'][0]['ever_lost_after_satisfied'] is True
    assert timeout['record_validation']['observations'] == 4
    missing = report['models']['openvla_oft']
    assert missing['status'] == 'not_run'
    assert missing['tasks'][0]['success_rate'] is None
    assert missing['tasks'][0]['success_rate_denominator'] == 0
    assert all(row['comparison'] == 'unpaired' for row in report['paired_initial_states'])
    assert run.exists()  # Evidence remains in place.


def test_synthetic_incomplete_errors_excluded_and_missing_timing_exposed(tmp_path, synthetic_plan):
    manifest, plan = synthetic_plan
    root = tmp_path / 'synthetic_openpi'
    synthetic_task_run(root, manifest, plan, outcomes=(True,), partial=True)
    report = analysis.analyze_challenge(manifest, root)
    task = report['models']['openpi']['tasks'][0]
    assert task['completed'] == task['incomplete'] == 1
    assert task['unattempted'] == 0
    assert task['successes'] == task['success_rate_denominator'] == 1
    assert task['success_rate'] == 1.
    assert task['latency']['rpc']['count'] == 1
    assert task['latency']['rpc']['expected_count'] == 2
    assert task['latency']['rpc']['missing_count'] == 1
    assert task['episodes'][1]['evidence']['classification'] == 'incomplete_error_excluded'
    assert task['episodes'][1]['record_validation']['validated'] is False
    assert report['models']['openpi']['macro']['fully_completed_tasks'] == 0


def test_synthetic_full_macro_is_equal_task_weight_and_pairs_are_complete(tmp_path, synthetic_plan):
    manifest, plan = synthetic_plan
    a, b = tmp_path / 'a', tmp_path / 'b'
    for root, model, outcomes in ((a, 'openpi', (True, False)),
                                  (b, 'openvla_oft', (False, True))):
        synthetic_task_run(root, manifest, plan, 0, outcomes, model)
        synthetic_task_run(root, manifest, plan, 1, (True, True), model)
    report = analysis.analyze_challenge(manifest, a, b)
    for model in analysis.MODELS:
        assert report['models'][model]['status'] == 'complete'
        macro = report['models'][model]['macro']
        assert macro['fixed_manifest_task_macro_success_rate'] == .75
        assert macro['fully_completed_tasks'] == 2
    assert [row['comparison'] for row in report['paired_initial_states']] == [
        'openpi_only_success', 'openvla_oft_only_success', 'both_success', 'both_success']


def test_synthetic_cleanup_error_preserves_completed_outcomes_but_withholds_full_macro(tmp_path, synthetic_plan):
    manifest, plan = synthetic_plan
    root = tmp_path / 'synthetic_openpi'
    run = synthetic_task_run(root, manifest, plan)
    result = json.loads((run / 'results.json').read_text())
    result.update(status='incomplete', error={'type': 'SYNTHETIC_CLEANUP_ERROR'})
    save(run / 'results.json', result)
    report = analysis.analyze_challenge(manifest, root)
    task = report['models']['openpi']['tasks'][0]
    assert task['status'] == 'partial'
    assert task['successes'] == 1 and task['completed'] == 2
    assert task['source_run_error']['type'] == 'SYNTHETIC_CLEANUP_ERROR'
    assert report['models']['openpi']['macro']['fixed_manifest_task_macro_success_rate'] is None


@pytest.mark.parametrize('damage', ['seed', 'init_state_index', 'episode_index', 'duplicate_episode',
                                    'protocol_hash', 'protocol_model', 'protocol_phase',
                                    'hdf_seed', 'hdf_init', 'diagnostic_goal', 'first_success',
                                    'request_count', 'latency_nonfinite', 'duplicate_run'])
def test_synthetic_strict_validation_rejects_mismatched_evidence(tmp_path, synthetic_plan, damage):
    manifest, plan = synthetic_plan
    root = tmp_path / 'synthetic_openpi'
    run = synthetic_task_run(root, manifest, plan)
    result_path, protocol_path = run / 'results.json', run / 'protocol.json'
    result = json.loads(result_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if damage in ('seed', 'init_state_index', 'episode_index'):
        result['episodes'][0][damage] = 99
    elif damage == 'duplicate_episode':
        result['episodes'][1] = result['episodes'][0]
    elif damage.startswith('protocol_'):
        if damage == 'protocol_hash':
            protocol['benchmark']['sha256'] = 'incorrect'
        elif damage == 'protocol_phase':
            protocol['benchmark']['phase'] = 'smoke'
        else:
            protocol['policy'] = 'SYNTHETIC_WRONG_MODEL'
    elif damage in ('hdf_seed', 'hdf_init'):
        with h5py.File(result['episodes'][0]['record_path'], 'r+') as handle:
            group = handle['data/demo_0']
            metadata = json.loads(group.attrs['metadata_json'])
            metadata['seed' if damage == 'hdf_seed' else 'init_state_index'] = 99
            group.attrs['metadata_json'] = json.dumps(metadata)
    elif damage in ('diagnostic_goal', 'latency_nonfinite'):
        path = run / 'episode_000_diagnostics.json'
        diagnostics = json.loads(path.read_text())
        if damage == 'diagnostic_goal':
            for row in diagnostics['steps']:
                row['native_goal_predicates'][0]['predicate'] = ['different_native_goal']
        else:
            diagnostics['requests'][0]['rpc_ms'] = float('nan')
        save(path, diagnostics)
    elif damage == 'first_success':
        result['episodes'][0]['first_success_step'] = 1
    elif damage == 'request_count':
        result['episodes'][0]['inference_requests'] = 99
    else:
        save(run.parent / 'second_synthetic_timestamp' / 'results.json', result)
    save(result_path, result)
    save(protocol_path, protocol)
    with pytest.raises(ValueError):
        analysis.analyze_challenge(manifest, root)


def test_synthetic_outputs_are_exclusive_and_keep_evidence_unchanged(tmp_path, synthetic_plan):
    manifest, plan = synthetic_plan
    root = tmp_path / 'synthetic_openpi'
    run = synthetic_task_run(root, manifest, plan)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in run.rglob('*') if path.is_file()}
    report = analysis.analyze_challenge(manifest, root)
    target = tmp_path / 'new_summary'
    analysis.write_report(report, target)
    assert set(path.name for path in target.iterdir()) == {'summary.json', 'summary.md', 'paired_initial_states.csv'}
    saved = json.loads((target / 'summary.json').read_text())
    assert saved['benchmark_id'] == 'SYNTHETIC_STATISTICS_TEST_ONLY'
    assert 'not_run' in (target / 'summary.md').read_text()
    assert len((target / 'paired_initial_states.csv').read_text().splitlines()) == 5
    with pytest.raises(FileExistsError):
        analysis.write_report(report, target)
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}


def test_synthetic_missing_ffmpeg_is_explicit_skip(tmp_path, synthetic_plan, monkeypatch):
    import sys

    manifest, _ = synthetic_plan
    report = analysis.analyze_challenge(manifest)
    monkeypatch.setitem(sys.modules, 'imageio_ffmpeg', None)
    videos = analysis.make_videos(report, tmp_path)
    assert videos[0]['status'] == 'skipped'
    assert 'unavailable' in videos[0]['reason']


def test_synthetic_videos_select_first_per_outcome_and_stream_t_plus_one(tmp_path, synthetic_plan, monkeypatch):
    import imageio.v2 as imageio
    import imageio_ffmpeg

    manifest, plan = synthetic_plan
    root = tmp_path / 'synthetic_openpi'
    synthetic_task_run(root, manifest, plan, 0, (True, False))
    synthetic_task_run(root, manifest, plan, 1, (True, True))
    report = analysis.analyze_challenge(manifest, root)
    visits, writes = [], []

    class StreamingSyntheticEpisode:
        metadata = {'control_dt': .05}

        def __init__(self, path):
            self.path = path

        def __len__(self):
            return 2

        def observation(self, index):
            visits.append((self.path, index))
            pixels = np.zeros((256, 256, 3), dtype=np.uint8)
            pixels[0, 0] = [1, 2, 3]
            pixels[-1, -1] = [9, 10, 11]
            return {'agentview_image': pixels, 'wrist_image': pixels}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class SyntheticWriter:
        def __init__(self, path, **kwargs):
            self.path = path
            assert kwargs['fps'] == 20.

        def append_data(self, frame):
            # Inspect a single frame, never collect the RGB sequence.
            assert frame.shape == (256, 512, 3)
            assert frame[0, 0].tolist() == [9, 10, 11]  # Display-only rotate180.
            writes.append(self.path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(analysis, 'EpisodeDataset', StreamingSyntheticEpisode)
    monkeypatch.setattr(imageio, 'get_writer', SyntheticWriter)
    monkeypatch.setattr(imageio_ffmpeg, 'get_ffmpeg_exe', lambda: 'synthetic_test_encoder')
    videos = analysis.make_videos(report, tmp_path / 'synthetic_display')
    assert len(videos) == 3  # First success/timeout task0, first success task1.
    assert [video['init_state_index'] for video in videos] == [0, 1, 0]
    assert all(video['status'] == 'complete' for video in videos)
    assert len(visits) == len(writes) == 9
    assert [index for _, index in visits] == [0, 1, 2] * 3
