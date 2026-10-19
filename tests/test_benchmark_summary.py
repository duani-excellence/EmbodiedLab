"""Analysis tests use explicit logged fixtures, never synthetic model scores."""
from copy import deepcopy

import pytest

from embodiedlab.benchmark import summarize_predicates, summarize_task, wilson_interval


def test_wilson_small_sample_and_boundary_intervals():
    assert wilson_interval(5, 10) == pytest.approx((.23659, .76341), abs=1e-5)
    assert wilson_interval(0, 10) == pytest.approx((0., .27753), abs=1e-5)
    assert wilson_interval(10, 10) == pytest.approx((.72247, 1.), abs=1e-5)
    assert wilson_interval(0, 0) is None


@pytest.mark.parametrize('successes,total', [(-1, 10), (11, 10), (0, -1),
                                             (True, 10), (1, 10.0)])
def test_wilson_rejects_invalid_counts(successes, total):
    with pytest.raises(ValueError, match='integers'):
        wilson_interval(successes, total)


def predicate_row(step, stove, pot):
    return {'step': step, 'native_goal_predicates': [
        {'predicate': ['turnon', 'stove'], 'satisfied': stove},
        {'predicate': ['on', 'pot', 'stove_region'], 'satisfied': pot},
    ]}


def test_predicate_summary_tracks_loss_recovery_and_independent_subgoals():
    rows = [predicate_row(1, False, True), predicate_row(2, True, False),
            predicate_row(3, True, True), predicate_row(4, False, True)]
    rows[2]['native_goal_predicates'].reverse()  # Identity, not list order, aligns predicates.
    original = deepcopy(rows)
    summaries = summarize_predicates(rows)
    assert summaries == [
        {'predicate': ['turnon', 'stove'], 'first_satisfied_step': 2,
         'final_satisfied': False, 'ever_lost_after_satisfied': True},
        {'predicate': ['on', 'pot', 'stove_region'], 'first_satisfied_step': 1,
         'final_satisfied': True, 'ever_lost_after_satisfied': True},
    ]
    assert rows == original


def test_predicate_summary_handles_never_satisfied_sparse_and_empty_logs():
    rows = [predicate_row(0, False, False), predicate_row(100, True, False)]
    summaries = summarize_predicates(rows)
    assert summaries[0]['first_satisfied_step'] == 100
    assert summaries[1]['first_satisfied_step'] is None
    assert summaries[1]['ever_lost_after_satisfied'] is False
    assert summarize_predicates([]) == []
    assert summarize_predicates([{'step': 0, 'native_goal_predicates': []}]) == []


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'wrong_bool', 'changed_goal',
                                    'unordered_steps', 'negative_step', 'late_goals'])
def test_predicate_summary_rejects_ambiguous_or_inconsistent_logs(damage):
    rows = [predicate_row(1, False, False), predicate_row(2, True, False)]
    if damage == 'missing':
        del rows[0]['native_goal_predicates']
    elif damage == 'duplicate':
        rows[0]['native_goal_predicates'].append(rows[0]['native_goal_predicates'][0])
    elif damage == 'wrong_bool':
        rows[0]['native_goal_predicates'][0]['satisfied'] = 1
    elif damage == 'changed_goal':
        rows[1]['native_goal_predicates'].pop()
    elif damage == 'unordered_steps':
        rows[1]['step'] = 1
    elif damage == 'negative_step':
        rows[0]['step'] = -1
    else:
        rows[0]['native_goal_predicates'] = []
    with pytest.raises(ValueError):
        summarize_predicates(rows)


def test_task_summary_keeps_incomplete_out_of_success_denominator():
    rows = [
        {'status': 'complete', 'success': True, 'length': 55,
         'first_success_step': 55, 'truncated': False},
        {'status': 'complete', 'success': False, 'length': 220,
         'first_success_step': None, 'truncated': True},
        {'status': 'incomplete', 'success': None, 'completed_steps': 7,
         'error': {'type': 'TimeoutError'}},
    ]
    original = deepcopy(rows)
    result = summarize_task(rows)
    assert result['episodes_total'] == 3
    assert result['completed'] == result['success_rate_denominator'] == 2
    assert result['incomplete'] == result['timeouts'] == result['successes'] == 1
    assert result['success_rate'] == .5
    assert result['wilson_95'] == wilson_interval(1, 2)
    assert result['lengths'] == [55, 220]
    assert result['mean_length'] == 137.5
    assert result['first_success_steps'] == [55, None]
    assert result['mean_first_success_step'] == 55
    assert rows == original


def test_empty_or_all_incomplete_task_has_no_fabricated_success_rate():
    for rows in ([], [{'status': 'incomplete', 'success': None}]):
        result = summarize_task(rows)
        assert result['success_rate_denominator'] == 0
        assert result['success_rate'] is result['wilson_95'] is None
        assert result['mean_length'] is result['mean_first_success_step'] is None


@pytest.mark.parametrize('damage', ['status', 'success', 'length', 'flags', 'first_success'])
def test_task_summary_rejects_inconsistent_completed_results(damage):
    row = {'status': 'complete', 'success': True, 'length': 5, 'truncated': False,
           'first_success_step': 5}
    if damage == 'status':
        del row['status']
    elif damage == 'success':
        row['success'] = None
    elif damage == 'length':
        row['length'] = -1
    elif damage == 'flags':
        row['truncated'] = True
    else:
        row['first_success_step'] = 6
    with pytest.raises(ValueError):
        summarize_task([row])
