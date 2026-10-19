"""Small, model-independent summaries of recorded LIBERO benchmark results."""
from math import sqrt
from typing import Sequence


def wilson_interval(successes: int, total: int) -> tuple[float, float] | None:
    """Return the Wilson 95% interval, or None when there is no denominator."""
    if (type(successes) is not int or type(total) is not int
            or not 0 <= successes <= total):
        raise ValueError('successes and total must be integers with 0 <= successes <= total')
    if total == 0:
        return None
    z = 1.95996
    proportion = successes / total
    denominator = 1. + z * z / total
    center = (proportion + z * z / (2. * total)) / denominator
    half_width = z * sqrt(proportion * (1. - proportion) / total
                         + z * z / (4. * total * total)) / denominator
    return max(0., center - half_width), min(1., center + half_width)


def summarize_predicates(step_rows: Sequence[dict]) -> list[dict]:
    """Summarize observed native predicates without imposing a subgoal order.

    Each row has ``step`` and ``native_goal_predicates``. First satisfaction
    means the first recorded true value; sparse logs cannot establish an
    earlier, unobserved transition. Diagnostics never become policy inputs.
    """
    summaries = {}
    expected_predicates = None
    previous_step = -1
    for row in step_rows:
        step = row.get('step')
        if type(step) is not int or step <= previous_step:
            raise ValueError('Predicate log steps must be nonnegative, strictly increasing integers')
        previous_step = step
        predicates = row.get('native_goal_predicates')
        if not isinstance(predicates, list):
            raise ValueError('Each step must contain a native_goal_predicates list')
        observed = {}
        for item in predicates:
            predicate, satisfied = item.get('predicate'), item.get('satisfied')
            if (not isinstance(predicate, list) or not predicate
                    or any(not isinstance(part, str) or not part for part in predicate)
                    or type(satisfied) is not bool):
                raise ValueError('Each predicate needs a nonempty string list and a boolean satisfied value')
            key = tuple(predicate)
            if key in observed:
                raise ValueError('A predicate may occur only once per step')
            observed[key] = satisfied
        if expected_predicates is None:
            expected_predicates = set(observed)
        elif set(observed) != expected_predicates:
            raise ValueError('Native goal predicates must remain the same across recorded steps')
        for key, satisfied in observed.items():
            summary = summaries.setdefault(key, {
                'predicate': list(key), 'first_satisfied_step': None,
                'final_satisfied': False, 'ever_lost_after_satisfied': False,
            })
            if satisfied and summary['first_satisfied_step'] is None:
                summary['first_satisfied_step'] = step
            if not satisfied and summary['first_satisfied_step'] is not None:
                summary['ever_lost_after_satisfied'] = True
            summary['final_satisfied'] = satisfied
    return list(summaries.values())


def summarize_task(episodes: Sequence[dict]) -> dict:
    """Count explicit complete/incomplete rows; errors are never task failures.

    The caller separately compares these rows with the fixed manifest to count
    unattempted episodes. Success rate uses completed episodes only, with the
    denominator and incomplete count exposed alongside it. Task macro averages
    must use these per-task rates, not pool successes across different tasks.
    """
    completed = []
    for episode in episodes:
        if episode.get('status') not in ('complete', 'incomplete'):
            raise ValueError('Each episode needs explicit status complete or incomplete')
        if episode['status'] == 'incomplete':
            continue
        success, length = episode.get('success'), episode.get('length')
        if type(success) is not bool or type(length) is not int or length < 0:
            raise ValueError('Complete episodes need boolean success and nonnegative integer length')
        if type(episode.get('truncated', False)) is not bool:
            raise ValueError('Episode truncated must be boolean')
        if success and episode.get('truncated', False):
            raise ValueError('A successful episode cannot also be a timeout')
        first_success = episode.get('first_success_step', length if success else None)
        if ((success and (type(first_success) is not int or not 0 <= first_success <= length))
                or (not success and first_success is not None)):
            raise ValueError('first_success_step must match success and lie within the episode')
        completed.append((episode, first_success))
    total = len(completed)
    successes = sum(episode['success'] for episode, _ in completed)
    lengths = [episode['length'] for episode, _ in completed]
    first_steps = [step for _, step in completed]
    success_steps = [step for step in first_steps if step is not None]
    return {
        'episodes_total': len(episodes), 'completed': total,
        'incomplete': len(episodes) - total,
        'timeouts': sum(episode.get('truncated', False) and not episode['success']
                        for episode, _ in completed),
        'successes': successes, 'success_rate_denominator': total,
        'success_rate': successes / total if total else None,
        'wilson_95': wilson_interval(successes, total),
        'lengths': lengths, 'mean_length': sum(lengths) / total if total else None,
        'first_success_steps': first_steps,
        'mean_first_success_step': (sum(success_steps) / len(success_steps)
                                    if success_steps else None),
    }
