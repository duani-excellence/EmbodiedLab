"""Run zero/random LIBERO interface checks and optionally record episodes."""
import argparse
import json
from pathlib import Path

from embodiedlab.evaluation import evaluate_suites


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', '--suites', dest='suites', nargs='+', default=['libero_spatial'],
                        help='Official suites: libero_spatial, libero_object, libero_goal, libero_10, libero_90')
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument('--task-ids', nargs='+', type=int, default=[0])
    tasks.add_argument('--all-tasks', action='store_true', help='Explicitly evaluate the complete suite')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--list-tasks', action='store_true', help='List native task identities without running episodes')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy', choices=['zero', 'random'], default='zero')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--max-episode-steps', type=int, default=None,
                        help='Optional smoke-test limit; omitted uses the suite protocol horizon')
    parser.add_argument('--settle-steps', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, default=Path('evaluation_results'))
    parser.add_argument('--record-dir', type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.list_tasks:
        from embodiedlab.envs.libero_env import list_tasks
        print(json.dumps([task for suite in arguments.suites for task in list_tasks(suite)],
                         ensure_ascii=False, indent=2))
        return
    report, output_dir = evaluate_suites(
        suites=arguments.suites, task_ids=None if arguments.all_tasks else arguments.task_ids,
        episodes=arguments.episodes, seed=arguments.seed, policy_name=arguments.policy,
        image_size=arguments.image_size, max_episode_steps=arguments.max_episode_steps,
        settle_steps=arguments.settle_steps, output_dir=arguments.output_dir,
        record_dir=arguments.record_dir,
    )
    for task in report['tasks']:
        print(f"{task['suite']} task={task['task_id']} {task['task_name']}: "
              f"success={task['success_rate']:.1%}, reward={task['mean_reward']:.4f}, "
              f"length={task['mean_length']:.1f}")
    print(f"Episodes={report['episode_count']}, success={report['success_rate']:.1%}")
    print(f'Reports: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
