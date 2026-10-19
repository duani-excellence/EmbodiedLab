"""Run exactly the frozen challenge plan; never select trials from outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from embodiedlab.benchmark import summarize_task

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('openpi', 'openvla_oft'), required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'benchmark_manifest.json')
    parser.add_argument('--server-manifest', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'formal'), default='formal')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'benchmark_id': plan['benchmark_id'], 'model': args.model, 'phase': args.phase,
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tasks': [], 'status': 'running'}
    started = perf_counter()
    for task in plan['tasks']:
        directory = args.output_dir / task['suite'] / f'task_{task["task_id"]}'
        settings = task[args.phase]
        script = 'evaluate_openpi.py' if args.model == 'openpi' else 'evaluate_openvla.py'
        command = [sys.executable, str(ROOT / 'scripts' / script), '--suite', task['suite'],
                   '--task-id', str(task['task_id']), '--init-state-indices',
                   *map(str, settings['initial_state_indices']), '--seed', str(settings['seeds'][0]),
                   '--server-manifest', str(args.server_manifest.resolve()),
                   '--benchmark-manifest', str(args.manifest.resolve()), '--benchmark-phase', args.phase,
                   '--output-dir', str(directory.resolve())]
        if args.model == 'openpi':
            command.append('--native-action-bounds')
        print('Running', task['suite'], task['task_id'], args.phase, flush=True)
        returncode = subprocess.run(command, cwd=ROOT).returncode
        files = list(directory.glob('*/results.json'))
        row = {'suite': task['suite'], 'task_id': task['task_id'], 'language': task['language'],
               'requested': len(settings['initial_state_indices']), 'returncode': returncode}
        if len(files) == 1:
            result = json.loads(files[0].read_text())
            row.update(result_path=str(files[0].resolve()), **summarize_task(result['episodes']))
            row['unattempted'] = row['requested'] - row['episodes_total']
        report['tasks'].append(row)
        report['wall_seconds'] = perf_counter() - started
        report['status'] = 'incomplete' if returncode else 'running'
        (args.output_dir / 'run_index.json').write_text(json.dumps(report, indent=2) + '\n')
        if returncode:
            raise SystemExit(returncode)
    rates = [task['success_rate'] for task in report['tasks']]
    report.update(status='complete', task_macro_success_rate=sum(rates) / len(rates),
                  wall_seconds=perf_counter() - started)
    (args.output_dir / 'run_index.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
