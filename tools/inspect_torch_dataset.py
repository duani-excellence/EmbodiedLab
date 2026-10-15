"""Validate and measure one streaming train/validation pass over official LIBERO data."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from embodiedlab.data.episode import EpisodeDataset
from embodiedlab.data.torch_dataset import LiberoTorchDataset, preprocess_observation, split_episodes

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_KEYS = ('robot_state', 'agentview_image', 'wrist_image', 'action')


def file_sha256(path: Path) -> str:
    """Hash the actual source bytes without loading the entire HDF5 into memory."""
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def memory_snapshot() -> dict:
    """Linux current RSS and cumulative process peak, both in MiB."""
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            rss_mib = int(line.split()[1]) / 1024
            break
    else:
        raise RuntimeError('Linux VmRSS is unavailable; cannot report process memory')
    return {'rss_mib': rss_mib,
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def batch_summary(batch: dict) -> dict:
    """Check every actual CPU batch and summarize all four tensors."""
    if set(batch) != set(SAMPLE_KEYS):
        raise ValueError(f'Batch keys must be exactly {SAMPLE_KEYS}')
    summary = {}
    count = batch['action'].shape[0]
    for key in SAMPLE_KEYS:
        value = batch[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != 'cpu':
            raise ValueError(f'{key} must be a CPU float32 tensor')
        if value.shape[0] != count or count < 1:
            raise ValueError(f'{key} batch size differs or is empty')
        if key == 'robot_state' and value.shape != (count, 8):
            raise ValueError('robot_state must have shape (B, 8)')
        if key == 'action' and value.shape != (count, 7):
            raise ValueError('action must have shape (B, 7)')
        if key.endswith('_image') and (value.ndim != 4 or value.shape[1] != 3 or min(value.shape[2:]) < 1):
            raise ValueError(f'{key} must have nonempty shape (B, 3, H, W)')
        # NumPy views avoid expensive CPU thread-pool reductions in the loader timer.
        array = value.numpy()
        finite = bool(np.isfinite(array).all())
        minimum, maximum = float(array.min()), float(array.max())
        if not finite:
            raise ValueError(f'{key} contains non-finite values')
        if key.endswith('_image') and (minimum < 0 or maximum > 1):
            raise ValueError(f'{key} pixels must be normalized to [0, 1]')
        if key == 'action' and (minimum < -1.000001 or maximum > 1.000001):
            raise ValueError('action exceeds the native normalized command range')
        summary[key] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'minimum': minimum, 'maximum': maximum, 'finite': finite}
    return summary


def alignment_checks(dataset: LiberoTorchDataset, path: Path) -> list[dict]:
    """Compare actual first/last samples with source obs[t-1] and action[t]."""
    checks = []
    for sample_index in (0, len(dataset) - 1):
        name, action_index = dataset.locate(sample_index)
        observation_index = action_index - 1
        if name not in dataset.episodes or action_index < 1:
            raise AssertionError('locate() returned an invalid official observation/action pair')
        with EpisodeDataset.load(path, name) as source:
            if source.metadata['observation_alignment'] != 'post_action':
                raise AssertionError('This inspector requires verified official post-action observations')
            if sample_index == 0 and action_index != 1:
                raise AssertionError('First sample must omit the unavailable action[0] pair')
            if sample_index == len(dataset) - 1 and action_index != len(source) - 1:
                raise AssertionError('Last sample must contain the final available source action')
            expected = preprocess_observation(source.observation(observation_index))
            expected['action'] = torch.as_tensor(np.asarray(source.actions[action_index]), dtype=torch.float32)
            actual = dataset[sample_index]
            if set(actual) != set(SAMPLE_KEYS):
                raise AssertionError('Endpoint sample keys differ from the four-key contract')
            for key in SAMPLE_KEYS:
                if not torch.equal(actual[key], expected[key]):
                    raise AssertionError(f'{name} sample {sample_index}: source alignment differs for {key}')
        checks.append({'sample_index': sample_index, 'episode': name,
                       'observation_index': observation_index, 'action_index': action_index,
                       'all_four_keys_match': True})
    return checks


def inspect_epoch(loader: DataLoader, label: str) -> dict:
    """Consume each batch once, validate it and discard it; no model or batch cache."""
    started = time.perf_counter()
    samples = batches = 0
    first_summary = first_memory = None
    for batch in loader:
        if batches == 0:
            first_memory = memory_snapshot()
        summary = batch_summary(batch)
        if batches == 0:
            first_summary = summary
            print(f'{label} first batch: {json.dumps(summary)}', flush=True)
        samples += batch['action'].shape[0]
        batches += 1
        del batch
    elapsed = time.perf_counter() - started
    if samples != len(loader.dataset) or batches != (samples + loader.batch_size - 1) // loader.batch_size:
        raise AssertionError(f'{label} loader omitted or duplicated expected samples')
    return {'samples': samples, 'batches': batches, 'elapsed_seconds': elapsed,
            'samples_per_second': samples / elapsed, 'all_batches_validated': True,
            'first_batch': first_summary, 'memory_first_batch': first_memory,
            'memory_after_epoch': memory_snapshot()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path,
                        default=ROOT / 'data/libero/libero_goal/turn_on_the_stove_demo.hdf5')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.seed < 0:
        parser.error('batch-size must be positive and seed must be nonnegative')
    if args.output is not None and args.output.exists():
        parser.error(f'Output already exists and will not be overwritten: {args.output}')

    baseline = memory_snapshot()  # Includes imported torch; no dataset or batches yet.
    hash_before = file_sha256(args.dataset)
    source_metadata = json.loads(args.dataset.with_suffix('.source.json').read_text())
    if hash_before != source_metadata['sha256']:
        raise AssertionError('Source HDF5 hash differs from its pinned provenance')
    episodes = EpisodeDataset.list_episodes(args.dataset)
    train_names, validation_names = split_episodes(episodes, seed=args.seed, train_fraction=.8)
    if (set(train_names) & set(validation_names)
            or set(train_names) | set(validation_names) != set(episodes)):
        raise AssertionError('Episode split leaked or omitted an episode')
    if len(episodes) == 50 and (len(train_names), len(validation_names)) != (40, 10):
        raise AssertionError('The verified 50-episode source must split into 40/10 episodes')
    with LiberoTorchDataset(args.dataset, episodes=train_names) as train, \
            LiberoTorchDataset(args.dataset, episodes=validation_names) as validation:
        memory_after_construction = memory_snapshot()
        if set(train.episodes) & set(validation.episodes):
            raise AssertionError('The actual datasets contain overlapping episodes')
        if set(train.episodes) != set(train_names) or set(validation.episodes) != set(validation_names):
            raise AssertionError('Dataset episodes differ from the requested partition')
        for dataset in (train, validation):
            if sum(dataset.sample_counts) != len(dataset) or tuple(t - 1 for t in dataset.episode_lengths) != dataset.sample_counts:
                raise AssertionError('Dataset sample counts do not match official T-1 alignment')
        endpoint_checks = {'train': alignment_checks(train, args.dataset),
                           'validation': alignment_checks(validation, args.dataset)}
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed), num_workers=0)
        validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
        report = {
            'dataset': str(args.dataset.resolve()), 'batch_size': args.batch_size,
            'num_workers': 0, 'seed': args.seed, 'torch_version': torch.__version__,
            'torch_num_threads': torch.get_num_threads(),
            'timing_scope': 'one DataLoader pass including batch validation; no model',
            'memory_scope': 'CPU process RSS; peak is cumulative; kernel file cache is excluded',
            'data_loading': {'rgb': 'lazy two-camera reads per sample',
                             'numeric_arrays': 'all selected episodes validated once at construction',
                             'episode_reader_count': len(train.episodes) + len(validation.episodes),
                             'memory_after_construction': memory_after_construction},
            'memory_torch_baseline': baseline, 'episode_leak': False,
            'train': {'episodes': list(train.episodes), 'episode_count': len(train.episodes),
                      'sample_count': len(train), **inspect_epoch(train_loader, 'train')},
            'validation': {'episodes': list(validation.episodes), 'episode_count': len(validation.episodes),
                           'sample_count': len(validation), **inspect_epoch(validation_loader, 'validation')},
            'endpoint_alignment_checks': endpoint_checks,
        }
    hash_after = file_sha256(args.dataset)
    if hash_before != hash_after:
        raise AssertionError('Source HDF5 bytes changed during inspection')
    report['source_sha256_before'] = hash_before
    report['source_sha256_after'] = hash_after
    report['source_unchanged'] = True
    for label in ('train', 'validation'):
        result = report[label]
        print(f"{label}: episodes={result['episode_count']}, samples={result['samples']}, "
              f"batches={result['batches']}, elapsed={result['elapsed_seconds']:.3f}s, "
              f"throughput={result['samples_per_second']:.1f} samples/s", flush=True)
    print(f'No episode leakage; source SHA256 unchanged: {hash_after}', flush=True)
    print(f'Memory snapshots (MiB): baseline={baseline}, '
          f"train first={report['train']['memory_first_batch']}, "
          f"after train={report['train']['memory_after_epoch']}, "
          f"after validation={report['validation']['memory_after_epoch']}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
        print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
