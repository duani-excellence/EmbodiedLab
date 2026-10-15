"""Inspect one lazy HDF5 episode without loading an entire RGB trajectory."""
import argparse
import json
from pathlib import Path

from embodiedlab.data.episode import EpisodeDataset, IMAGE_KEYS


def main() -> None:
    """Print alignment/state/action shapes and optionally save one camera preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episode', default='demo_0')
    parser.add_argument('--suite')
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--preview-dir', type=Path)
    parser.add_argument('--rotate-preview', action='store_true', help='rotate preview 180 degrees; stored pixels are unchanged')
    args = parser.parse_args()
    try:
        with EpisodeDataset.load(args.dataset, args.episode, suite=args.suite, task_id=args.task_id) as episode:
            observation, action, reward = episode.step(args.index)
            report = {'path': str(args.dataset), 'episode': args.episode,
                      'episodes_in_file': len(EpisodeDataset.list_episodes(args.dataset)),
                      'length': len(episode), 'metadata': episode.metadata,
                      'sample_index': args.index, 'action': action.tolist(), 'reward': reward,
                      'observation_shapes': {key: list(value.shape) if hasattr(value, 'shape') else 'str'
                                             for key, value in observation.items()}}
            if args.preview_dir is not None:
                from PIL import Image
                args.preview_dir.mkdir(parents=True, exist_ok=True)
                for key in IMAGE_KEYS:
                    pixels = observation[key]
                    if args.rotate_preview:
                        pixels = pixels[::-1, ::-1]
                    target = args.preview_dir / f'{args.episode}_{args.index:06d}_{key}.png'
                    if target.exists():
                        raise FileExistsError(f'Preview already exists: {target}')
                    Image.fromarray(pixels).save(target)
            print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, IndexError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
