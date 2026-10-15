"""Numerically replay an EmbodiedLab HDF5 recording, not an official expert file."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from embodiedlab.data.episode import EpisodeDataset
from embodiedlab.data.replay import replay_episode
from embodiedlab.envs.libero_env import LiberoEnv


def main() -> None:
    """Load recorded configuration, run actions, and report measured deviations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--atol',type=float,default=1e-6)
    parser.add_argument('--pixel-atol',type=int,default=0)
    args = parser.parse_args()
    with EpisodeDataset.load(args.episode) as episode:
        with LiberoEnv(**episode.environment_config) as env:
            result = replay_episode(env,episode,atol=args.atol,pixel_atol=args.pixel_atol)
    print(json.dumps(asdict(result),indent=2,allow_nan=False))
    if not result.matches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
