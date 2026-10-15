"""Training-specific regression cases exposed by the second real task."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from embodiedlab.data.torch_dataset import split_episodes
from embodiedlab.policies.visual_bc import VisualBC

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('train_bc_entry', ROOT / 'scripts/train_bc.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_one_class_expert_data_has_finite_statistics_and_constant_loss() -> None:
    # Small numeric fixture: a legitimate task may never close the gripper.
    actions = torch.zeros((3, 7))
    actions[:, 6] = -1.
    batch = {'robot_state': torch.arange(24).reshape(3, 8).float(), 'action': actions}
    stats = training.training_statistics([batch])
    assert stats['gripper_counts'] == [3, 0]
    assert stats['gripper_probabilities'] == pytest.approx([.8, .2])
    result = training.run_epoch(VisualBC(), [batch], torch.device('cpu'), constant=stats)
    assert torch.isfinite(torch.tensor(result['loss']))
    assert result['gripper_accuracy'] == 1. and result['samples'] == 3


def test_non_fifty_episode_split_and_output_directory_protection(tmp_path) -> None:
    train, validation = split_episodes([f'demo_{i}' for i in range(9)], seed=19)
    assert (len(train), len(validation)) == (7, 2)
    assert set(train).isdisjoint(validation)
    sentinel = tmp_path / 'best.pt'
    sentinel.write_bytes(b'existing research result')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_bc.py'),
                             '--device', 'cpu', '--epochs', '1', '--output-dir', str(tmp_path)],
                            cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and 'FileExistsError' in result.stderr
    assert sentinel.read_bytes() == b'existing research result'
    assert set(tmp_path.iterdir()) == {sentinel}
