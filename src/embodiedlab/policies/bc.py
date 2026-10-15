"""Inference-only bridge from raw LIBERO observations to the saved visual BC."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from embodiedlab.data.torch_dataset import preprocess_observation
from embodiedlab.interfaces import Action, Environment, Observation
from embodiedlab.policies.checkpoint import read_contract
from embodiedlab.policies.visual_bc import VisualBC, decode_actions


class BCPolicy:
    """Restore visual_bc_v1, including normalization buffers; never alter actions.

    Six outputs are normalized world-frame OSC pose deltas. Gripper classes
    decode to -1=open and +1=close. No filtering or controller logic is applied.
    """

    def __init__(self, checkpoint: str | Path, *, device: str = 'auto') -> None:
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('device must be auto, cpu or cuda')
        selected = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if selected == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; use cpu or check GPU access')
        self.device = torch.device(selected)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.contract = read_contract(saved)
        self.suite = self.contract['task']['suite']
        self.task_id = self.contract['task']['task_id']
        self.task_name = self.contract['task']['task_name']
        self.image_size = self.contract['training']['image_shape'][1]
        if not isinstance(saved.get('model_state_dict'), dict):
            raise ValueError('Checkpoint model_state_dict must be a dictionary')
        if any(key not in saved for key in ('epoch', 'metrics')):
            raise ValueError('Checkpoint epoch and metrics are required')
        self.model = VisualBC()
        self.model.load_state_dict(saved['model_state_dict'], strict=True)
        if (not all(torch.isfinite(v).all() for v in self.model.state_dict().values())
                or not (self.model.state_std > 0).all()):
            raise ValueError('Checkpoint weights and normalization must be finite; std must be positive')
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.checkpoint_metadata = {k: saved[k] for k in ('epoch', 'config', 'metrics')}
        self.reset()

    def assert_compatible(self, env: Environment) -> None:
        """Reject a different task, camera format or OSC convention before rollout."""
        metadata = env.episode_metadata()
        if not isinstance(metadata, dict):
            raise ValueError('BC environment episode_metadata() must return a dictionary')
        training = self.contract['training']
        expected = {**self.contract['task'], 'image_size': self.image_size,
                    'image_orientation': training['image_orientation'],
                    'control_mode': training['action_semantics'],
                    'action_dim': training['action_dim'], 'control_dt': training['control_dt'],
                    'position_delta_scale_m': training['position_delta_scale_m'],
                    'axis_angle_delta_scale_rad': training['axis_angle_delta_scale_rad'],
                    'action_low': [-1.] * 7, 'action_high': [1.] * 7,
                    'gripper': '-1 open, +1 close'}
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if type(metadata.get('task_id')) is not int or type(metadata.get('image_size')) is not int:
            mismatches.append('task_id/image_size types')
        if mismatches:
            raise ValueError('BC checkpoint is incompatible with environment: ' + ', '.join(mismatches))

    def reset(self) -> None:
        """Clear latency measurements; the feed-forward policy has no hidden state."""
        self.inference_times_ms: list[float] = []

    def act(self, observation: Observation) -> Action:
        """Preprocess raw RGB/state and return finite float64 native action (7,)."""
        started = perf_counter()
        inputs = {k: v.unsqueeze(0).to(self.device)
                  for k, v in preprocess_observation(observation).items()}
        with torch.inference_mode():
            output = self.model(**inputs)
            # CPU transfer waits for CUDA completion, so latency includes GPU work,
            # preprocessing and transfers, not just asynchronous kernel dispatch.
            action = decode_actions(output)[0].cpu().numpy().astype(np.float64)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('BC produced an invalid normalized LIBERO action')
        self.inference_times_ms.append((perf_counter() - started) * 1000.)
        return action
