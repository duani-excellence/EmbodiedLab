"""Small offline visual BC network; no simulator or online Policy wrapper."""
import torch
from torch import nn
from torch.nn import functional as F


class VisualBC(nn.Module):
    """Two RGB cameras plus 8 state values -> 6 OSC deltas and 2 gripper logits.

    Images are already RGB/CHW/float32 in [0,1] from preprocess_observation().
    State normalization uses training-only statistics saved as model buffers.
    Gripper classes are 0=open (-1), 1=close (+1); this source has no hold labels.
    """

    def __init__(self, state_mean: torch.Tensor | None = None,
                 state_std: torch.Tensor | None = None) -> None:
        super().__init__()
        mean = (torch.zeros(8) if state_mean is None
                else torch.as_tensor(state_mean, dtype=torch.float32).detach().cpu().clone())
        std = (torch.ones(8) if state_std is None
               else torch.as_tensor(state_std, dtype=torch.float32).detach().cpu().clone())
        if (mean.shape != (8,) or std.shape != (8,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or not (std > 0).all()):
            raise ValueError('State mean/std must be finite shape (8,), with positive std')
        self.register_buffer('state_mean', mean)
        self.register_buffer('state_std', std)
        # One encoder instance is reused for both cameras. The 4x4 grid retains
        # coarse spatial layout instead of averaging the entire scene to a point.
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            # Fixed 128px inputs give 16x16 features. Non-overlapping 4x4
            # averages are equivalent here and support strict CUDA determinism.
            nn.AvgPool2d(4), nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(128 + 128 + 32, 128), nn.ReLU())
        self.motion_head = nn.Sequential(nn.Linear(128, 6), nn.Tanh())
        self.gripper_head = nn.Linear(128, 2)

    def forward(self, agentview_image: torch.Tensor, wrist_image: torch.Tensor,
                robot_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return motion (B,6) in [-1,1] and unnormalized gripper logits (B,2)."""
        if (agentview_image.ndim != 4 or agentview_image.shape[1:] != (3, 128, 128)
                or wrist_image.shape != agentview_image.shape or agentview_image.shape[0] == 0
                or robot_state.shape != (agentview_image.shape[0], 8)):
            raise ValueError('Expected two (B,3,128,128) cameras and (B,8) state, B>0')
        agent_features = self.image_encoder(agentview_image)
        wrist_features = self.image_encoder(wrist_image)
        state_features = self.state_encoder((robot_state - self.state_mean) / self.state_std)
        features = self.fusion(torch.cat([agent_features, wrist_features, state_features], dim=1))
        return {'motion': self.motion_head(features), 'gripper_logits': self.gripper_head(features)}


def bc_loss(output: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mean MSE over six normalized OSC deltas + mean binary gripper CE."""
    if action.shape != (output['motion'].shape[0], 7):
        raise ValueError('Expected action shape (B,7)')
    gripper = action[:, 6]
    if not torch.all((gripper == -1) | (gripper == 1)):
        raise ValueError('VisualBC supports only -1/+1 gripper labels; 0 hold is unsupported')
    motion_mse = F.mse_loss(output['motion'], action[:, :6])
    gripper_ce = F.cross_entropy(output['gripper_logits'], (gripper > 0).long())
    return {'loss': motion_mse + gripper_ce, 'motion_mse': motion_mse, 'gripper_ce': gripper_ce}


def decode_actions(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Decode (B,7) native normalized OSC deltas with gripper exactly -1 or +1."""
    gripper = (2 * output['gripper_logits'].argmax(dim=1) - 1).to(output['motion'].dtype)
    return torch.cat([output['motion'], gripper[:, None]], dim=1)
