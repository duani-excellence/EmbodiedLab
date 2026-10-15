"""Visual BC shapes, native action semantics, gradients and checkpoint integrity."""
import math
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from embodiedlab.data.torch_dataset import LiberoTorchDataset
from embodiedlab.policies.visual_bc import VisualBC, bc_loss, decode_actions


OFFICIAL = Path(__file__).resolve().parents[1] / 'data/libero/libero_goal/turn_on_the_stove_demo.hdf5'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


@pytest.fixture(scope='module')
def real_batch() -> dict[str, torch.Tensor]:
    """Four genuine aligned expert pairs; close HDF5 before any model work."""
    with LiberoTorchDataset(OFFICIAL, episodes=('demo_0',)) as dataset:
        return next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))


def predict(model: VisualBC, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(batch['agentview_image'], batch['wrist_image'], batch['robot_state'])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_batch_forward_backward_and_inputs_unchanged(real_batch, device: str) -> None:
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable; real forward/backward is verified on CPU')
    torch.manual_seed(42)
    batch = {key: value.to(device).clone() for key, value in real_batch.items()}
    before = {key: value.clone() for key, value in batch.items()}
    model = VisualBC().to(device)
    output = predict(model, batch)
    assert set(output) == {'motion', 'gripper_logits'}
    assert output['motion'].shape == (4, 6) and output['gripper_logits'].shape == (4, 2)
    assert torch.all(output['motion'] >= -1.) and torch.all(output['motion'] <= 1.)
    for value in output.values():
        assert value.dtype == torch.float32 and value.device.type == device
        assert torch.isfinite(value).all()
        value.retain_grad()
    terms = bc_loss(output, batch['action'])
    assert set(terms) == {'loss', 'motion_mse', 'gripper_ce'}
    assert all(value.ndim == 0 and torch.isfinite(value) for value in terms.values())
    torch.testing.assert_close(terms['loss'], terms['motion_mse'] + terms['gripper_ce'])
    terms['loss'].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    for value in output.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad).item() > 0
    for key in batch:
        torch.testing.assert_close(batch[key], before[key], rtol=0, atol=0)


def test_normalization_statistics_are_owned_registered_buffers() -> None:
    defaults = VisualBC()
    buffers = dict(defaults.named_buffers())
    assert {'state_mean', 'state_std'} <= buffers.keys()
    torch.testing.assert_close(defaults.state_mean, torch.zeros(8))
    torch.testing.assert_close(defaults.state_std, torch.ones(8))
    mean, std = torch.arange(8, dtype=torch.float32) / 10., torch.arange(1, 9, dtype=torch.float32) / 4.
    expected_mean, expected_std = mean.clone(), std.clone()
    model = VisualBC(mean, std)
    mean.add_(10.)
    std.mul_(2.)
    torch.testing.assert_close(model.state_dict()['state_mean'], expected_mean)
    torch.testing.assert_close(model.state_dict()['state_std'], expected_std)
    assert not model.state_mean.requires_grad and not model.state_std.requires_grad


def test_nondefault_normalization_matches_explicit_input_transform(real_batch) -> None:
    mean = torch.linspace(-.7, .7, 8)
    std = torch.linspace(.5, 2., 8)
    torch.manual_seed(17)
    identity_model = VisualBC().eval()
    torch.manual_seed(17)
    normalized_model = VisualBC(mean, std).eval()
    transformed = {**real_batch, 'robot_state': (real_batch['robot_state'] - mean) / std}
    with torch.no_grad():
        expected = predict(identity_model, transformed)
        actual = predict(normalized_model, real_batch)
        unnormalized = predict(identity_model, real_batch)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-7)
    assert any(not torch.allclose(actual[key], unnormalized[key]) for key in actual)


@pytest.mark.parametrize('training', [True, False])
def test_repeated_predictions_are_deterministic_in_train_and_eval(real_batch, training: bool) -> None:
    torch.manual_seed(21)
    model = VisualBC().train(training)
    with torch.no_grad():
        first = predict(model, real_batch)
        second = predict(model, real_batch)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_weights_only_checkpoint_restores_stats_and_exact_predictions(real_batch, tmp_path) -> None:
    torch.manual_seed(31)
    model = VisualBC(torch.linspace(-.5, .5, 8), torch.linspace(.2, 1.2, 8)).eval()
    with torch.no_grad():
        before = predict(model, real_batch)
    path = tmp_path / 'visual_bc.pt'
    torch.save(model.state_dict(), path)
    restored = VisualBC().eval()
    restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    with torch.no_grad():
        after = predict(restored, real_batch)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    torch.testing.assert_close(restored.state_mean, model.state_mean, rtol=0, atol=0)
    torch.testing.assert_close(restored.state_std, model.state_std, rtol=0, atol=0)


def test_gripper_classes_loss_and_decode_preserve_native_action_semantics() -> None:
    output = {'motion': torch.full((2, 6), .5),
              'gripper_logits': torch.tensor([[4., -4.], [-4., 4.]])}
    action = torch.zeros((2, 7))
    action[:, 6] = torch.tensor([-1., 1.])
    terms = bc_loss(output, action)
    assert terms['motion_mse'].item() == pytest.approx(.25)
    assert terms['gripper_ce'].item() == pytest.approx(math.log1p(math.exp(-8.)), rel=5e-4)
    assert terms['loss'].item() == pytest.approx(.25 + math.log1p(math.exp(-8.)))
    decoded = decode_actions(output)
    assert decoded.shape == (2, 7) and decoded.dtype == torch.float32
    torch.testing.assert_close(decoded[:, :6], output['motion'], rtol=0, atol=0)
    torch.testing.assert_close(decoded[:, 6], torch.tensor([-1., 1.]), rtol=0, atol=0)


def test_zero_gripper_supervision_is_explicitly_rejected() -> None:
    output = {'motion': torch.zeros((3, 6)), 'gripper_logits': torch.zeros((3, 2))}
    action = torch.zeros((3, 7))
    action[:, 6] = torch.tensor([-1., 0., 1.])
    with pytest.raises(ValueError, match='gripper|hold|label'):
        bc_loss(output, action)


@pytest.mark.parametrize('mean, std', [
    (torch.zeros(7), torch.ones(8)),
    (torch.zeros(8), torch.tensor([0., 1., 1., 1., 1., 1., 1., 1.])),
    (torch.zeros(8), -torch.ones(8)),
    (torch.zeros(8), torch.full((8,), float('inf'))),
    (torch.full((8,), float('nan')), torch.ones(8)),
])
def test_invalid_normalization_statistics_are_rejected(mean, std) -> None:
    with pytest.raises(ValueError, match='mean|std|finite|shape'):
        VisualBC(mean, std)


@pytest.mark.parametrize('damage', ['camera_channels', 'camera_batch', 'state_width', 'empty_batch'])
def test_mismatched_forward_shapes_are_rejected(real_batch, damage: str) -> None:
    batch = dict(real_batch)
    if damage == 'camera_channels':
        batch['agentview_image'] = batch['agentview_image'][:, :2]
    elif damage == 'camera_batch':
        batch['wrist_image'] = batch['wrist_image'][:3]
    elif damage == 'state_width':
        batch['robot_state'] = batch['robot_state'][:, :7]
    else:
        batch = {key: value[:0] for key, value in batch.items()}
    with pytest.raises(ValueError, match='Expected|shape|camera|state'):
        predict(VisualBC(), batch)


def test_loss_rejects_wrong_action_width(real_batch) -> None:
    output = predict(VisualBC(), real_batch)
    with pytest.raises(ValueError, match='action|shape'):
        bc_loss(output, real_batch['action'][:, :6])


def test_fixed_pool_matches_legacy_adaptive_pool_predictions(real_batch) -> None:
    model = VisualBC().eval()
    legacy = deepcopy(model)
    legacy.image_encoder[8] = torch.nn.AdaptiveAvgPool2d((4, 4))
    with torch.no_grad():
        actual, expected = predict(model, real_batch), predict(legacy, real_batch)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_backward_supports_strict_determinism(real_batch) -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        batch = {key: value.cuda() for key, value in real_batch.items()}
        model = VisualBC().cuda()
        bc_loss(predict(model, batch), batch['action'])['loss'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    finally:
        torch.use_deterministic_algorithms(previous)
