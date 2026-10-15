"""Train single-task VisualBC from an audited official LIBERO dataset."""
import argparse
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from embodiedlab.data.episode import EpisodeDataset
from embodiedlab.data.torch_dataset import LiberoTorchDataset, split_episodes
from embodiedlab.policies.visual_bc import VisualBC, bc_loss, decode_actions
from embodiedlab.policies.checkpoint import make_contract

ROOT = Path(__file__).resolve().parents[1]


def save_json(path: Path, value: dict | list) -> None:
    """Write finite, human-readable experiment metadata."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_hash(path: Path) -> str:
    """Fingerprint the original file without modifying it."""
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def training_statistics(loader: DataLoader) -> dict:
    """Fit state normalization and constant predictions using training samples only."""
    states, actions = [], []
    for batch in loader:
        # Keep only small numeric arrays; RGB batches are released each iteration.
        states.append(batch['robot_state'].double())
        actions.append(batch['action'].double())
    state, action = torch.cat(states), torch.cat(actions)
    gripper = action[:, 6]
    if not torch.all((gripper == -1) | (gripper == 1)):
        raise ValueError('Expected only -1/+1 expert gripper labels')
    raw_std = state.std(dim=0, unbiased=False)
    counts = torch.bincount((gripper > 0).long(), minlength=2)
    # Some legitimate tasks use only open-gripper demonstrations. Keep the
    # network's two-class semantics; Laplace smoothing only makes the constant
    # baseline finite if validation happens to contain a class absent in train.
    probabilities = (counts + 1) / (counts.sum() + 2)
    return {'samples': len(state), 'state_mean': state.mean(dim=0).tolist(),
            'state_std_raw': raw_std.tolist(), 'state_std': raw_std.clamp_min(.001).tolist(),
            'motion_mean': action[:, :6].mean(dim=0).tolist(),
            'gripper_counts': counts.tolist(), 'gripper_probabilities': probabilities.tolist(),
            'constant_gripper_laplace_count': 1}


def run_epoch(model: VisualBC, batches: Iterable[dict[str, torch.Tensor]], device: torch.device,
              optimizer: torch.optim.Optimizer | None = None,
              constant: dict | None = None) -> dict:
    """Run train/eval batches and report normalized OSC errors and gripper confusion."""
    model.train(optimizer is not None)
    sums = {key: 0. for key in ('loss', 'motion_mse', 'gripper_ce')}
    squared, absolute = torch.zeros(6, dtype=torch.float64), torch.zeros(6, dtype=torch.float64)
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    count = 0
    started = time.perf_counter()
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            target = batch['action']
            if constant is None:
                output = model(batch['agentview_image'], batch['wrist_image'], batch['robot_state'])
            else:
                output = {'motion': torch.tensor(constant['motion_mean'], device=device).expand(len(target), -1),
                          'gripper_logits': torch.tensor(constant['gripper_probabilities'], device=device).log().expand(len(target), -1)}
            losses = bc_loss(output, target)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Nonfinite BC loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses['loss'].backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite BC gradient')
                optimizer.step()
            size = len(target)
            count += size
            for key in sums:
                sums[key] += float(losses[key].detach()) * size
            error = (output['motion'].detach() - target[:, :6]).double().cpu()
            squared += error.square().sum(dim=0)
            absolute += error.abs().sum(dim=0)
            truth = (target[:, 6] > 0).long()
            predicted = output['gripper_logits'].argmax(dim=1)
            confusion += torch.bincount((truth * 2 + predicted).cpu(), minlength=4).reshape(2, 2)
    if count == 0:
        raise ValueError('Cannot evaluate an empty loader')
    mse, mae = squared / count, absolute / count
    precision = confusion.diag() / confusion.sum(dim=0).clamp_min(1)
    recall = confusion.diag() / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {**{key: value / count for key, value in sums.items()}, 'samples': count,
            'seconds': time.perf_counter() - started,
            'per_dim_mse': mse.tolist(), 'per_dim_mae': mae.tolist(),
            'translation_mse': float(mse[:3].mean()), 'rotation_mse': float(mse[3:].mean()),
            'translation_mae': float(mae[:3].mean()), 'rotation_mae': float(mae[3:].mean()),
            'gripper_accuracy': float(confusion.diag().sum() / count),
            'gripper_confusion': confusion.tolist(), 'gripper_precision': precision.tolist(),
            'gripper_recall': recall.tolist(), 'gripper_macro_f1': float(f1.mean())}


def check_gradients(model: VisualBC, batch: dict, device: torch.device) -> dict:
    """Use a real batch to check finite forward/backward without updating weights."""
    probe = deepcopy(model).to(device)
    data = {key: value.to(device) for key, value in batch.items()}
    output = probe(data['agentview_image'], data['wrist_image'], data['robot_state'])
    losses = bc_loss(output, data['action'])
    losses['loss'].backward()
    grads = [p.grad for p in probe.parameters() if p.requires_grad]
    if not torch.isfinite(losses['loss']) or any(g is None or not torch.isfinite(g).all() for g in grads):
        raise FloatingPointError(f'Invalid gradients on {device}')
    norm = float(torch.sqrt(sum(g.detach().square().sum() for g in grads)))
    if norm == 0:
        raise FloatingPointError('All gradients are zero')
    return {'device': str(device), 'loss': float(losses['loss'].detach()), 'gradient_l2': norm,
            'action_shape': list(decode_actions(output).shape), 'all_gradients_finite': True}


def plot_history(history: list[dict], output: Path, mode: str) -> None:
    """Save one loss figure from measured train and validation records."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for axis, key in zip(axes, ('loss', 'motion_mse', 'gripper_ce')):
        if mode == 'overfit':
            axis.plot([r['step'] for r in history], [r['metrics'][key] for r in history])
            axis.set_xlabel('Updates on the same batch')
        else:
            for split in ('train', 'validation'):
                axis.plot([r['epoch'] for r in history], [r[split][key] for r in history], label=split)
            axis.set_xlabel('Epoch (0 = untrained)')
            axis.legend()
        axis.set_title(key)
        axis.set_yscale('log')
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/libero/libero_goal/turn_on_the_stove_demo.hdf5')
    parser.add_argument('--suite', default=None, help='Must match the audited dataset identity')
    parser.add_argument('--task-id', type=int, default=None)
    parser.add_argument('--train-fraction', type=float, default=.8)
    parser.add_argument('--split-seed', type=int, default=None, help='Defaults to --seed')
    parser.add_argument('--mode', choices=('overfit', 'train'), default='train')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--steps', type=int, default=300, help='Updates in overfit mode only')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch_size, args.threads) < 1 or args.seed < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Epochs, steps, batch size, threads and lr must be positive; seed >= 0')
    if ((args.suite is None) != (args.task_id is None) or not 0 < args.train_fraction < 1
            or (args.task_id is not None and args.task_id < 0)
            or (args.split_seed is not None and args.split_seed < 0)):
        parser.error('Specify suite and task-id together; train-fraction in (0,1), seeds/IDs >= 0')
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error(f'CUDA unavailable in this process (torch={torch.__version__}, runtime={torch.version.cuda}); '
                     'check the GPU environment and NVIDIA device visibility')
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print(f'Device: {device} ({device_name}); torch={torch.__version__}; '
          f'CUDA runtime={torch.version.cuda}; CUDA available={torch.cuda.is_available()}', flush=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + args.mode
    output_dir = args.output_dir or ROOT / 'training_results' / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before_hash = file_hash(args.dataset)
    names = EpisodeDataset.list_episodes(args.dataset)
    train_names, val_names = split_episodes(names, seed=args.split_seed, train_fraction=args.train_fraction)
    with LiberoTorchDataset(args.dataset, train_names, suite=args.suite, task_id=args.task_id) as train_data, \
            LiberoTorchDataset(args.dataset, val_names, suite=args.suite, task_id=args.task_id) as val_data:
        identity = train_data.metadata['verified_source']
        args.suite, args.task_id = identity['suite'], identity['task_id']
        contract = make_contract(train_data.metadata, train_episodes=train_names,
                                 validation_episodes=val_names, split_seed=args.split_seed)
        print(f"Task: {args.suite}/{args.task_id} {identity['task_name']}; "
              f"episodes={len(names)} split={len(train_names)}/{len(val_names)}; "
              f"samples={len(train_data)}/{len(val_data)}", flush=True)
        def loader(dataset, *, shuffle=False):
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
                              generator=torch.Generator().manual_seed(args.seed))
        stats = training_statistics(loader(train_data))
        model = VisualBC(torch.tensor(stats['state_mean']), torch.tensor(stats['state_std'])).to(device)
        # This independent loader does not advance the main training shuffle RNG.
        fixed_batch = next(iter(loader(train_data, shuffle=True)))
        verification_batch = {key: value[:4].to(device) for key, value in fixed_batch.items()}
        checks = [check_gradients(model, fixed_batch, torch.device('cpu'))]
        if device.type == 'cuda':
            checks.append(check_gradients(model, fixed_batch, torch.device('cuda')))
        config = {**vars(args), 'dataset': str(args.dataset.resolve()), 'output_dir': str(output_dir.resolve()),
                  'device': str(device), 'torch_version': str(torch.__version__), 'torch_cuda': torch.version.cuda,
                  'python_version': platform.python_version(), 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
                  'cuda_available': torch.cuda.is_available(), 'numpy_version': np.__version__,
                  'dataset_sha256': before_hash, 'provenance': train_data.metadata['provenance'],
                  'split_seed': args.split_seed, 'train_episodes': list(train_names), 'validation_episodes': list(val_names),
                  'train_samples': len(train_data), 'validation_samples': len(val_data),
                  'parameters': sum(p.numel() for p in model.parameters()), 'statistics': stats,
                  'alignment': 'obs[t-1] -> action[t]', 'images': 'unchanged RGB/CHW float32 [0,1]',
                  'gripper_classes': [-1, 1], 'loss': 'mean MSE(6 OSC deltas) + mean CE(2 gripper classes)',
                  'architecture': 'visual_bc_v1', 'deterministic_algorithms': True,
                  'contract': contract}
        save_json(output_dir / 'config.json', config)
        save_json(output_dir / 'gradient_check.json', checks)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        history, best_reference = [], None
        best_loss, best_epoch = float('inf'), 0

        def save_checkpoint(epoch: int, metrics: dict) -> None:
            nonlocal best_reference
            model.eval()
            with torch.no_grad():
                reference = model(verification_batch['agentview_image'], verification_batch['wrist_image'], verification_batch['robot_state'])
                best_reference = {key: value.detach().clone() for key, value in reference.items()}
            torch.save({'format_version': 2, 'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'metrics': metrics, 'config': config,
                        'contract': contract}, output_dir / 'best.pt')

        if args.mode == 'overfit':
            initial = run_epoch(model, [fixed_batch], device)
            history.append({'step': 0, 'metrics': initial})
            best_loss = initial['loss']
            save_checkpoint(0, initial)
            print('overfit initial', initial['loss'], flush=True)
            for step in range(1, args.steps + 1):
                run_epoch(model, [fixed_batch], device, optimizer)
                if step % 10 == 0 or step == args.steps:
                    metrics = run_epoch(model, [fixed_batch], device)
                    history.append({'step': step, 'metrics': metrics})
                    if metrics['loss'] < best_loss:
                        best_loss, best_epoch = metrics['loss'], step
                        save_checkpoint(step, metrics)
                    print(f'overfit step={step} loss={metrics["loss"]:.6f} motion={metrics["motion_mse"]:.6f} grip_acc={metrics["gripper_accuracy"]:.3f}', flush=True)
            final = {'initial': initial, 'last': history[-1]['metrics'], 'best_step': best_epoch}
        else:
            train_loader, val_loader = loader(train_data, shuffle=True), loader(val_data)
            baseline = {'train': run_epoch(model, loader(train_data), device, constant=stats),
                        'validation': run_epoch(model, val_loader, device, constant=stats)}
            save_json(output_dir / 'constant_baseline.json', baseline)
            initial = {'epoch': 0, 'train': run_epoch(model, loader(train_data), device),
                       'validation': run_epoch(model, val_loader, device)}
            history.append(initial)
            best_loss = initial['validation']['loss']
            save_checkpoint(0, initial['validation'])
            for epoch in range(1, args.epochs + 1):
                train_metrics = run_epoch(model, train_loader, device, optimizer)
                val_metrics = run_epoch(model, val_loader, device)
                history.append({'epoch': epoch, 'train': train_metrics, 'validation': val_metrics})
                save_json(output_dir / 'history.json', history)
                if val_metrics['loss'] < best_loss:
                    best_loss, best_epoch = val_metrics['loss'], epoch
                    save_checkpoint(epoch, val_metrics)
                print(f'epoch={epoch} train={train_metrics["loss"]:.6f} val={val_metrics["loss"]:.6f} motion={val_metrics["motion_mse"]:.6f} grip_acc={val_metrics["gripper_accuracy"]:.3f} seconds={train_metrics["seconds"] + val_metrics["seconds"]:.2f}', flush=True)
            final = {'best_epoch': best_epoch, 'constant_baseline': baseline, 'last': history[-1]}

        checkpoint = torch.load(output_dir / 'best.pt', map_location=device, weights_only=True)
        restored = VisualBC().to(device)
        restored.load_state_dict(checkpoint['model_state_dict'])
        restored.eval()
        with torch.no_grad():
            prediction = restored(verification_batch['agentview_image'], verification_batch['wrist_image'], verification_batch['robot_state'])
        errors = {key: float((prediction[key] - best_reference[key]).abs().max()) for key in prediction}
        if max(errors.values()) > 1e-6:
            raise AssertionError(f'Checkpoint restore mismatch: {errors}')
        final['restored_best'] = ({'batch': run_epoch(restored, [fixed_batch], device)} if args.mode == 'overfit'
                                  else {'train': run_epoch(restored, loader(train_data), device),
                                        'validation': run_epoch(restored, loader(val_data), device)})
        final.update({'checkpoint_output_max_error': errors, 'dataset_unchanged': file_hash(args.dataset) == before_hash,
                      'total_seconds': time.perf_counter() - started})
        if not final['dataset_unchanged']:
            raise AssertionError('Original HDF5 changed')
        save_json(output_dir / 'history.json', history)
        save_json(output_dir / 'summary.json', final)
        plot_history(history, output_dir / 'losses.png', args.mode)
        print('saved', output_dir, 'seconds=', round(final['total_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
