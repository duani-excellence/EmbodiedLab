"""Measure FP32 CPU/CUDA BC updates on real data, without a full training epoch."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import threading
import time

import torch
from torch.utils.data import DataLoader

from embodiedlab.data.episode import EpisodeDataset
from embodiedlab.data.torch_dataset import LiberoTorchDataset, split_episodes
from embodiedlab.policies.visual_bc import VisualBC, bc_loss

ROOT = Path(__file__).resolve().parents[1]
WARMUP = 10


def file_hash(path: Path) -> str:
    """Fingerprint source bytes without depending on the training entry point."""
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    """Complete CUDA work before reading a wall-clock boundary."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def transfer(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=False) for key, value in batch.items()}


def loss_on(model: VisualBC, batch: dict) -> torch.Tensor:
    output = model(batch['agentview_image'], batch['wrist_image'], batch['robot_state'])
    return bc_loss(output, batch['action'])['loss']


def timing(values: list[float], batch_size: int) -> dict:
    return {'mean_ms': statistics.mean(values) * 1000., 'median_ms': statistics.median(values) * 1000.,
            'samples_per_second': len(values) * batch_size / sum(values)}


def gradients(model: VisualBC) -> dict:
    """Inspect gradients outside timed regions, rather than synchronizing every tensor."""
    values = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if any(value is None for value in values) or not torch.stack([torch.isfinite(value).all() for value in values]).all():
        raise FloatingPointError('Missing or nonfinite BC gradients')
    return {'finite': True, 'l2': float(torch.stack([value.detach().square().sum() for value in values]).sum().sqrt())}


def benchmark(initial: dict, fixed: dict, device: torch.device, steps: int,
              loader: DataLoader | None = None) -> tuple[dict, VisualBC]:
    """Time one independent run; pipeline runs also measure synchronous next(loader)."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = VisualBC()
    model.load_state_dict(initial)
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    probe = transfer(fixed, device)
    with torch.no_grad():
        initial_loss = float(loss_on(model, probe))
    for _ in range(WARMUP):
        optimizer.zero_grad(set_to_none=True)
        loss_on(model, probe).backward()
        optimizer.step()
    synchronize(device)
    warmup_gradients = gradients(model)
    del probe
    iterator = iter(loader) if loader is not None else None
    durations = {key: [] for key in ('data_loading', 'transfer', 'forward_loss', 'backward_adam', 'iteration')}
    losses = []
    for _ in range(steps):
        started = time.perf_counter()
        cpu_batch = next(iterator) if iterator is not None else fixed
        loaded = time.perf_counter()
        batch = transfer(cpu_batch, device)
        synchronize(device)
        moved = time.perf_counter()
        loss = loss_on(model, batch)
        synchronize(device)
        forwarded = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        synchronize(device)
        finished = time.perf_counter()
        for key, elapsed in [('data_loading', loaded - started), ('transfer', moved - loaded),
                             ('forward_loss', forwarded - moved), ('backward_adam', finished - forwarded),
                             ('iteration', finished - started)]:
            durations[key].append(elapsed)
        losses.append(float(loss.detach()))
    last_gradients = gradients(model)
    with torch.no_grad():
        final_loss = float(loss_on(model, transfer(fixed, device)))
    if not all(math.isfinite(value) for value in [initial_loss, final_loss, *losses]):
        raise FloatingPointError('Nonfinite measured BC loss')
    memory = None if device.type != 'cuda' else {
        'max_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'max_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    result = {'device': str(device), 'measured_updates': steps, 'warmup_updates': WARMUP,
              'samples': steps * len(fixed['action']),
              'timing': {key: timing(values, len(fixed['action'])) for key, values in durations.items()},
              'data_loading_wall_seconds': sum(durations['data_loading']) if iterator is not None else 0.,
              'initial_fixed_batch_loss': initial_loss, 'first_measured_pre_update_loss': losses[0],
              'last_measured_pre_update_loss': losses[-1], 'final_fixed_batch_loss': final_loss,
              'warmup_gradients': warmup_gradients, 'last_gradients': last_gradients,
              'cuda_allocator_peak': memory}
    print(str(device), 'pipeline' if loader is not None else 'fixed', result['timing']['iteration'], flush=True)
    return result, model.cpu()


def inference(model: VisualBC, batch: dict) -> dict:
    model.eval()
    with torch.no_grad():
        output = model(batch['agentview_image'], batch['wrist_image'], batch['robot_state'])
    return {key: value.cpu() for key, value in output.items()}


def output_errors(first: dict, second: dict) -> dict:
    return {key: {'max_abs_error': float((first[key] - second[key]).abs().max()),
                  'within_tolerance': torch.allclose(first[key], second[key], atol=1e-5, rtol=1e-4)}
            for key in first}


def checkpoint_checks(cpu_model: VisualBC, gpu_model: VisualBC, fixed: dict) -> dict:
    """Exercise actual CPU/GPU serialization through temporary weights-only files."""
    device = torch.device('cuda')
    gpu_batch = transfer(fixed, device)
    with tempfile.TemporaryDirectory(prefix='embodiedlab-bc-') as directory:
        cpu_path, gpu_path = Path(directory) / 'cpu.pt', Path(directory) / 'gpu.pt'
        torch.save(cpu_model.state_dict(), cpu_path)
        restored_gpu = VisualBC().to(device)
        restored_gpu.load_state_dict(torch.load(cpu_path, map_location=device, weights_only=True))
        cpu_to_gpu = output_errors(inference(cpu_model, fixed), inference(restored_gpu, gpu_batch))
        gpu_model.to(device)
        torch.save(gpu_model.state_dict(), gpu_path)
        restored_cpu = VisualBC()
        restored_cpu.load_state_dict(torch.load(gpu_path, map_location='cpu', weights_only=True))
        gpu_to_cpu = output_errors(inference(gpu_model, gpu_batch), inference(restored_cpu, fixed))
        gpu_model.cpu()
    return {'atol': 1e-5, 'rtol': 1e-4, 'cpu_checkpoint_to_gpu': cpu_to_gpu,
            'gpu_checkpoint_to_cpu': gpu_to_cpu}


def sample_gpu(stop: threading.Event, rows: list[dict]) -> None:
    """Board-global samples include desktop/other processes, not model utilization."""
    query = '--query-gpu=index,utilization.gpu,memory.used,memory.total,driver_version'
    while not stop.is_set():
        try:
            result = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, check=True, timeout=2)
            for line in result.stdout.splitlines():
                index, utilization, used, total, driver = [value.strip() for value in line.split(',')]
                rows.append({'time_monotonic_s': time.perf_counter(), 'gpu_index': int(index),
                             'utilization_percent': float(utilization), 'used_MiB': float(used),
                             'total_MiB': float(total), 'driver': driver})
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            rows.append({'error': str(error)})
            return
        stop.wait(.15)


def cuda_sanity() -> dict:
    matrix = torch.randn((128, 128), device='cuda', dtype=torch.float32)
    product = matrix @ matrix.T
    synchronize(torch.device('cuda'))
    reference = matrix.cpu() @ matrix.cpu().T
    error = float((product.cpu() - reference).abs().max())
    if product.device.type != 'cuda' or not torch.isfinite(product).all():
        raise FloatingPointError('CUDA matrix multiplication failed')
    return {'device': str(product.device), 'dtype': str(product.dtype), 'shape': list(product.shape),
            'finite': True, 'cpu_reference_max_abs_error': error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/libero/libero_goal/turn_on_the_stove_demo.hdf5')
    parser.add_argument('--config', type=Path, default=ROOT / 'training_results/phase5b3_train_seed42/config.json')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--pipeline-batches', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.pipeline_batches, args.batch_size, args.threads) < 1:
        parser.error('Steps, pipeline batches, batch size and threads must be positive')
    if args.output.exists():
        parser.error('Output already exists; choose a new JSON path')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if not torch.cuda.is_available():
        parser.error('Real CUDA is required; run the GPU environment with device access')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    train, validation = split_episodes(EpisodeDataset.list_episodes(args.dataset), seed=42)
    config = json.loads(args.config.read_text())
    before_hash = file_hash(args.dataset)
    if (len(train) != 40 or len(validation) != 10 or config['train_episodes'] != list(train)
            or config.get('split_seed', config.get('seed')) != 42 or config['dataset_sha256'] != before_hash):
        raise ValueError('Config must contain statistics from this dataset and the fixed seed-42 train40 split')
    stats = config['statistics']
    model = VisualBC(torch.tensor(stats['state_mean']), torch.tensor(stats['state_std']))
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    initial_hash = hashlib.sha256(b''.join(key.encode() + value.numpy().tobytes() for key, value in initial.items())).hexdigest()
    with LiberoTorchDataset(args.dataset, train) as dataset:
        def loader() -> DataLoader:
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              drop_last=True, generator=torch.Generator().manual_seed(42))
        if args.pipeline_batches > len(loader()) or not len(loader()):
            raise ValueError('Requested pipeline batches exceed one complete training pass')
        fixed = next(iter(loader()))
        cpu_fixed, cpu_model = benchmark(initial, fixed, torch.device('cpu'), args.steps)
        cpu_pipeline, _ = benchmark(initial, fixed, torch.device('cpu'), args.pipeline_batches, loader())
        samples, stop = [], threading.Event()
        sampler = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
        sampler.start()
        try:
            sanity = cuda_sanity()
            gpu_fixed, gpu_model = benchmark(initial, fixed, torch.device('cuda'), args.steps)
            checkpoints = checkpoint_checks(cpu_model, gpu_model, fixed)
            gpu_pipeline, _ = benchmark(initial, fixed, torch.device('cuda'), args.pipeline_batches, loader())
        finally:
            stop.set()
            sampler.join(timeout=3)
        report = {'torch_version': str(torch.__version__), 'torch_cuda': torch.version.cuda,
                  'device_name': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
                  'compiled_architectures': torch.cuda.get_arch_list(), 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
                  'seed': 42, 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                  'dtype': 'float32', 'amp': False, 'tf32_matmul': False, 'tf32_cudnn': False,
                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                  'dataset_sha256': before_hash, 'train_episodes': list(train), 'validation_episodes': list(validation),
                  'train_samples': len(dataset), 'batch_size': args.batch_size, 'num_workers': 0,
                  'normalization_config': str(args.config.resolve()), 'initial_state_dict_sha256': initial_hash,
                  'fixed': {'cpu': cpu_fixed, 'cuda': gpu_fixed}, 'pipeline': {'cpu': cpu_pipeline, 'cuda': gpu_pipeline},
                  'cuda_sanity': sanity, 'checkpoints': checkpoints, 'nvidia_smi_samples': samples,
                  'sampling_note': 'Board-global, includes desktop/other workloads; target sleep 0.15 s, GPU phase only.',
                  'timing_note': 'Synchronized wall time; excludes loss/gradient inspection, probes and checkpoint checks. CPU transfer is a no-op. Fixed batches never reload data. Both runs use 10 fixed-batch warmup updates.',
                  'cache_note': 'CPU runs precede GPU; HDF5/OS caches are warm, not a cold-disk benchmark.',
                  'memory_note': 'Actual Torch allocator peaks, including warmup/model/Adam/batches/probes; excludes non-Torch driver and desktop allocations.'}
    report['dataset_unchanged'] = file_hash(args.dataset) == before_hash
    if not report['dataset_unchanged']:
        raise AssertionError('Original dataset changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    print('Saved', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
