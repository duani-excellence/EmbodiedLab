"""Serve fixed official OpenVLA-OFT on loopback; weights remain immutable."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from threading import Lock
from time import perf_counter
from types import SimpleNamespace
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = 'e4287e94541f459edc4feabc4e181f537cd569a8'
TRANSFORMERS_SHA = 'bc339d9ad707454c0c115970db43c260067c61ab'
REVISION = '638918f3d1c2e43a39a8a20772bdb8b91835e4b7'
REPO = 'moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8007)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    runtime = args.runtime_dir.resolve()
    if args.manifest.exists() or runtime.exists():
        raise FileExistsError('Use new manifest/runtime paths; preserve previous launches')
    verification = json.loads(args.verification.read_text())
    pin = json.loads((ROOT / 'openvla-upstream.json').read_text())
    if (verification['repo_id'], verification['revision'], verification['status']) != (REPO, REVISION, 'verified'):
        raise ValueError('Only the fixed verified joint OFT checkpoint is supported')
    if verification['content_sha256'] != pin['checkpoint']['content_sha256']:
        raise ValueError('Verification file differs from independently pinned official object hashes')
    if verification['files'] != pin['checkpoint']['files']:
        raise ValueError('Checkpoint object manifest differs from fixed official identities')
    source = ROOT / '.deps' / ('openvla-oft-' + SOURCE_SHA)
    fork = ROOT / '.deps' / ('transformers-openvla-oft-' + TRANSFORMERS_SHA)
    for path, expected in ((source, SOURCE_SHA), (fork, TRANSFORMERS_SHA)):
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise ValueError('Official source revision differs')
        subprocess.run(['git', '-C', str(path), 'diff', '--quiet', 'HEAD', '--'], check=True)
    # The upstream local loader syncs config/model Python. Give it disposable
    # copies of small files, never writable aliases to the verified originals.
    runtime.mkdir(parents=True, exist_ok=False)
    for item in verification['files']:
        path = checkpoint / item['path']
        if path.stat().st_size != item['size'] or sha256(path) != item['sha256']:
            raise ValueError(f'Checkpoint changed: {item["path"]}')
        target = runtime / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in ('.pt', '.safetensors'):
            target.symlink_to(path)
        else:
            shutil.copy2(path, target)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')  # TF only performs official image preprocessing.
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    import torch
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve
    from experiments.robot.openvla_utils import (
        get_vla, get_processor, get_action_head, get_proprio_projector,
        get_vla_action, resize_image_for_policy, prepare_images_for_vla, normalize_proprio)
    from experiments.robot.robot_utils import (
        set_seed_everywhere, normalize_gripper_action, invert_gripper_action)
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU; refuse to pretend OFT GPU inference ran')
    if metadata.version('torch') != '2.9.1+cu128' or metadata.version('transformers') != '4.40.1':
        raise ValueError('Use the verified isolated CUDA Torch and official Transformers fork')
    os.chdir(source)  # Upstream local model-sync searches ./prismatic.
    set_seed_everywhere(args.seed)
    cfg = SimpleNamespace(pretrained_checkpoint=str(runtime), use_l1_regression=True,
        use_diffusion=False, use_film=False, num_images_in_input=2, use_proprio=True,
        load_in_8bit=False, load_in_4bit=False, center_crop=True, num_open_loop_steps=8,
        unnorm_key='libero_spatial_no_noops', num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50)
    start = perf_counter()
    model = get_vla(cfg)
    processor = get_processor(cfg)
    head = get_action_head(cfg, llm_dim=model.llm_dim)
    projector = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=8)
    attention = model.language_model.config._attn_implementation
    if attention != 'sdpa':
        raise ValueError(f'Official bidirectional SDPA required, got {attention}')
    if next(model.parameters()).device.type != 'cuda':
        raise RuntimeError('Official model parameters must be on CUDA')
    info = {'run_id': uuid4().hex, 'model': 'openvla_oft', 'source_commit': SOURCE_SHA,
        'transformers_commit': TRANSFORMERS_SHA, 'checkpoint': REPO, 'revision': REVISION,
        'checkpoint_sha256': verification['content_sha256'], 'backend': 'torch',
        'gpu': torch.cuda.get_device_name(), 'pid': os.getpid(), 'seed': args.seed,
        'action_horizon': 8, 'dtype': 'bfloat16', 'quantization': False,
        'attention': attention, 'model_load_seconds': perf_counter() - start,
        'launcher_sha256': sha256(Path(__file__).resolve()),
        'framework_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'parameter_counts': {'base_model': sum(p.numel() for p in model.parameters()),
                             'action_head': sum(p.numel() for p in head.parameters()),
                             'proprio_projector': sum(p.numel() for p in projector.parameters())},
        'cuda_allocated_after_load_mib': torch.cuda.memory_allocated() / 2**20,
        'versions': {name: metadata.version(name) for name in
                     ('torch', 'torchvision', 'transformers', 'timm', 'tensorflow-cpu', 'tokenizers', 'peft')},
        'protocol': 'raw256 -> rotate180 -> official TF JPEG Lanczos224 -> center90%area -> processor; suite proprio/action stats; official grip transform'}
    args.manifest = args.manifest.resolve()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open('x') as stream:
        json.dump(info, stream, indent=2); stream.write('\n')
    print(json.dumps(info, indent=2), flush=True)
    requests = 0
    inference_lock = Lock()
    sample_times = []
    original_predict = model.predict_action

    def timed_predict(*pos, **kwargs):
        torch.cuda.synchronize()
        started = perf_counter()
        result = original_predict(*pos, **kwargs)
        torch.cuda.synchronize()
        sample_times.append((perf_counter() - started) * 1000.)
        return result

    # Timing only: preserve the official forward, action head and decode.
    model.predict_action = timed_predict

    def infer(payload: dict) -> dict:
        nonlocal requests
        suite = payload['suite']
        cfg.unnorm_key = suite + '_no_noops'
        if cfg.unnorm_key not in model.norm_stats:
            raise ValueError('No official joint checkpoint statistics for ' + suite)
        images = []
        for key in ('agentview_image', 'wrist_image'):
            image = np.asarray(payload[key])
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError('Expected raw256 RGB camera')
            images.append(resize_image_for_policy(np.ascontiguousarray(image[::-1, ::-1]), 224))
        state = np.array(payload['robot_state'], dtype=np.float64, copy=True)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError('Invalid native state')
        observation = {'full_image': images[0], 'wrist_image': images[1], 'state': state}
        # Synchronization around the official prediction captures preprocessing,
        # dispatch and GPU execution, not pure kernel-only profiler time.
        torch.cuda.synchronize()
        start = perf_counter()
        with torch.inference_mode():
            raw = np.asarray(get_vla_action(cfg, model, processor, observation, payload['prompt'],
                                           head, projector), dtype=np.float64)
        torch.cuda.synchronize()
        elapsed = (perf_counter() - start) * 1000.
        actions = invert_gripper_action(normalize_gripper_action(raw, binarize=True))
        preprocessing = {'resized_image_sha256': [hashlib.sha256(x.tobytes()).hexdigest() for x in images],
                         'normalized_state': observation['state'].tolist(), 'unnorm_key': cfg.unnorm_key,
                         'prompt': f'In: What action should the robot take to {payload["prompt"].lower()}?\nOut:'}
        result = {'actions': actions, 'raw_model_chunk': raw, 'preprocessing': preprocessing,
                  'synchronized_sample_ms': sample_times[-1], 'server_request_index': requests,
                  'policy_timing': {'infer_ms': elapsed,
                                   'timing_scope': 'total OFT pipeline; sample_ms is synchronized official predict_action',
                                   'torch_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}}
        requests += 1
        return result

    def connection(socket):
        socket.send(msgpack_numpy.packb(info))
        try:
            for data in socket:
                with inference_lock:
                    socket.send(msgpack_numpy.packb(infer(msgpack_numpy.unpackb(data))))
        except Exception:
            try:
                socket.send(traceback.format_exc())
            finally:
                socket.close()
            traceback.print_exc()

    with serve(connection, '127.0.0.1', args.port, compression=None, max_size=None) as server:
        print('Listening on loopback', args.port, flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
