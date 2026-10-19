"""Serve the verified official JAX pi05_libero policy on loopback, without training."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / '.deps/openpi-cache/openpi-assets/checkpoints/pi05_libero')
    parser.add_argument('--port', type=int, default=8006)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be in [1,65535]')
    if args.manifest.exists():
        raise FileExistsError(f'Preserve existing launch manifest: {args.manifest}')
    pin = json.loads((ROOT / 'openpi-upstream.json').read_text())
    source = ROOT / '.deps' / ('openpi-' + pin['source']['commit'])
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != pin['source']['commit']:
        raise ValueError('OpenPI source differs from the reviewed commit')
    subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--'], check=True)
    if hashlib.sha256((source / 'uv.lock').read_bytes()).hexdigest() != pin['source']['uv_lock_sha256']:
        raise ValueError('OpenPI dependency lock differs from the reviewed version')
    # No download here: refuse an incomplete or substituted checkpoint.
    for item in pin['checkpoint']['files']:
        path = args.checkpoint / item['path']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != item['expected_size'] or digest.hexdigest() != item['sha256']:
            raise ValueError(f'Official checkpoint fingerprint differs: {path}')

    import jax
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    if jax.devices()[0].platform != 'gpu':
        raise RuntimeError('pi05_libero requires the verified GPU backend; no CPU fallback')
    if not Path(policy_config.__file__).resolve().is_relative_to(source.resolve()):
        raise ValueError('The active interpreter imported another OpenPI installation')
    started = perf_counter()
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), args.checkpoint)
    info = {'run_id': uuid4().hex, 'openpi_commit': revision, 'config': 'pi05_libero',
            'checkpoint': pin['checkpoint']['uri'], 'checkpoint_sha256': pin['checkpoint']['content_sha256'],
            'backend': 'jax', 'jax': metadata.version('jax'), 'jaxlib': metadata.version('jaxlib'),
            'gpu': jax.devices()[0].device_kind, 'pid': os.getpid(),
            'model_load_seconds': perf_counter() - started, 'action_horizon': 10,
            'server_rng': 'official key(0), advances per request; reset does not reset server',
            'xla_flags': os.environ.get('XLA_FLAGS', ''),
            'ptxas': subprocess.check_output(['ptxas', '--version'], text=True).strip(),
            'compilation_cache': os.environ.get('JAX_COMPILATION_CACHE_DIR'),
            'norm_stats_sha256': next(item['sha256'] for item in pin['checkpoint']['files']
                                      if item['path'].endswith('/norm_stats.json'))}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open('x') as stream:
        json.dump(info, stream, indent=2)
        stream.write('\n')
    print(json.dumps(info, indent=2), flush=True)
    original_sample = policy._sample_actions
    timings = []

    def timed_sample(*pos, **kwargs):
        start = perf_counter()
        result = original_sample(*pos, **kwargs)
        jax.block_until_ready(result)
        timings.append((perf_counter() - start) * 1000.)
        return result

    # A timing boundary around the upstream compiled sampler changes no weights,
    # model inputs, noise draws, action transforms or execution order.
    policy._sample_actions = timed_sample

    class MeasuredPolicy:
        def infer(self, observation: dict) -> dict:
            result = policy.infer(observation)
            result['synchronized_sample_ms'] = timings[-1]
            result['server_request_index'] = len(timings) - 1
            return result

    WebsocketPolicyServer(MeasuredPolicy(), host='127.0.0.1', port=args.port, metadata=info).serve_forever()


if __name__ == '__main__':
    main()
