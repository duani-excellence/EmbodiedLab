"""Download only the generation-pinned official pi05_libero checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path, item: dict) -> bool:
    """Check a complete object against the previously audited official bytes."""
    if not path.is_file() or path.stat().st_size != item['expected_size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == item['sha256']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / '.deps/openpi-cache/openpi-assets/checkpoints/pi05_libero')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    pin = json.loads((ROOT / 'openpi-upstream.json').read_text())['checkpoint']

    def fetch(item: dict) -> str:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe checkpoint object path')
        target = args.output_dir / relative
        if target.exists():
            if not verify(target, item):
                raise ValueError(f'Existing object checksum differs; left untouched: {target}')
            return item['path']
        if args.verify_only:
            raise FileNotFoundError(f'Missing official checkpoint object: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.partial')
        url = ('https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/'
               + quote(item['path'], safe='/') + '?generation=' + item['generation'])
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--retry', '5', '--retry-all-errors', '--connect-timeout', '30',
                        '--max-time', '7200', '--continue-at', '-', '--output', str(partial), url], check=True)
        if not verify(partial, item):
            raise ValueError(f'Official object checksum mismatch; partial retained: {partial}')
        partial.rename(target)
        return item['path']

    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(fetch, pin['files']):
            print(f'Verified {name}', flush=True)
    print(f'Official pi05_libero: {len(pin["files"])} files, {pin["total_bytes"]} bytes; '
          f'content identity {pin["content_sha256"]}', flush=True)


if __name__ == '__main__':
    main()
