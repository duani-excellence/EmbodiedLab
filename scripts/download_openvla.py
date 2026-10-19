"""Download/verify only the fixed official joint OpenVLA-OFT checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter, sleep, time_ns
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path, item: dict) -> bool:
    """Compare a complete local object with independently pinned official bytes."""
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == item['sha256']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--verification-file', type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads((ROOT / 'openvla-upstream.json').read_text())['checkpoint']
    directory = args.output_dir or ROOT / '.deps/openvla-cache' / pin['revision']
    if args.verification_file.exists():
        raise FileExistsError('Preserve the existing verification report; choose a new path')
    started = perf_counter()

    def fetch(item: dict) -> None:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe checkpoint object path')
        target = directory / relative
        if target.exists():
            if not verify(target, item):
                raise ValueError(f'Existing checkpoint object differs; left untouched: {target}')
        else:
            if args.verify_only:
                raise FileNotFoundError(f'Missing official object: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.partial')
            # A fresh curl process resumes from the bytes actually received.
            # Internal curl retries can restart the original range after EOF.
            for attempt in range(20):
                url = (f'https://huggingface.co/{pin["repo_id"]}/resolve/{pin["revision"]}/'
                       + quote(item['path'], safe='/') + f'?download=true&resume={time_ns()}')
                run = subprocess.run(['curl', '--http1.1', '--fail', '--location', '--silent',
                    '--show-error', '--connect-timeout', '30', '--max-time', '1200',
                    '--continue-at', '-', '--output', str(partial), url])
                if run.returncode == 0:
                    break
                sleep(2)
            else:
                raise RuntimeError(f'Download interrupted; partial preserved: {partial}')
            if not verify(partial, item):
                raise ValueError(f'Official object checksum mismatch; partial preserved: {partial}')
            partial.rename(target)
        print('Verified', item['path'], flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, pin['files']))
    report = {'repo_id': pin['repo_id'], 'revision': pin['revision'], 'files': pin['files'],
              'total_bytes': pin['total_bytes'], 'content_sha256': pin['content_sha256'],
              'status': 'verified', 'elapsed_seconds': perf_counter() - started}
    args.verification_file.parent.mkdir(parents=True, exist_ok=True)
    with args.verification_file.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print('Verified official checkpoint:', pin['content_sha256'])


if __name__ == '__main__':
    main()
