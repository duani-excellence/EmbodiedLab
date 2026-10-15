"""Download versioned upstream assets or one official demo; never overwrite user data."""
import argparse
import hashlib
import importlib.util
from importlib.resources import files
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Fetch pinned Hub files and install the native package's local-asset link."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',action='store_true')
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--suite', default=None, help='Select one audited official dataset')
    parser.add_argument('--task-id', type=int, default=None)
    args = parser.parse_args()
    if not args.assets and not args.demo:
        parser.error('select --assets and/or --demo')
    if (args.suite is None) != (args.task_id is None):
        parser.error('Specify --suite and --task-id together')
    from huggingface_hub import hf_hub_download, snapshot_download
    pins = json.loads((ROOT/'upstream.json').read_text())
    cache = ROOT/'.deps/hf-cache'
    if args.assets:
        spec = importlib.util.find_spec('libero')
        if spec is None:
            raise ImportError('Install the LIBERO optional dependencies first')
        link = Path(next(iter(spec.submodule_search_locations)))/'libero/assets'
        pin = pins['assets']
        destination = ROOT/'.deps'/f'libero-assets-{pin["revision"]}'
        snapshot_download(repo_id=pin['repo_id'],repo_type=pin['repo_type'],revision=pin['revision'],
                          local_dir=destination,cache_dir=cache,max_workers=4,token=False)
        # get_assets_path() uses package/assets before auto-downloading latest.
        # A local link supplies original pinned assets without patching upstream.
        if link.is_symlink():
            if link.resolve() != destination.resolve():
                raise ValueError(f'Existing native assets link points elsewhere: {link}')
        elif link.exists():
            raise ValueError(f'Native assets already exist; use a clean isolated environment: {link}')
        else:
            link.symlink_to(destination,target_is_directory=True)
        print(f'Pinned native assets: {destination}',flush=True)
    if args.demo:
        if args.suite is None:
            pin = pins['demo']  # Preserve the original default download.
        else:
            audits = json.loads(files('embodiedlab.data').joinpath('verified_sources.json').read_text())
            matches = [source for source in audits['sources']
                       if (source['suite'], source['task_id']) == (args.suite, args.task_id)]
            if len(matches) != 1:
                raise ValueError('No audited data for this task; review source/alignment before adding a dataset')
            pin = matches[0]
        path = ROOT/'data/libero'/pin['path']
        if not path.exists():
            path = Path(hf_hub_download(repo_id=pin['repo_id'],repo_type='dataset',revision=pin['revision'],
                                      filename=pin['path'],local_dir=ROOT/'data/libero',cache_dir=cache,token=False))
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda:source.read(1024*1024),b''):
                digest.update(chunk)
        if path.stat().st_size != pin['size'] or digest.hexdigest() != pin['sha256']:
            raise ValueError(f'Existing demo checksum differs; file left untouched: {path}')
        sidecar = path.with_suffix('.source.json')
        if not sidecar.exists():
            sidecar.write_text(json.dumps(pin,indent=2)+'\n')
        print(f'Verified official demo: {path}',flush=True)


if __name__ == '__main__':
    main()
