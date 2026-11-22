"""Prepare the 159 registered pigeon windows directly from the provider ZIPs."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
from urllib.error import URLError
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from coordinates import elte_windows, huang_windows

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
HUANG_FILE = 'pigeon_homing_public_trials_v1.0.0.zip'
HUANG_URL = 'https://zenodo.org/records/21092705/files/' + HUANG_FILE + '?download=1'
RELEASE_SIZES = {HUANG_FILE: 126588691, 'trajectoriesD1.zip': 68598224, 'trajectoriesD2.zip': 84085139}
ARCHIVES = {'Huang': HUANG_FILE, 'D1': 'trajectoriesD1.zip', 'D2': 'trajectoriesD2.zip'}


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def download_archive(url, destination, size):
    # Some gateways terminate long responses without an HTTP error. Fetch short,
    # checked ranges so a partial response can be retried at low cost.
    block = 2 * 1024 * 1024
    def fetch_with_curl(start, end):
        # The OS curl and Python can use different TLS stacks or proxy settings.
        # Verify the range metadata before accepting a response from either.
        with tempfile.TemporaryDirectory() as tmp:
            header = Path(tmp) / 'response.txt'
            result = subprocess.run(['curl', '--fail', '--location', '--silent',
                                     '--show-error', '--max-time', '120', '--range',
                                     f'{start}-{end}', '--dump-header', str(header), url],
                                    capture_output=True, check=True)
            expected = f'content-range: bytes {start}-{end}/{size}'
            fields = header.read_text(encoding='latin1').lower().splitlines()
            if expected not in fields or len(result.stdout) != end - start + 1:
                raise OSError('Unexpected curl range response')
            return result.stdout

    def fetch(start):
        end = min(start + block, size) - 1
        headers = {'User-Agent': 'pigeon-gap-association-reproduction/1',
                   'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}
        for attempt in range(4):
            try:
                if attempt >= 2 and shutil.which('curl'):
                    return fetch_with_curl(start, end)
                with urlopen(Request(url, headers=headers), timeout=120) as response:
                    if start == 0 and response.status == 200:
                        with destination.open('wb') as out:
                            shutil.copyfileobj(response, out, length=1024 * 1024)
                        if destination.stat().st_size != size:
                            raise OSError('Truncated full-file response')
                        return None
                    expected = f'bytes {start}-{end}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise OSError(f'Unexpected response for {expected}')
                    data = response.read()
                    if len(data) != end - start + 1:
                        raise OSError(f'Truncated response for {expected}')
                    return data
            except (URLError, OSError, IncompleteRead, subprocess.CalledProcessError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    first = fetch(0)
    if first is None:
        return
    with destination.open('wb') as out, ThreadPoolExecutor(max_workers=3) as pool:
        out.write(first)
        for data in pool.map(fetch, range(block, size, block)):
            out.write(data)


def open_archive(directory, entry, download):
    archive = directory / Path(entry['path']).name
    if not archive.exists() and (directory / entry['path']).is_file():
        archive = directory / entry['path']
    if archive.is_file():
        if digest(archive) != entry['sha256']:
            raise ValueError(f'Incorrect SHA-256 for {archive}. Check the release version.')
        return archive
    if not download:
        raise FileNotFoundError(f'Missing {archive}. Supply the original ZIP or add --download.')
    url = HUANG_URL if archive.name == HUANG_FILE else entry['access']
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
        partial = Path(tmp.name)
    try:
        download_archive(url, partial, RELEASE_SIZES[archive.name])
        if digest(partial) != entry['sha256']:
            raise ValueError(f'{url}: download did not match the frozen archive SHA-256')
        os.replace(partial, archive)
    finally:
        partial.unlink(missing_ok=True)
    return archive


def position_digest(window):
    coords = np.ascontiguousarray(window[:, :, :3], dtype='<f4')
    return hashlib.sha256(coords.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True,
                        help='Directory with the original ZIPs (flat or registry layout).')
    parser.add_argument('--output', type=Path, required=True,
                        help='Root for registered window paths and preprocessing-report.json.')
    parser.add_argument('--download', action='store_true',
                        help='Fetch missing archives anonymously from the upstream providers.')
    parser.add_argument('--data', type=Path, default=DATA,
                        help='Directory containing source-provenance.json and frozen-registry.json.')
    args = parser.parse_args()
    provenance = {Path(r['path']).name: r for r in
                  json.loads((args.data / 'source-provenance.json').read_text())}
    registry = json.loads((args.data / 'frozen-registry.json').read_text())
    registered = {r['window']: r for r in registry['windows']}
    expected = json.loads((HERE / 'position-sha256.json').read_text())
    if set(expected) != set(registered) or len(registered) != 159:
        raise ValueError('Position manifest does not identify the frozen 159 windows')
    args.output.mkdir(exist_ok=True, parents=True)
    report = {'numpy': np.__version__, 'sources': {}, 'windows': []}
    seen = set()
    for source, filename in ARCHIVES.items():
        archive = open_archive(args.archives, provenance[filename], args.download)
        generated = huang_windows(archive) if source == 'Huang' else elte_windows(archive, source)
        count = byte_matches = 0
        for record, name, array in generated:
            reg = registered[name]
            if (reg['source'], reg['record']) != (source, record) or name in seen:
                raise ValueError(f'Unexpected or repeated window: {source}/{record}/{name}')
            if array.shape != (131, reg['targets'], 9) or array.dtype != np.float32:
                raise ValueError(f'{name}: unexpected window dimensions or dtype')
            if not np.isfinite(array).all() or position_digest(array) != expected[name]:
                raise ValueError(f'{name}: positions differ from the frozen inference input')
            dest = args.output / reg['path']
            dest.parent.mkdir(exist_ok=True, parents=True)
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                partial = Path(tmp.name)
            try:
                with partial.open('wb') as out:
                    np.savez_compressed(out, truth=array)
                checksum = digest(partial)
                os.replace(partial, dest)
            finally:
                partial.unlink(missing_ok=True)
            seen.add(name)
            count += 1
            byte_matches += checksum == reg['sha256']
            report['windows'].append({'path': reg['path'], 'position_sha256': expected[name],
                                      'npz_sha256': checksum, 'frozen_npz_sha256': reg['sha256']})
        report['sources'][source] = {'archive': filename, 'archive_sha256': provenance[filename]['sha256'],
                                     'exact_position_arrays': count, 'byte_identical_npz': byte_matches}
        print(f'{source}: matched {count} position arrays; {byte_matches} exact NPZ files.', flush=True)
    if seen != set(registered):
        raise ValueError(f'Missing registered windows: {sorted(set(registered) - seen)}')
    with tempfile.NamedTemporaryFile(dir=args.output, mode='w', delete=False, encoding='utf-8') as tmp:
        json.dump(report, tmp, indent=2)
        tmp.write('\n')
        complete = Path(tmp.name)
    os.replace(complete, args.output / 'preprocessing-report.json')


if __name__ == '__main__':
    main()
