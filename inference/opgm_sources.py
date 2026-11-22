"""Fetch and verify the exact four OPGM author modules used in this comparison."""
import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

COMMIT = "39b5344c7a31c3a764f91be754932c7160a8bcfa"
BASE = f"https://raw.githubusercontent.com/Gathika94/OPGMrs/{COMMIT}/"
FILES = {
    "src/pgm_solver/pma.py": "38f392edaa96117e71c2d23318ecd6aef160a19899136cc6ca84f6bd0751de9a",
    "src/pgm_solver/pmatnew.py": "c5d2d76241ab9229f96f62fb25201f796cfae880e69679148688c019a4648345",
    "src/loss_func.py": "cecc26ad1c57449b73f4de01fe611461f01327ae1951e869f17f9e8d29dda92c",
    "src/lap_solvers/sinkhorn.py": "0b8a210bdf037d799ac3b84d03ffcc613e52275f1eb0a23d876d014dae6887dc",
}
DEFAULT_DIR = Path(os.environ.get("PIGEON_OPGM_DIR", Path.home()/".cache"/"pigeon-gap-association"/"opgm"))

def checked_source(folder, name):
    path = folder / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != FILES[name]:
        raise ValueError(f"Missing or incorrect OPGM source {path}. Run python inference/opgm_sources.py.")
    return path

def fetch(folder):
    for name, expected in FILES.items():
        path = folder / name
        if path.exists():
            checked_source(folder, name)
            continue
        with urlopen(BASE + name, timeout=90) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Incorrect upstream file digest: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temp.write(content)
        os.replace(temp.name, path)
    print("Verified pinned OPGM modules in", folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    fetch(parser.parse_args().output)
