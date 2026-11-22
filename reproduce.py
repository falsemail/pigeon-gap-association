"""Rebuild and verify the published scores using the Python standard library."""
import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def verify_table(path):
    expected_file = ROOT / "reference" / path.name
    with path.open(newline="", encoding="utf-8") as left, expected_file.open(newline="", encoding="utf-8") as right:
        actual, expected = list(csv.reader(left)), list(csv.reader(right))
    if not expected or not actual or actual[0] != expected[0] or len(actual) != len(expected):
        raise ValueError(f"Column names or row count differ in {path.name}")
    for num, (a, b) in enumerate(zip(actual[1:], expected[1:]), start=2):
        if len(a) != len(b):
            raise ValueError(f"Different number of fields in {path.name} row {num}")
        for v, w in zip(a, b):
            if v == w:
                continue
            try:
                equal = math.isclose(float(v), float(w), rel_tol=0, abs_tol=1e-12)
            except ValueError:
                equal = False
            if not equal:
                raise ValueError(f"Difference in {path.name}, row {num}: {v} vs {w}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    output = parser.parse_args().output.resolve()
    if output == ROOT or any(p in (output, *output.parents) for p in (ROOT / "reference", ROOT / "data", ROOT / "scoring")):
        parser.error("Choose a separate output directory, e.g. results/")
    output.mkdir(exist_ok=True, parents=True)
    python = [sys.executable, "-I", "-S", "-B"]
    jobs = (("external.py", "--output", str(output)),
            ("calibration.py", "--output", str(output / "table-s9-calibration.csv")),
            ("attention.py", str(output / "20s-attention-paired.csv")))
    for script, *args in jobs:
        subprocess.run([*python, str(ROOT / "scoring" / script), *args], check=True)
    for reference in sorted((ROOT / "reference").glob("*.csv")):
        verify_table(output / reference.name)
    print(f"Verified all six output tables against reference/ in {output}")

if __name__ == "__main__":
    main()
