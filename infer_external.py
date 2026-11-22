"""Rebuild the registered pigeon windows, run frozen inference, and audit the assignments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="A new directory for this run")
    parser.add_argument("--archives", type=Path, required=True, help="Directory for the original ZIPs")
    parser.add_argument("--download", action="store_true", help="Fetch missing ZIPs directly from the providers")
    parser.add_argument("--workers", type=int, default=1, help="Parallel inference processes sharing the device")
    parser.add_argument("--device", choices=("registered", "cpu"), default="registered")
    args = parser.parse_args()
    if not 1 <= args.workers <= 159:
        parser.error("--workers should be between 1 and the 159 registered windows")
    if not os.environ.get("PIGEON_BCPD_BIN"):
        parser.error("Set PIGEON_BCPD_BIN as described in environment/README.md")
    work = args.work.resolve()
    if work == ROOT or work.exists():
        parser.error("Use a new working directory; previous outputs are preserved")
    work.mkdir(parents=True)
    windows, decisions = work / "windows", work / "decisions"
    run("rawprep/prepare.py", "--archives", args.archives.resolve(), "--output", windows,
        *(["--download"] if args.download else []))

    def infer_shard(i):
        run("inference/run.py", "--windows", windows, "--output", decisions,
            "--device", args.device, "--shards", args.workers, "--shard", i)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(infer_shard, range(args.workers)))
    run("compare_predictions.py", "--predictions", decisions, "--output", work / "comparison")


if __name__ == "__main__":
    main()
