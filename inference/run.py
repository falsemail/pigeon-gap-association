"""Run frozen matchers on the 159 registered windows of pigeon GPS recordings."""
import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY_SHA256 = "d28b2e95905de43c9d1e99654148621ec91c5e4febb61a6164c0ced5589fe50f"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def infer(args):
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    import numpy as np
    import torch

    runtime = HERE / "runtime"
    sys.path.insert(0, str(runtime))
    for ver in (3, 6, 9, 10, 17):
        sys.path.append(str(args.repo / "model" / f"paper-public-association-v{ver}"))
    from methods import FrozenMatcher, Inputs, validate_file
    from core import score_assignment
    from data_protocol import project2d
    from episodes import episode
    from benchmark import observed

    torch.set_num_threads(1)
    registry_path = args.repo / "data" / "frozen-registry.json"
    validate_file(registry_path, REGISTRY_SHA256)
    reg = read(registry_path)
    config = read(HERE / "configurations.json")
    name_list = args.method or list(reg["methods"])
    if set(name_list) - reg["methods"].keys():
        raise ValueError(f"Registered methods: {', '.join(reg['methods'])}")
    matchers = {}
    for name in name_list:
        row = reg["methods"][name]
        device = row["device"] if args.device == "registered" else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("The registered device for this method is CUDA. Use --device cpu for the portable replay.")
        if config[name]["validation_result_sha256"] != reg["sha256"][row["spec"]["result"]]:
            raise ValueError(f"Configuration source does not match the frozen registry for {name}")
        matchers[name] = FrozenMatcher(name, row, config[name], args.repo, HERE, reg["sha256"], device)
    cfg = reg["conditions"]
    grid = list(itertools.product(cfg["gap_seconds"], cfg["added_sigma_metres"],
                                  cfg["turnover_fraction"], cfg["clutter_fraction"]))
    position_hashes = read(HERE.parent / "rawprep" / "position-sha256.json")
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for wi, win in enumerate(reg["windows"]):
            if wi % args.shards != args.shard or (args.window is not None and wi != args.window):
                continue
            with np.load(args.windows / win["path"], allow_pickle=False) as archive:
                trajectory = archive["truth"]
            positions = np.ascontiguousarray(trajectory[:, :, :3], dtype="<f4")
            if hashlib.sha256(positions.tobytes()).hexdigest() != position_hashes[win["window"]]:
                raise ValueError(f"Coordinate digest differs for {win['window']}")
            for ci, (gap, sigma, turn, clutter) in enumerate(grid):
                if args.condition is not None and ci != args.condition:
                    continue
                path = args.output / f"w{wi:03d}-c{ci:02d}.json"
                if path.exists():
                    raise FileExistsError(f"Output already exists: {path}. Use an empty output directory.")
                seed = cfg["random_seed_base"] + 100*wi + ci
                scenario = episode(seed, n=win["targets"], truth=trajectory, engine="external",
                                   kind="reserved", gap=gap, sigma=sigma, turnover=turn,
                                   clutter=clutter, onset=4., recovery=2.)
                if win["dimensions"] == 2:
                    scenario = project2d(scenario)
                visible = Inputs(observed(scenario))
                n, m = visible.sample["cost"].shape
                outcomes = {}
                for name, matcher in matchers.items():
                    pred, fallback = matcher.call(visible)
                    pred = np.asarray(pred, dtype=int)
                    positive = pred[pred >= 0]
                    if pred.shape != (n,) or np.any(pred < -1) or np.any(pred >= m) or len(set(positive)) != len(positive):
                        raise ValueError(f"Invalid assignment for {name} at {wi}/{ci}")
                    scores = score_assignment(scenario, pred)
                    outcomes[name] = dict(prediction=pred.tolist(), fallbacks=fallback,
                                          metrics={k: float(scores[k]) for k in ("retention", "precision", "f1", "union_count_error")})
                result = dict(registry=REGISTRY_SHA256, window_index=wi, condition_index=ci, source=win["source"],
                              record=win["record"], gap=gap, seed=seed, sigma=sigma, turnover=turn, clutter=clutter,
                              truth_a=scenario["truth_a"].tolist(), truth_b=scenario["truth_b"].tolist(), outcomes=outcomes)
                with tempfile.NamedTemporaryFile(dir=args.output, delete=False, mode="w") as tmp:
                    json.dump(result, tmp, separators=(",", ":"), allow_nan=False)
                os.replace(tmp.name, path)
            print(f"Scored window {wi:03d}: {win['source']} / {win['window']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parent, help=argparse.SUPPRESS)
    parser.add_argument("--windows", required=True, type=Path, help="Output directory of rawprep/prepare.py")
    parser.add_argument("--output", required=True, type=Path, help="Empty directory for per-condition assignments")
    parser.add_argument("--device", choices=("registered", "cpu"), default="registered")
    parser.add_argument("--window", type=int, choices=range(159), metavar="0..158")
    parser.add_argument("--condition", type=int, choices=range(36), metavar="0..35")
    parser.add_argument("--method", action="append", help="Restrict a diagnostic run to this registered configuration; repeatable")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must satisfy 0 <= shard < shards")
    infer(args)
