"""Post hoc 20 s retention contrast using the published recording bootstrap."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

# Isolated Python omits this directory from the import path.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from external import EVIDENCE, SOURCES, recording_scores, verify_manifest


def compare():
    verify_manifest()
    frozen = (EVIDENCE / "frozen-registry.json").read_bytes()
    registry = json.loads(frozen)
    records, _, _ = recording_scores(registry, hashlib.sha256(frozen).hexdigest())
    setup = registry["descriptive_intervals"]
    for source in SOURCES:
        ids = sorted(rec for s, gap, method, rec in records
                     if (s, gap, method) == (source, 20, registry["candidate"]))
        differences = [
            100 * (records[source, 20, registry["candidate"], rec][1][0]
                   - records[source, 20, "superglue-11", rec][1][0])
            for rec in ids
        ]
        rng = random.Random(setup["seed"])
        draws = sorted(mean(rng.choices(differences, k=len(ids))) for _ in range(setup["draws"]))
        yield (source, len(ids), mean(differences), draws[249], draws[9749],
               sum(d > 0 for d in differences), sum(d == 0 for d in differences))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source", "recordings", "mixture_minus_attention_pp",
                         "lower_pp", "upper_pp", "mixture_ahead", "ties"))
        writer.writerows(compare())
