"""Rebuild Supplementary Table S9 from paired development record means."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("retention", "precision", "f1", "union_count_error")
THRESHOLDS = (("registered", "-3"), ("calibrated", "-5.25"))
EXPECTED = {"15": 336, "20": 351}


def rebuild(evidence):
    with evidence.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({(r["source"], r["record_id"], r["gap_s"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate record and duration")
    totals = defaultdict(int)
    for r in rows:
        totals[r["gap_s"]] += int(r["conditions"])
    if dict(totals) != EXPECTED:
        raise ValueError(f"Unexpected development condition totals: {dict(totals)}")

    for gap in ("all", "15", "20"):
        records = defaultdict(list)
        for r in rows:
            if gap == "all" or r["gap_s"] == gap:
                records[r["source"], r["record_id"]].append(r)
        for name, threshold in THRESHOLDS:
            result = {"gap_s": gap, "threshold": threshold}
            for k in METRICS:
                sources = defaultdict(list)
                for (source, _), pairs in records.items():
                    # Pool both durations *within a record* using their case counts.
                    n = sum(int(r["conditions"]) for r in pairs)
                    value = sum(int(r["conditions"]) * float(r[f"{name}_{k}"])
                                for r in pairs) / n
                    sources[source].append(value)
                result[k] = mean(mean(values) for values in sources.values())
            yield result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rebuilt/table-s9-calibration.csv"))
    args = parser.parse_args()
    rows = list(rebuild(ROOT / "data" / "development-calibration-records.csv"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Table S9 rows to {args.output}. Retention, precision and F1 are proportions.")
