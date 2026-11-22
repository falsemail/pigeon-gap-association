"""Compare newly inferred pigeon assignments with the paper's frozen predictions."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from scoring.external import METRICS, aggregate, emit_external_tables, score
from scoring.attention import compare_records, write_contrasts

ROOT = Path(__file__).resolve().parent
CASE_FIELDS = ("registry", "window_index", "condition_index", "source", "record",
               "gap", "seed", "sigma", "turnover", "clutter", "truth_a", "truth_b")


def compare(predictions, output, selected=None):
    registry_bytes = (ROOT / "data/frozen-registry.json").read_bytes()
    registry = json.loads(registry_bytes)
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    methods = list(registry["methods"] if selected is None else selected)
    if len(set(methods)) != len(methods) or not set(methods) <= set(registry["methods"]):
        raise ValueError("Duplicate or unregistered method")
    counts = {name: Counter() for name in methods}
    per_record = defaultdict(list)
    cases = set()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "decision-differences.csv").open("w", encoding="utf-8", newline="") as stream:
        differences = csv.writer(stream)
        differences.writerow(["source", "record", "window", "condition", "method",
                             "changed_predecessors", "archived_assignment", "new_assignment",
                             "archived_fallbacks", "new_fallbacks"])
        for archive in sorted((ROOT / "data/decisions").glob("*/*.jsonl")):
            with archive.open(encoding="utf-8") as source:
                for line in source:
                    expected = json.loads(line)
                    wi, ci = expected["window_index"], expected["condition_index"]
                    filename = f"w{wi:03d}-c{ci:02d}.json"
                    if filename in cases or expected["registry"] != registry_hash:
                        raise ValueError(f"Foreign or repeated archived condition: {filename}")
                    cases.add(filename)
                    actual = json.loads((predictions / filename).read_bytes())
                    for field in CASE_FIELDS:
                        if actual[field] != expected[field]:
                            raise ValueError(f"{filename}: different {field}; inputs cannot be paired")
                    if set(actual["outcomes"]) != set(methods):
                        raise ValueError(f"{filename}: method list differs from the requested set")
                    for name in methods:
                        old, new = expected["outcomes"][name], actual["outcomes"][name]
                        a, b = old["prediction"], new["prediction"]
                        values = score(actual["truth_a"], actual["truth_b"], b)
                        if any(not math.isclose(v, new["metrics"][m], rel_tol=0, abs_tol=1e-12)
                               for m, v in zip(METRICS, values)):
                            raise ValueError(f"{filename}, {name}: stored score differs from new prediction")
                        diff = sum(i != j for i, j in zip(a, b))
                        different_fallback = old["fallbacks"] != new["fallbacks"]
                        if diff or different_fallback:
                            differences.writerow([actual["source"], actual["record"], wi, ci, name,
                                                  diff, json.dumps(a), json.dumps(b),
                                                  old["fallbacks"], new["fallbacks"]])
                        counts[name].update(cases=1, changed_cases=int(diff > 0),
                                            changed_predecessors=diff,
                                            different_fallback_counts=int(different_fallback))
                        per_record[actual["source"], actual["gap"], name, actual["record"]].append(values)
    extras = {p.name for p in predictions.glob("w*-c*.json")} - cases
    ngrid = math.prod(len(registry["conditions"][k]) for k in
                      ("gap_seconds", "added_sigma_metres", "turnover_fraction", "clutter_fraction"))
    if extras or len(cases) != len(registry["windows"]) * ngrid:
        raise ValueError("Unexpected cases or an incomplete condition grid")

    records = {k: (len(rows), tuple(mean(col) for col in zip(*rows))) for k, rows in per_record.items()}
    references = {}
    with (ROOT / "reference/external-results.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            references[row["source"], int(row["gap_s"]), row["method"]] = row
    maxdiff = dict.fromkeys(METRICS, 0.0)
    with (output / "paper-comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "gap_s", "method", "records", "metric",
                         "paper", "regenerated", "regenerated_minus_paper"])
        for (src, gap, method), ref in references.items():
            if method not in methods:
                continue
            subset = [v for k, v in records.items() if k[:3] == (src, gap, method)]
            if int(ref["records"]) != len(subset) or int(ref["conditions"]) != sum(n for n, _ in subset):
                raise ValueError(f"Different recording coverage for {src}/{gap}/{method}")
            for metric, values in zip(METRICS, zip(*(r for _, r in subset))):
                result = mean(values)
                delta = result - float(ref[metric])
                writer.writerow([src, gap, method, len(subset), metric, ref[metric], result, delta])
                maxdiff[metric] = max(maxdiff[metric], abs(delta))
    all_methods = set(methods) == set(registry["methods"])
    if all_methods:
        rows, paired = aggregate(registry, records)
        emit_external_tables(registry, rows, paired, output)
        write_contrasts(output / "20s-attention-paired.csv", compare_records(registry, records))
    report = dict(registry_sha256=registry_hash, windows=len(registry["windows"]),
                  conditions=len(cases), all_methods=all_methods, per_method=counts,
                  max_absolute_metric_difference=maxdiff,
                  identical_assignments=all(c["changed_cases"] == 0 for c in counts.values()),
                  identical_fallbacks=all(c["different_fallback_counts"] == 0 for c in counts.values()))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for audit results")
    parser.add_argument("--methods", nargs="+", help="Audit an explicitly named subset only")
    opt = parser.parse_args()
    result = compare(opt.predictions, opt.output, opt.methods)
    print(json.dumps(result, indent=2))
    if not (result["identical_assignments"] and result["identical_fallbacks"]
            and all(v <= 1e-12 for v in result["max_absolute_metric_difference"].values())):
        raise SystemExit(1)
