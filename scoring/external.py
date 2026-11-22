"""Reconstruct the external results from local tracklet labels and saved assignments."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from functools import reduce
from operator import add
from pathlib import Path
import random
from statistics import mean

HERE = Path(__file__).resolve().parents[1]
EVIDENCE = HERE / "data"
METRICS = ("retention", "precision", "f1", "union_count_error")
SOURCES = ("Huang", "D1", "D2")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_manifest():
    for line in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        check(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest,
              f"Integrity check failed for {name}")


def score(a, b, prediction):
    check(len(prediction) == len(a), "Prediction has the wrong number of predecessors")
    check(all(isinstance(j, int) and -1 <= j < len(b) for j in prediction),
          "Successor index outside its window")
    links = [(i, j) for i, j in enumerate(prediction) if j >= 0]
    check(len({j for _, j in links}) == len(links), "Duplicated successor")
    ids_a = {i for i in a if i >= 0}
    ids_b = {i for i in b if i >= 0}
    correct = sum(a[i] >= 0 and a[i] == b[j] for i, j in links)
    continuing = len(ids_a & ids_b)
    accepted = len(links)
    count_error = abs((len(a) + len(b) - accepted) - len(ids_a | ids_b))
    return (correct / max(1, continuing), correct / max(1, accepted),
            2 * correct / max(1, continuing + accepted), count_error)


def recording_scores(registry, digest):
    cfg = registry["conditions"]
    grid = list(itertools.product(cfg["gap_seconds"], cfg["added_sigma_metres"],
                                  cfg["turnover_fraction"], cfg["clutter_fraction"]))
    values = defaultdict(list)
    fallbacks = defaultdict(int)
    seen = set()
    for file in sorted((EVIDENCE / "decisions").glob("*/*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            check((file.parent.name, file.stem) == (case["source"], case["record"]),
                  f"Misfiled decision in {file.name}")
            wi, ci = case["window_index"], case["condition_index"]
            check(0 <= wi < len(registry["windows"]) and 0 <= ci < len(grid),
                  "Unknown window or condition")
            check((wi, ci) not in seen and case["registry"] == digest, "Duplicate or foreign case")
            seen.add((wi, ci))
            window = registry["windows"][wi]
            check((case["source"], case["record"]) == (window["source"], window["record"]),
                  "Source or recording mismatch")
            check(tuple(case[k] for k in ("gap", "sigma", "turnover", "clutter")) == grid[ci]
                  and case["seed"] == cfg["random_seed_base"] + 100 * wi + ci,
                  "Condition or seed mismatch")
            check(set(case["outcomes"]) == set(registry["methods"]), "Incomplete method set")
            for name, outcome in case["outcomes"].items():
                scored = score(case["truth_a"], case["truth_b"], outcome["prediction"])
                check(all(abs(value - outcome["metrics"][field]) < 1e-12
                          for field, value in zip(METRICS, scored)),
                      f"Archived score disagrees at window {wi}, condition {ci}, method {name}")
                values[(case["source"], case["gap"], name, case["record"])].append(scored)
                fallbacks[name] += outcome["fallbacks"]
    expected = len(registry["windows"]) * len(grid)
    check(len(seen) == expected, "Missing registered cases")
    records = {group: (len(cases), tuple(mean(column) for column in zip(*cases)))
               for group, cases in values.items()}
    return records, fallbacks, expected


def aggregate(registry, scores):
    rows = []
    paired = []
    for source in SOURCES:
        for gap in registry["conditions"]["gap_seconds"]:
            record_ids = sorted({record for s, g, _, record in scores if (s, g) == (source, gap)})
            group = {}
            for method in registry["methods"]:
                records = [scores[source, gap, method, record] for record in record_ids]
                avg = [mean(column) for column in zip(*(values for _, values in records))]
                group[method] = dict(source=source, gap_s=gap, method=method, records=len(records),
                                     conditions=sum(n for n, _ in records), **dict(zip(METRICS, avg)))
            candidate = registry["candidate"]
            for row in group.values():
                row["candidate_minus_method_pp"] = 100 * (group[candidate]["retention"] - row["retention"])
                rows.append(row)
            ci = registry["descriptive_intervals"]
            for method in ci["comparators"]:
                diffs = [100 * (scores[source, gap, candidate, rec][1][0] -
                                scores[source, gap, method, rec][1][0]) for rec in record_ids]
                rng = random.Random(ci["seed"])
                n = len(diffs)
                # The archived intervals summed each bootstrap draw from left to right.
                bootstrap = sorted(reduce(add, rng.choices(diffs, k=n), 0.0) / n
                                   for _ in range(ci["draws"]))
                # Order statistics used in the frozen script, with zero-based indexing.
                paired.append(dict(source=source, gap_s=gap, baseline=method, records=n,
                                   delta_pp=mean(diffs), low_pp=bootstrap[249],
                                   high_pp=bootstrap[9749], positive_records=sum(d > 0 for d in diffs)))
    return rows, paired


def check_reference(rebuilt, filename, labels):
    reference = read_csv(HERE / "reference" / filename)
    check(len(rebuilt) == len(reference), f"Row count mismatch in {filename}")
    for row, expected in zip(rebuilt, reference):
        check(set(row) == set(expected), f"Column mismatch in {filename}")
        for key, value in row.items():
            equal = str(value) == expected[key] if key in labels else math.isclose(
                value, float(expected[key]), abs_tol=1e-12, rel_tol=0)
            check(equal, f"{filename} mismatch in {key}: {row}")


def main(output):
    verify_manifest()
    regfile = EVIDENCE / "frozen-registry.json"
    registry = json.loads(regfile.read_text(encoding="utf-8"))
    scores, fallbacks, ncases = recording_scores(registry, hashlib.sha256(regfile.read_bytes()).hexdigest())
    rows, paired = aggregate(registry, scores)
    emit_external_tables(registry, rows, paired, output, verify=True)
    print(f"Passed: {ncases} cases, {len(rows)} summary rows, {len(paired)} paired intervals; "
          f"{fallbacks['bcpd-author']} registration fallbacks. Outputs: {output.resolve()}")


def emit_external_tables(registry, rows, paired, output, verify=False):
    if verify:
        check_reference(rows, "external-results.csv", {"source", "method"})
        check_reference(paired, "paired-intervals.csv", {"source", "baseline"})
    names = {row["configuration"]: row["display_name"] for row in read_csv(EVIDENCE / "frozen-methods.csv")}
    check(list(names) == list(registry["methods"]), "Unexpected method order")
    by_key = {(r["method"], r["source"], r["gap_s"]): r for r in rows}
    table3 = []
    for method, name in names.items():
        entry = {"Method": name}
        for s in SOURCES:
            for g in registry["conditions"]["gap_seconds"]:
                entry[f"{s}/{g}"] = f"{100 * by_key[method, s, g]['retention']:.2f}"
        table3.append(entry)
    table4 = []
    for s in SOURCES:
        for method in ("classic-relative_history", registry["candidate"], "superglue-11"):
            row = by_key[method, s, 20]
            table4.append({"Source": s, "Method": names[method],
                           "Retention (%)": f"{100 * row['retention']:.2f}",
                           "Precision (%)": f"{100 * row['precision']:.2f}",
                           "Count MAE": f"{row['union_count_error']:.2f}"})
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in (("table3-retention.csv", table3), ("table4-selection.csv", table4),
                              ("external-results.csv", rows), ("paired-intervals.csv", paired)):
        write_csv(output / filename, content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rebuilt"),
                        help="Directory for the reconstructed CSV files (default: ./rebuilt)")
    main(parser.parse_args().output)
