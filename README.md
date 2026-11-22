# Identity association across gaps in recorded pigeon flight

This repository accompanies *Retaining individual identities across gaps in group flight trajectories*, by Simiao Fei, Yue Wang, Xueer Hao and Huicun Li. It provides both the frozen matchers for the external evaluation and the scoring code for the paper. [Read in Chinese](README.zh-CN.md).

## From the public trajectories to identity matches

First follow [the environment instructions](environment/README.md) to install the registered dependencies, prepare the point-registration executable, and fetch the hash-pinned graph matcher modules. In this environment, from the repository root, run:

```sh
python infer_external.py --archives downloads --work work/frozen-run --download --workers 4
```

This downloads the three original recording archives, rebuilds 159 windows, and generates assignments for all 14 methods under 5,724 conditions. The nine learned configurations use the frozen weights. The matchers only receive the two visible sets of tracklets. After inference, the audit pairs every regenerated assignment with its archived counterpart and checks the registration fallback flags. See `work/frozen-run/comparison/report.json` for the outcome, `decision-differences.csv` for individual discrepancies, and `paper-comparison.csv` for the 84 method/source/gap comparisons across four metrics. The same directory contains the directly regenerated external tables and paired intervals. Use `--workers 1` for a sequential run, or omit `--download` when the original archives are already present.

The default device choice follows the registry. For an explicit CPU portability test, pass `--device cpu`; numerical differences, notably in point-set registration, can change individual assignments. Use the original Linux executable and registered GPU environment to check the paper's frozen results. [Preprocessing](rawprep/README.md) · [Inference details and diagnostic options](inference/README.md)

## Recalculate the tables from archived decisions

From the repository root, run the following command with Python 3.8 or later. The scripts use the standard library and run offline on a CPU.

```bash
python3 -I -S -B reproduce.py
```

On Windows use `py -3 -I -S -B reproduce.py`. Use `--output PATH` for a different destination. A successful run checks the input SHA-256 values, all 5,724 registered conditions and the per-condition scores for 14 methods. It verifies every output against its published reference within an absolute numerical tolerance of `1e-12`.

| Output under `results/` | Contents |
|---|---|
| `table3-retention.csv` | External identity retention for all methods, as percentages |
| `table4-selection.csv` | Retention, precision and count error for three methods after 20 s |
| `external-results.csv` | 84 full-precision rows for three sources, two gap lengths and 14 methods |
| `paired-intervals.csv` | 12 registered descriptive intervals from resampling paired recordings |
| `table-s9-calibration.csv` | Two thresholds, evaluated on the same 687 development conditions, with equal source weights |
| `20s-attention-paired.csv` | Three post hoc retention contrasts with the attention matcher after 20 s |

## What the reader can verify

The saved external records contain the local reference labels and every method's assignment. The scorer checks one-to-one matching, reconstructs each metric and gives each recording equal weight within its source and gap. For development Table S9, paired record means and condition counts reproduce the published aggregation. These inputs support score reconstruction. The [model directory](model/) contains the frozen checkpoint and the paper’s model code in the original layout. The raw-track inference entry point above uses these weights and the additional baseline checkpoints. Supplementary Table S9 is a separate development-set calibration comparison and is reconstructed from its paired record summaries. Access links and digests for the original external trajectory releases are in [data/source-provenance.json](data/source-provenance.json).

[Evidence and units](docs/evidence.md) · [Reference tables](reference/) · [Citation](CITATION.cff) · [Reuse terms](NOTICE.md)

[Data access, splits and protocols](docs/data-access.md)
