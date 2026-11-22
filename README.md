# Identity association across gaps in recorded pigeon flight

This repository accompanies *Retaining individual identities across gaps in group flight trajectories*, by Simiao Fei, Yue Wang, Xueer Hao and Huicun Li. It reconstructs the external evaluation and the development-set threshold comparison from the archived decisions and recording summaries. [Read in Chinese](README.zh-CN.md).

## Reproduce the results

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

The saved external records contain the local reference labels and every method's assignment. The scorer checks one-to-one matching, reconstructs each metric and gives each recording equal weight within its source and gap. For development Table S9, paired record means and condition counts reproduce the published aggregation. These inputs support score reconstruction. The [model directory](model/) contains the frozen checkpoint and the paper’s model code in the original layout. Starting from raw trajectories to train or rerun the original matchers additionally requires the coordinate preprocessing, experimental implementations and their dependencies. Access links and digests for the original external trajectory releases are in [data/source-provenance.json](data/source-provenance.json).

[Evidence and units](docs/evidence.md) · [Reference tables](reference/) · [Citation](CITATION.cff) · [Reuse terms](NOTICE.md)

[Data access, splits and protocols](docs/data-access.md)
