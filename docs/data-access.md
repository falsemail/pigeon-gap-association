# Data and protocol for the pigeon-recording benchmark

The paper compares identity assignments across artificial gaps in previously published pigeon trajectories. The frozen comparison starts with short, pre-existing coordinate tracks. The upstream studies supply the named recordings below.

| Role in this paper | Data release and file selection |
|---|---|
| Development — Watts 2015 | [Dryad, 10.5061/dryad.f9n8t](https://doi.org/10.5061/dryad.f9n8t). The study archive identifies the [Zenodo mirror](https://zenodo.org/records/5027145), ZIP MD5 `ba5ca180805742d0ed1bbac12abc780d`. |
| Development — Watts 2016 | [Dryad, 10.5061/dryad.508j2](https://doi.org/10.5061/dryad.508j2). The study archive identifies the [Zenodo mirror](https://zenodo.org/records/4943714), ZIP MD5 `ab0950ffebd924950632f727cd487c27`. |
| Development — planar escape flights | Papadopoulou et al., [Zenodo, 10.5281/zenodo.4993109](https://doi.org/10.5281/zenodo.4993109), version 1. Download **empirical.zip** (MD5 `1db3e28466a1382c4cceb962966be6bb`). The simulated archives do not enter the analysis. |
| External — Huang | Huang et al., [Zenodo, 10.5281/zenodo.21092705](https://doi.org/10.5281/zenodo.21092705), version 1.0.0. Download **pigeon_homing_public_trials_v1.0.0.zip** (MD5 `5c1e9b386bff5794e177b122b1fcb244`). Use the horizontal GPS positions from the three trials. |
| External — Coordinated D1 and D2 | The [ELTE data site](https://coordinatedflock.elte.hu/) distributes [trajectoriesD1.zip](https://coordinatedflock.elte.hu/data/trajectoriesD1.zip) and [trajectoriesD2.zip](https://coordinatedflock.elte.hu/data/trajectoriesD2.zip). Cite the provider's [shared methods paper](https://doi.org/10.1371/journal.pone.0140558) and the underlying [D1](https://doi.org/10.1038/nature08891) and [D2](https://doi.org/10.1073/pnas.1305552110) articles. Each series contributed 15 recordings. |

## Recorded splits and aggregation

Parameter fitting used 7,200 conditions, validation used 612 and exploratory development used 687. The three development sources were Watts 2015, Watts 2016 and the planar escape data. The escape source separated the 2018 flights by recording date. Its training dates were October 3, 4, 7, 8, 11, 12, 13 and 22 (22 recordings, 116 windows), validation dates were October 2, 5 and 19 (9 recordings, 54 windows), and development dates were October 9, 10, 16 and 18 (10 recordings, 53 windows). Validation used F1 averaged within recording groups, then within sources, then equally across sources. Table S9 compares two thresholds on the same development conditions with the same weights at both thresholds.

The external registry specifies the windows, condition grid, seeds, configuration identifiers and original input digests. Its SHA-256 is `d28b2e95905de43c9d1e99654148621ec91c5e4febb61a6164c0ced5589fe50f`. Cross-check 159 windows and 5,724 window–condition pairs covering all 14 methods. Within each source and gap, the paper averages first within recording and then equally across recordings. The 5,724 pairs share 33 recordings.

The following SHA-256 values identify the original author-archive artifacts. They can be used to check a copy obtained for an independent rerun.

| Author-archive artifact | SHA-256 |
|---|---|
| Development split registration (v14) | `0e5c9e3b214570613ed8b8a1eb40c3e74e313bfd09cd24e8f7f757c3bf05b2cb` |
| Escape visible-window manifest (v14) | `236692d627e12e57d57eac90515fcd81df671da2c8ec2ccd339315f9f095b012` |
| Selected model registration (v17, seed 11) | `8909b70973fc3c379b303142f72f67128bf80ca32ab96c460849d34401d95f49` |
| Selected checkpoint | `daca16121276805f9501ed8d2981a5e0f68b1888ab355b443d89303b1d9289ac` |
| Matched development input cache (687 conditions) | `69a0da53a8b3cb1d7510bcf326a74ef219a32a09d8db6c9e1e857c6780702378` |

## Reference software and settings

The recorded experiment environment used Python 3.12.8, NumPy 1.26.4, SciPy 1.14.1, PyTorch 2.6.0 and CUDA 12.4 on the server. The table-reconstruction scripts run with the Python standard library. The 14 registered configurations include classical assignments, point-set registration, graph methods, an adapted coordinate transformer and learned motion mixtures. The manuscript gives implementation-specific adaptations. Selected references include [Murty](https://doi.org/10.1287/opre.16.3.682), [Bayesian coherent point drift](https://doi.org/10.1109/TPAMI.2020.2971687), [integer projected fixed-point matching](https://proceedings.neurips.cc/paper_files/paper/2009/hash/fc2c7c47b918d0c2d792a719dfb602ef-Abstract.html), [SuperGlue](https://doi.org/10.1109/CVPR42600.2020.00499), [TWiX](https://doi.org/10.1016/j.patcog.2024.111169) and [optimal partial graph matching](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cfa8440d500a6a6867157dfd4eaff66e-Abstract-Conference.html).

## Run the external comparison

The [preprocessing script](../rawprep/README.md) reads the three original provider archives. The [inference entry point](../inference/README.md) uses the resulting coordinates, registered conditions, nine checkpoints and frozen thresholds. The command in the [main README](../README.md) runs all 14 configurations and compares the new assignments against the published evaluation records. The development-set threshold analysis uses the separately archived paired recording summaries.
