# Inference on the registered pigeon recordings

Run from the repository root after preparing the software and the three source archives. The registry defines the order of the 159 windows, 36 conditions per window, the random seeds, the 14 configurations and the validation-selected threshold for each. Eight additional checkpoints are in `inference/weights/`. The candidate checkpoint is under `model/`. Checksums for all nine are checked against `data/frozen-registry.json` before deserialization.

```bash
python inference/opgm_sources.py
python inference/run.py --windows work/windows --output work/assignments --device registered
python compare_predictions.py --predictions work/assignments --output work/comparison
```

`registered` uses the original device assignment (CPU for classical, registration and tree methods, CUDA for neural matchers). On a CPU-only machine, use `--device cpu`. Floating-point libraries, device kernels and registration convergence can affect individual decisions at a threshold. Compare the resulting decisions with the archive on each platform. Runtime measured on a different device does not reproduce the timing experiment.

For a quick check, append `--window 0 --condition 0` and choose a new output directory. To restrict a diagnostic run to a single method, pass `--method joint_full-11`; repeat the argument to select additional methods. Multiple worker processes can use `--shards K --shard I` with distinct indices from 0 through K–1. Every window runs on exactly one worker. Each condition writes its own JSON file atomically. Existing output files are protected against accidental replacement. Use separate directories for different runs or device modes.

## What goes into a prediction

The runner checks the SHA-256 of the input position tensor, adds the registered noise, omissions, clutter and set turnover, and constructs the two visible short tracklets. It strips the reference identities, replaces the local tracklet keys and supplies the observed positions, histories and geometric features to each matcher. The model returns one recovered tracklet index for each predecessor, or −1 if no match is accepted. Only after inference does the scorer use the reference identities to calculate the metrics. `inference/run.py` reads no saved predictions. `compare_predictions.py` performs the separate comparison with the archive.

The inference port reuses the original measured-motion feature functions and decoders. `implementation-provenance.json` identifies their frozen source file digests and extracted symbols. `configurations.json` contains the selected settings read from the registered validation artifacts. The CPU OPGM path computes the original motion-backbone affinity, the author's Sinkhorn probabilities and the author's assignment. It omits three outputs used only by the training loss; the original code allocates them on CUDA, and the inference decoder reads none of them. The registered CUDA path keeps that computation. Loading model weights and scoring conditions never fits a model or selects a threshold.

## Third-party components

* **TWiX.** The author attention implementation is included with its MIT license under `runtime/twix_vendor/`. Source commit `3cff9cce39fe77eca78c2a2662aa655cd3215e7c`. The paper uses the group-centred point adapter and its own frozen checkpoint.
* **OPGM-rs.** `opgm_sources.py` downloads four files from Gathika94/OPGMrs, commit `39b5344c7a31c3a764f91be754932c7160a8bcfa`, and verifies each SHA-256. They are stored outside the repository in `~/.cache/pigeon-gap-association/opgm`. For an offline reproduction, copy an existing exact checkout into a private cache and set `PIGEON_OPGM_DIR` to its root. The adapter retains the source bodies and skips only the originally excluded optional debugger/solver imports. See the [upstream project and license](https://github.com/Gathika94/OPGMrs/tree/39b5344c7a31c3a764f91be754932c7160a8bcfa).
* **BCPD.** Use the executable supplied or built by the environment preparation instructions, and set `PIGEON_BCPD_BIN` to its absolute path. The frozen adapter invokes the author executable with the registered options, counts numerical failures and applies the registered fallback. The output includes one fallback count per method and condition.
* **PCA-GM and IPFP.** These use pygmtools 0.6.0, with the PyTorch and NumPy backends respectively. The project motion-node PCA-GM checkpoint is loaded explicitly; no image weights are downloaded.
