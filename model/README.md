# Frozen pigeon-trajectory model

The selected checkpoint is `paper-public-association-v17/models/joint_full-11.pt`. The model definition is in `paper-public-association-v17/kinematic_mixture.py`. The other modules contain the feature encoders and mixture components used by that implementation. Source files and checkpoint retain the names and bytes recorded in the external evaluation registry. Each can be checked against the corresponding entry in `../data/frozen-registry.json`.

The score reconstruction uses the saved link decisions and runs independently of this directory. The original development environment used Python 3.12.8, PyTorch 2.6.0, NumPy 1.26.4 and SciPy 1.14.1. The training driver is included as a record of the optimization procedure. It references the original prepared coordinate windows and development-data adapter, which are external to this release. The checkpoint and source describe the model used with the recorded pigeon trajectories.
