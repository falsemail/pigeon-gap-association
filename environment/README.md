# Inference environment

The registered run used Linux x86-64, Python 3.12.8, NumPy 1.26.4, SciPy 1.14.1, pandas 2.2.3, scikit-learn 1.8.0, and PyTorch 2.6.0 with CUDA 12.4 on an NVIDIA RTX 4090. The registry selects CPU or GPU for each configuration. For the same execution path, use a CUDA-capable Linux system, an NVIDIA driver compatible with CUDA 12.4, and the registered binary below. The CPU mode uses the saved weights on CPU. On another architecture or math library, compare the generated pairs and the registration fallback flags against the recorded decisions before interpreting any score differences.

From the repository root, with a Python 3.12 conda environment activated:

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r environment/requirements.txt
python inference/opgm_sources.py
python environment/prepare_bcpd.py --output "$PWD/build/bcpd"
export PIGEON_BCPD_BIN="$PWD/build/bcpd"
```

The first command installs the CUDA build for a compatible Linux machine. An existing environment that already has this build can proceed to the next command. Readers running CPU inference on another platform should install the PyTorch 2.6.0 wheel for their platform. `requirements.txt` gives the Python versions for inference. The core table scorer runs separately with the Python standard library. The inference script uses one numerical thread per worker. The original runtime used pygmtools 0.6.0 for IPFP and PCA-GM, einops 0.8.1 for the coordinate transformer, and scikit-learn 1.8.0 for the persisted histogram boosting model. The OPGM download checks the four upstream modules against their registered SHA-256 values. Model checkpoints and the histogram boosting pickle are data for the frozen evaluation. The runner checks the registered hashes before loading them.

## Point-set registration executable

`bcpd/registered/bcpd-linux-x86_64` is the actual patched Linux AMD64 executable used for scoring. Its SHA-256 is `12f0d75b45ba5c46922291a420ed35db3fcb740bfe28cf339874c9b030cfdf42`. Its OpenMP runtime is provided by the compiler or OS (libgomp), and its LP64 OpenBLAS is the library shipped in the Linux x86-64 `opencv-python==4.10.0.84` wheel. The preparation script checks the binary hash, locates the wheel's bundled libraries in the current environment, and runs a small input/output smoke test. Keep the checkout and Python environment at the same paths while using the generated launcher. The actual benchmark applies the registered ten-second per-call limit and reports fallback flags.

The upstream BCPD source is Osamu Hirose's [commit 078ca6b](https://github.com/ohirose/bcpd/tree/078ca6b4983577e6a7790ab11b6218bca8d46088). The `bcpd/base/` and `bcpd/register/` trees include the C sources needed by the benchmark. `bcpd/bcpd-full-output.patch` records the paper's change to `register/corresp.c`: the output file includes every positive correspondence probability. This patch is already applied in the distributed source. The MIT license and citation are supplied alongside it.

To build a native registration executable on another platform:

```sh
python environment/prepare_bcpd.py --mode native --output "$PWD/build/bcpd"
```

The native Linux build requires a C compiler with OpenMP, LP64 BLAS and LAPACK. The macOS build uses clang, Apple Accelerate and the LLVM OpenMP runtime. For a conda-managed OpenMP installation, `$CONDA_PREFIX/include/omp.h` and `$CONDA_PREFIX/lib/libomp.dylib` must be available. Use `--prefix PATH` when these libraries are installed elsewhere.
