"""Prepare the registration executable used by the public pigeon benchmark."""
import argparse
import hashlib
from pathlib import Path
import os
import platform
import shlex
import subprocess
import sys
import sysconfig
import tempfile

SOURCE = Path(__file__).resolve().parent / "bcpd"
REGISTERED_SHA256 = "12f0d75b45ba5c46922291a420ed35db3fcb740bfe28cf339874c9b030cfdf42"
BLAS_SONAME = "libopenblas-r0-f650aae0.3.3.so"


def registered(output):
    binary = SOURCE / "registered" / "bcpd-linux-x86_64"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != REGISTERED_SHA256:
        raise ValueError("Unexpected registered BCPD executable hash")
    if not os.access(binary, os.X_OK):
        # Some ZIP extractors preserve bytes but discard the executable permission.
        binary.chmod(binary.stat().st_mode | 0o100)
    libraries = Path(sysconfig.get_path("purelib")) / "opencv_python.libs"
    if not (libraries / BLAS_SONAME).is_file():
        raise FileNotFoundError(f"{BLAS_SONAME} was not found in {libraries}. Install opencv-python==4.10.0.84.")
    # The original executable locates its LP64 OpenBLAS/Fortran dependencies via this directory.
    output.write_text(
        "#!/bin/sh\n"
        f"export LD_LIBRARY_PATH={shlex.quote(str(libraries))}\"${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\"\n"
        f"exec {shlex.quote(str(binary))} \"\u0024@\"\n"
    )
    output.chmod(0o755)


def native(output, prefix):
    sources = sorted((SOURCE / "register").glob("*.c")) + sorted((SOURCE / "base").glob("*.c"))
    if platform.system() == "Darwin":
        arguments = [os.environ.get("CC", "clang"), "-O3", "-Xpreprocessor", "-fopenmp", "-DUSE_OPENMP"]
        libraries = ["-lomp", "-framework", "Accelerate", "-lm"]
    else:
        arguments = [os.environ.get("CC", "gcc"), "-O3", "-fopenmp", "-DUSE_OPENMP"]
        libraries = ["-llapack", "-lblas", "-lm"]
    arguments += [f"-I{prefix / 'include'}", *map(str, sources), f"-L{prefix / 'lib'}", f"-Wl,-rpath,{prefix / 'lib'}"]
    subprocess.run([*arguments, *libraries, "-o", str(output)], check=True)


def smoke_test(binary):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        points = "0 0\n1 3\n2 -1\n4 2\n-3 1\n"
        (path / "x.txt").write_text(points)
        run = subprocess.run(
            [str(binary), "-x", "x.txt", "-y", "x.txt", "-Tn", "-b2", "-l2", "-w0.1",
             "-g1", "-ux", "-N1", "-n100", "-c1e-4", "-d1000000", "-sPce", "-ofit_", "-q"],
            cwd=path, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
            capture_output=True, text=True, timeout=20)
        if run.returncode or not (path / "fit_P.txt").is_file():
            raise RuntimeError(f"BCPD failed its basic input/output check. {run.stderr}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Path to the prepared executable or launcher")
    parser.add_argument("--mode", choices=("auto", "registered", "native"), default="auto")
    parser.add_argument("--prefix", type=Path, default=Path(sys.prefix), help="Native build prefix containing OpenMP, BLAS and LAPACK")
    args = parser.parse_args()
    linux_x64 = platform.system() == "Linux" and platform.machine() in ("AMD64", "x86_64")
    use_registered = args.mode == "registered" or (args.mode == "auto" and linux_x64)
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if use_registered and not linux_x64:
        parser.error("The registered binary requires Linux x86-64. Choose --mode native on this platform.")
    with tempfile.TemporaryDirectory(dir=target.parent) as directory:
        prepared = Path(directory) / "bcpd"
        if use_registered:
            registered(prepared)
        else:
            native(prepared, args.prefix)
        smoke_test(prepared)
        os.replace(prepared, target)
    print(f"Prepared and smoke-tested {'registered' if use_registered else 'native'} BCPD: {target}")


if __name__ == "__main__":
    main()
