
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from create_driver import gen_driver


def _to_path(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _create_io_json(tmp_file: Path, io_json: Path):
    """Parse the `*_iotemp` printout into structured JSON."""
    io_pairs = []
    cur: dict[str, dict[str, tuple[str, str]]] = {"input": {}, "output": {}}
    section: str | None = None

    with tmp_file.open() as fp:
        for raw in fp:
            line = raw.rstrip()
            if line == "input":
                section = "input"
                continue
            if line == "output":
                section = "output"
                continue
            if line.startswith("sample_id"):
                io_pairs.append(cur)
                cur = {"input": {}, "output": {}}
                section = None
                continue

            # var: dim: values (values can legally contain colons, so split max 2)
            var, dim, values = line.split(":", 2)
            cur[section][var.strip()] = (dim.strip(), values.strip())

    with io_json.open("w") as fp:
        json.dump(io_pairs, fp, indent=2)

    tmp_file.unlink(missing_ok=True)



def run_pipeline(benchmark: Path, vprofile: Path, n: int, outvar: str | None):
    bench_dir = benchmark.parent
    bench_name = benchmark.stem

    # 1. generate driver
    gen_driver(str(benchmark), outvar, str(vprofile))

    driver = bench_dir / f"main_{bench_name}.c"
    exe    = bench_dir / f"{bench_name}.out"
    tmp_io = bench_dir / f"{bench_name}_iotemp"
    io_json= bench_dir / f"{bench_name}_io.json"

    # 2. compile
    subprocess.run([
        "gcc", "-O3", driver, benchmark, "io_gen.c", "-o", exe
    ], check=True)

    # 3. run
    with tmp_io.open("w") as fout:
        subprocess.run([exe, str(n)], stdout=fout, check=True)

    exe.unlink(missing_ok=True)

    # 4. parse to JSON
    _create_io_json(tmp_io, io_json)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--benchmark", required=True, help="kernel .c path")
    ap.add_argument("-vp", "--valueprofile", required=True, help="value‑profile JSON")
    ap.add_argument("-n", "--ninputs", type=int, required=True, help="#I/O pairs")
    ap.add_argument("-ov", "--outvar", help="name of output variable (void kernels)")
    args = ap.parse_args()

    run_pipeline(_to_path(args.benchmark), _to_path(args.valueprofile), args.ninputs, args.outvar)


if __name__ == "__main__":
    main()

