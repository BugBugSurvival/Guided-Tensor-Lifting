import argparse
import copy
import json
import os
import subprocess
from create_driver import gen_driver


def create_io_file(benchmark_path):
    benchmark = os.path.basename(benchmark_path)[:-2]
    benchmark_dir = os.path.dirname(benchmark_path)

    io_pairs = []
    io_dict = {"input": {}, "output": {}}
    try:
        with open(os.path.join(benchmark_dir, f"{benchmark}_iotemp"), "r") as f:
            for line in f:
                l = line.rstrip()

                if l.startswith("sample"):
                    io_pairs.append(copy.deepcopy(io_dict))
                    io_dict = {"input": {}, "output": {}}
                    continue

                if l in ("input", "output"):
                    key = l
                    continue

                parts = l.split(":")
                var, dim = parts[0].strip(), parts[1].strip()
                values = ":".join(parts[2:]).strip()

                io_dict[key][var] = (dim, values)

        os.remove(os.path.join(benchmark_dir, f"{benchmark}_iotemp"))
    except FileNotFoundError as e:
        print("Could not open temporary io file")
        raise e

    with open(os.path.join(benchmark_dir, f"{benchmark}_io.json"), "w") as out:
        json.dump(io_pairs, out, indent=2)


def run_benchmark(benchmark_path, n_inputs):
    benchmark_dir = os.path.dirname(benchmark_path)
    benchmark = os.path.basename(benchmark_path)[:-2]
    driver_path = os.path.join(benchmark_dir, f"main_{benchmark}.c")

    try:
        subprocess.run(
            ["gcc", "-O3", driver_path, benchmark_path, "io_gen.c", "-o", os.path.join(benchmark_dir, f"{benchmark}.out")],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("Could not compile benchmark")
        raise e

    try:
        subprocess.run(
            [os.path.join(benchmark_dir, f"{benchmark}.out"), str(n_inputs)],
            check=True,
            stdout=open(os.path.join(benchmark_dir, f"{benchmark}_iotemp"), "w"),
        )
        os.remove(os.path.join(benchmark_dir, f"{benchmark}.out"))
    except subprocess.CalledProcessError as e:
        print("Could not run benchmark")
        raise e


def io_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", required=True, help="Path to the program to be lifted")
    parser.add_argument("-vp", "--valueprofile", required=True, help="Path to value profile file")
    parser.add_argument("-n", "--ninputs", type=int, required=True, help="Number of IO samples to be generated")
    parser.add_argument(
        "-ov",
        "--outvar",
        help="Variable that corresponds to the output. It does not need to be given in case the function is returning the output",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.benchmark):
        raise FileNotFoundError(f"Could not find {args.benchmark}")
    if not os.path.isfile(args.valueprofile):
        raise FileNotFoundError(f"Could not find {args.valueprofile}")
    if args.ninputs <= 0:
        raise ValueError("Please provide a valid (>0) number of desired IO samples")

    benchmark_path = os.path.abspath(args.benchmark)
    value_profile_file = os.path.abspath(args.valueprofile)

    print("-" * 7, f"Generating driver for {benchmark_path}", "-" * 7)
    gen_driver(benchmark_path, args.outvar, value_profile_file)

    print("-" * 7, f"Compiling and running {benchmark_path}", "-" * 7)
    run_benchmark(benchmark_path, args.ninputs)

    print("-" * 7, f"Generating IO file for {benchmark_path}", "-" * 7)
    create_io_file(benchmark_path)


if __name__ == "__main__":
    io_gen()
