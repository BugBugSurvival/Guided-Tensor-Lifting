# Instructions

The script `run_cpu.sh` runs all the benchmarks, first the original version then the taco-CPU version. `run_gpu.sh` only runs the taco-GPU version.
To run the scripts, do the following:

```
./run_cpu.sh <list-of-benchmarks> <n> <output-file>
```

The first argument is a list with the name of all benchmarks to run. An example is the file benchmark_list. The second is the number of times to run each benchmark. The third one is the output file. The script will write in the output file the average running of the benchmarks. `run_cpu` will write the time for both original and taco-CPU version and `run_gpu` will write the time for the taco-GPU version. 

Example:

```
./run_cpu.sh benchmark_list 10 output_cpu
```

