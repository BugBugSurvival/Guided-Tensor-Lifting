#!/bin/bash

set -ex

if [[ $# -ne 3 ]]; then
	echo "Usage: $0 <benchmarks file> <num times to execute> <output file>"
	exit
fi

# load the benchmarks in
benchmarks=( $(cat $1) )
times=$2
outf=$3

rm -rf $outf
touch $outf

# runtime scheuling prarams for taco
export OMP_SCHEDULE="guided,4"
export OMP_NUM_THREADS=4

for benchmark in ${benchmarks[@]}; do
	gcc -O3 -fopenmp -I. ./util.c ./benchmarks/${benchmark}.c ./taco_cpu_drivers/main_${benchmark}_full_cpu.c -o ./benchmarks/$benchmark.run
	
	echo "Built $benchmark"
	# Consistent caches
	data=$(eval ./benchmarks/$benchmark.run $times)
	echo "Initial run data is $data"
	data=$(eval ./benchmarks/$benchmark.run $times)
	echo "$benchmark $data" >> $outf
done
