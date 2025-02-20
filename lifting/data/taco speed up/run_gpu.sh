#!/bin/bash

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

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH


echo "Dumping into $outf"
for benchmark in ${benchmarks[@]}; do
       	echo "Running $benchmark...."
	set -x
	nvcc -o ../$benchmark.run_gpu ../taco_gpu_drivers/full/main_${benchmark}_full_gpu.cu ../util.cu -I .. -lrt
	#Consistent caches
	data=$(eval "../$benchmark.run_gpu $times")
	echo "Ran $benchmark got $data"
	data=$(eval "../$benchmark.run_gpu $times")
        echo "$benchmark $data" | tee -a ../scripts/$outf
        #break
done
