#!/bin/bash

# Set parameters
benchmarks_dir="data/benchmarks"
script_path="./run.py"
grammar_style="wcfg_equal_p"
timeout=$((60 * 60))  # Timeout for each benchmark in seconds (1 hour)
enumerator="top_down"
check_llm="false"  # Whether to use check_llm (true/false)

# Count the number of .c benchmark files
benchmarks=($(ls ${benchmarks_dir}/*.c))
num_benchmarks=${#benchmarks[@]}
echo "Number of benchmarks: ${num_benchmarks}"
echo "Grammar Style: ${grammar_style}"
echo "Timeout: $(($timeout / 60)) minutes"
echo "Enumerator: ${enumerator}"
echo "Check LLM Solutions: ${check_llm}"

# Iterate over each benchmark
for benchmark_file in "${benchmarks[@]}"; do
    benchmark=$(basename "$benchmark_file" .c)
    
    # Run the Python script with the specified arguments
    python3 "$script_path" \
        --grammar_style "$grammar_style" \
        --benchmark "$benchmark" \
        --timeout "$timeout" \
        --enumerator "$enumerator" \
        --check_llm "$check_llm"

    # Check for any errors in the last command
    if [[ $? -ne 0 ]]; then
        echo "Error processing ${benchmark}"
    fi
done

echo "All benchmarks completed."
