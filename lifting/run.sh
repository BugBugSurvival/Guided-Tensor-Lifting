#!/bin/bash

# Run each script in the background
./run_bu_equal.sh &
./run_bu_full.sh &
./run_bu_orig.sh &
./run_bu_wcfg.sh &
./run_td_equal.sh &
./run_td_full.sh &
./run_td_orig.sh &
./run_td_wcfg.sh &

# Wait for all background processes to finish
wait

echo "All scripts have completed."
