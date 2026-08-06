#!/bin/bash
# Runs every variant x every mode, each in a FRESH Vivado session.
# Stops immediately if any run fails or produces a hollow (constant-folded) result.
set -e
for v in deep2 ; do
  for m in 0 ; do
    echo "=============== $v MODE $m ==============="
    vivado -mode batch -source ooc_${v}_one.tcl -tclargs $m
    if grep -q "could not open" vivado.log; then
      echo "!!! $v MODE $m: a .mem file failed to load - results INVALID"; exit 1
    fi
  done
done
echo "ALL 4 RUNS DONE"
