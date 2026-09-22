#!/usr/bin/env bash
set -euo pipefail

python run_sweep.py \
  --widths 8 16 32 64 128 256 \
  --seeds 42 43 44 \
  --epochs 30 \
  --patience 5

python run_sweep.py \
  --widths 4 \
  --seeds 42 43 44 \
  --epochs 60 \
  --patience 5

python analyze.py
python benchmark.py
python analyze_tradeoff.py
