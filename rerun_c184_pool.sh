#!/usr/bin/env bash
# Homogeneous eight-retrain pool on the CORRECTED 184-configuration corpus.
#
# Why: six of the eight runs behind Table 7 trained on the 185-corpus, which
# still contained BS8414_DCLG_Test7 - excluded 2026-08-19 by
# explain_part1.check_group_determinism as a different system under a shared
# label. Held-out and validation sets were identical throughout (verified from
# the stored test_chids/valid_chids), so only the training set differed, but a
# single-corpus pool removes the disclosure rather than managing it.
#
# Seed pattern replicates the original pool exactly: five runs at base seed 42,
# one each at 45, 48, 52. The five 42-runs are not duplicates - cudnn.benchmark
# leaves the per-member draw non-deterministic, which is what the retrain band
# actually measures.
set -u
cd "$(dirname "$0")"
PY=./venv/Scripts/python.exe

run () {                                  # $1 = tag, $2 = base seed
  d="models_part1_fix184_$1"
  if [ -d "$d" ]; then echo "  SKIP $d (exists)"; return; fi
  echo "  === $d  seed $2  ==="
  $PY -u train_part1.py --model-dir "$d" --members 3 --seed "$2" \
      > "train_part1_fix184_$1.log" 2> "train_part1_fix184_$1.err.log"
  $PY -u evaluate_part1.py --model-dir "$d" \
      > "evaluate_part1_fix184_$1.log" 2> "evaluate_part1_fix184_$1.err.log"
  echo "      done $(date +%H:%M)"
}

run a1 42
run a2 42
run a3 42
run a4 42
run a5 42
run s45 45
run s48 48
run s52 52

echo "  ALL EIGHT COMPLETE $(date +%H:%M)"
