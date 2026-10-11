#!/bin/bash
# catboost_seeds_cv_per_seed.sh - per-seed wrapper for catboost_seeds_cv.py: the same setup as the earlier all-seeds run script for folds 1 and 7 (same feat md5 check, same symlinks, same parameters, same code catboost_seeds_cv.py);
# the only difference is that every seed runs in its own python process (--train-seeds s): the process exits between seeds and all memory is returned, so the RSS of 8 fits in one process does not accumulate.
# The recipe is unchanged: rounds, lr, depth, border, threads, categorical encoding and seed numbers are those of the earlier run script; parts already on disk are skipped by catboost_seeds_cv.py itself (the script can be restarted from the top).
# Usage: bash catboost_seeds_cv_per_seed.sh fold1    fold 1, seeds 0-7
#        bash catboost_seeds_cv_per_seed.sh fold7b   fold 7, seeds 0-7 (run after fold 1 on the same machine, when a second machine with 6 free cores is not available)
#        bash catboost_seeds_cv_per_seed.sh fold7a   fold 7, seeds 0-7 (on a second machine; copy the parts into oof_full/parts of the first machine afterwards)
set -eu -o pipefail
MODE=${1:?fold1|fold7a|fold7b}
if [ "$MODE" = fold7a ]; then
  M=${PRC_ROOT:?set PRC_ROOT to the working root}; FEAT=$M/night2/feat.parquet; HIST=$M/night2/oof; FOLD=7
else
  M=${PRC_ROOT:?set PRC_ROOT to the working root}; FEAT=$M/final/inputs/night2_feat.parquet; HIST=$M/eightseed/hist; FOLD=1; [ "$MODE" = fold7b ] && FOLD=7
fi
[ "$MODE" = fold1 ] || [ "$MODE" = fold7a ] || [ "$MODE" = fold7b ] || { echo "unknown mode $MODE"; exit 2; }
V=$M/eightseed; PY=$M/.venv/bin/python; CODE=$M/rerun/code_final_20261008; W=$V/oof_full
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONUNBUFFERED=1
mkdir -p "$W"
[ "$(md5 -q $FEAT)" = c3aabafc7059588e7d11fec643261685 ] || { echo "feat md5 mismatch"; exit 4; }
ln -sfn $FEAT "$W/feat.parquet"; ln -sfn $M/night2/results.tsv "$W/results.tsv"
for f in ens_v5b ens_v6 cat_delta cat_taxi; do for mo in 1 7; do test -s "$HIST/${f}_$mo.npy"; done; done
for s in 0 1 2 3 4 5 6 7; do
  echo "== seed $s $(date -u +%FT%TZ)"
  $PY -u $V/code/catboost_seeds_cv.py --code $CODE --work "$W" --hist $HIST --threads 6 --folds $FOLD --seeds 8 --train-only --train-seeds $s
done
ls -la "$W/parts"
echo "CAT8_OOF_SEEDS_DONE $MODE"
