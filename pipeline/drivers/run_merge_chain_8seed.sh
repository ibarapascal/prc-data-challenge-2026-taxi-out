#!/bin/bash
# Copy for the 8-seed-baseline candidate, derived from run_merge_chain.sh: root F -> $M/final_eightseed; the output of final_full becomes submit/upstanding-firefly_final_full_eightseed.parquet,
# no promote step (check_write_full writes that name directly), the variable FULL_FIRST points to submit/full_ref.parquet (read-only symlink to the earlier full version on the 3-seed baseline), cmpb also writes a comparison of this candidate against it; everything else is unchanged.
# run_merge_chain_8seed.sh - final-phase merge chain on a single compute machine: merge without stand correction (merge_nostand), drift gate (driftgate), final merge (merge_final), checks and writer (check_write).
# Algorithm and parameters as in the residual-layer configuration: variant_merged = merge of variant_simple and variant_full (conservative cells taken from variant_simple); drift gate = the pre-written rule in drift_gate.py;
# merge = merge_predictions.py (w=0.25; gated rows take variant_merged_sc; non-gated rows outside the LIRF artefact class get baseline + w*D; gated rows of cells the drift gate reverts go back to variant_simple). Only paths and the shell wrapper differ.
# Usage (any cwd):
#   bash run_merge_chain_8seed.sh gate                 merge_nostand: resid_merged -> merge_norevert: merge without reverts (merged_norevert = the drift-gate reference "without reverts") -> driftgate (final/driftgate/drift_gate_final.json)
#   bash run_merge_chain_8seed.sh final '<cells>'      merge_final: merge_predictions --revert-cells '<cells>' (cells judged "revert" by driftgate; empty string = no reverts) -> check_write checks and writes final/submit/ from the template
#   bash run_merge_chain_8seed.sh gate_full            full version (ADS-B in all four months): merge_nostand -> merged_norevert -> driftgate_full (final/driftgate/drift_gate_final_full.json)
#   bash run_merge_chain_8seed.sh final_full '<cells>' full version: merge_final_full (resid_pred/..._final_full_raw) -> partial-day patch (patch_partial_day.py) -> check_write_full writes submit/upstanding-firefly_final_full.parquet
#                                           -> comparison with the fallback version (compare_prediction_files.py -> logs/full_vs_baseline.json); the fallback file is read-only
#   Optional environment variables of final_full (all unset = the first full version, unchanged):
#     FULL_TAG=_patched    suffix for intermediate files and report names; check_write_full first writes submit/upstanding-firefly_final_full.tmp_patched.parquet, and the promote step renames it over final_full only after
#                          checking FULL_FIRST_MD5 (submit/upstanding-firefly_final_full_first.parquet must exist and its md5 must match); also writes a comparison of the new file against the first one
#     PATCH_MARGIN=64800   --margin of the partial-day patch (default 1800); PATCH_UNTIL=2026-06-08T00:00 upper bound of the affected rows' MVT (default 2026-06-07T03:00)
#     FLOOR_Q=0.001        enables the lower-bound post-processing (floor_clip.py: global q-quantile of the 2025 training departure labels, rounded; only rows below the bound are raised); unset = off
# Logs go to final/logs/fin_<step>.log; outputs are written to .tmp first and renamed.
set -u -o pipefail
MODE=${1:?gate|final}
M=${PRC_ROOT:?set PRC_ROOT to the working root}
F=$M/final_eightseed
PY=$M/.venv/bin/python
C=$F/code
STANDS=ext/xplane-apt/stand_coords_merged.parquet
S=resid_pred/upstanding-firefly
KEEP=EGLL-06,EGLL-07,LIRF   # conservative cells (February->January, June->July), as in run_stage_chain.sh
for v in $(env | grep -oE '^(RESID|TC)_[A-Z0-9_]*'); do unset "$v"; done
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONUNBUFFERED=1
E_RESID=(RESID_EXTRA=art RESID_MONTH=hyb:01 RESID_FULL_THR=0.15 RESID_NG=off RESID_THREADS=4 RESID_STAND=fix
       RESID_STAND_DIR=feats_tma RESID_FEATS=feats_tma RESID_POOL_FEATS=feats_tmafy RESID_CELLFB=pred
       RESID_TK_FILE=v22/tk_main.parquet RESID_TK_POOL=v22/tk_pool.parquet RESID_CODE_DIR=$C RESID_STANDS=$STANDS
       TC_CODE_DIR=$C TC_CROP=crop TC_STANDS=$STANDS)
E_FIN=(RESID_BASE_SUB=baseline_used.parquet RESID_N26=auto RESID_WINTER_TEST=01,02 RESID_RAW_DIR=raw)
MERGE=(--cand ${S}_resid_merged_sc.parquet --refpred n2/baseline_used.parquet --ensavg n2/baseline_used.parquet --res ${S}_resid_full.res.npy
       --tab gaptab/out/fy26_final_k8.npz --keys raw/ranking.parquet --w 0.25 --n26 auto --n-gated auto)
cd $F && mkdir -p logs driftgate/dir driftgate/sub submit
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a logs/fin_chain.log; }
step() {  # step <name> <failure code> <command...>
  local n=$1 rc=$2; shift 2; log "start $n"
  ( "$@" ) > logs/fin_$n.log 2>&1 || { log "FAIL $n -> exit $rc"; exit "$rc"; }
  log "done $n"
}
merge_nostand() {  # merge variant_simple + variant_full -> variant_merged, environment as the merge in standcorr of run_stage_chain.sh
  env "${E_RESID[@]}" "${E_FIN[@]}" $PY -W ignore $C/residual_context.py merge --base ${S}_resid_simple.parquet --new ${S}_resid_full.parquet \
      --keep $KEEP --out ${S}_resid_merged.tmp.parquet || return 1
  mv ${S}_resid_merged.tmp.merge.json ${S}_resid_merged.merge.json && mv ${S}_resid_merged.tmp.parquet ${S}_resid_merged.parquet
}
merge_norevert() {
  $PY -u $C/merge_predictions.py "${MERGE[@]}" --out ${S}_merged_norevert.tmp.parquet --report logs/mergepred_merged_norevert.json \
    && mv ${S}_merged_norevert.tmp.parquet ${S}_merged_norevert.parquet
}
driftgate() {  # --dir/--sub are assembled from symlinks into the directory layout drift_gate expects; the original cv_eval sits next to the output (used for rollback_cost)
  ln -sfn ../../v22/tk_main.parquet driftgate/dir/tk_main.parquet; ln -sfn ../../v22/tk_pool.parquet driftgate/dir/tk_pool.parquet
  ln -sfn ../../v22/sc_table_all_full.parquet driftgate/dir/sc_table_all_full.parquet; ln -sfn ../../feats_tma driftgate/dir/feats_tma
  for x in resid_simple resid_merged resid_merged_sc merged_norevert; do ln -sfn ../../${S}_$x.parquet driftgate/sub/upstanding-firefly_$x.parquet; done
  ln -sfn ../../${S}_resid_full.res.npy driftgate/sub/upstanding-firefly_resid_full.res.npy; ln -sfn ../../raw/ranking.parquet driftgate/sub/ranking.parquet
  ln -sfn ../ref/20260926-v22-cv_eval.json driftgate/20260926-v22-cv_eval.json
  $PY -u $C/drift_gate.py --dir driftgate/dir --sub driftgate/sub --out driftgate/drift_gate_final.tmp.json \
      --months 01,02,06,07 --ref-map 02:01,06:07 --cv-map 02:01,06:07 \
      --names variant_simple=resid_simple,variant_merged=resid_merged,variant_merged_sc=resid_merged_sc,variant_norevert=merged_norevert,res=resid_full && mv driftgate/drift_gate_final.tmp.json driftgate/drift_gate_final.json
}
merge_final() {
  local rv=$1; local extra=(); [ -n "$rv" ] && extra=(--revert-cells "$rv" --simple_file ${S}_resid_simple.parquet)
  $PY -u $C/merge_predictions.py "${MERGE[@]}" ${extra[@]+"${extra[@]}"} --ref ${S}_merged_norevert.parquet --out ${S}_final.tmp.parquet --report logs/mergepred_final.json \
    && mv ${S}_final.tmp.parquet ${S}_final.parquet
}
check_write() {
  $PY -u $C/check_submission.py --pred ${S}_final.parquet --tmpl raw/submitting.parquet --earlypred $M/sub/upstanding-firefly_v22.parquet \
      --basefile n2/baseline_used.parquet --oldbase $M/sub/upstanding-firefly_v8.parquet --ranking raw/ranking.parquet --train-dir raw \
      --out submit/upstanding-firefly_final.parquet --report logs/submission_check.json || return 1
  cp -p submit/upstanding-firefly_final.parquet submit/upstanding-firefly_final_baseline_20261008.parquet
  md5 -r submit/*.parquet ${S}_final.parquet ${S}_merged_norevert.parquet ${S}_resid_merged.parquet | tee submit/md5.txt
}
# ---------------- full version (ADS-B also in February and June): no file name of the fallback version is touched; its two copies (submit/upstanding-firefly_final{,_baseline_20261008}.parquet) are read-only
BASE=submit/upstanding-firefly_final_baseline_20261008.parquet
merge_norevert_full() {
  $PY -u $C/merge_predictions.py "${MERGE[@]}" --out ${S}_merged_norevert.tmp.parquet --report logs/mergepred_merged_norevert_full.json \
    && mv ${S}_merged_norevert.tmp.parquet ${S}_merged_norevert.parquet
}
driftgate_full() {  # as driftgate, writes drift_gate_final_full.json
  ln -sfn ../../v22/tk_main.parquet driftgate/dir/tk_main.parquet; ln -sfn ../../v22/tk_pool.parquet driftgate/dir/tk_pool.parquet
  ln -sfn ../../v22/sc_table_all_full.parquet driftgate/dir/sc_table_all_full.parquet; ln -sfn ../../feats_tma driftgate/dir/feats_tma
  for x in resid_simple resid_merged resid_merged_sc merged_norevert; do ln -sfn ../../${S}_$x.parquet driftgate/sub/upstanding-firefly_$x.parquet; done
  ln -sfn ../../${S}_resid_full.res.npy driftgate/sub/upstanding-firefly_resid_full.res.npy; ln -sfn ../../raw/ranking.parquet driftgate/sub/ranking.parquet
  ln -sfn ../ref/20260926-v22-cv_eval.json driftgate/20260926-v22-cv_eval.json
  $PY -u $C/drift_gate.py --dir driftgate/dir --sub driftgate/sub --out driftgate/drift_gate_final_full.tmp.json \
      --months 01,02,06,07 --ref-map 02:01,06:07 --cv-map 02:01,06:07 \
      --names variant_simple=resid_simple,variant_merged=resid_merged,variant_merged_sc=resid_merged_sc,variant_norevert=merged_norevert,res=resid_full && mv driftgate/drift_gate_final_full.tmp.json driftgate/drift_gate_final_full.json
}
merge_final_full() {
  local rv=$1; local extra=(); [ -n "$rv" ] && extra=(--revert-cells "$rv" --simple_file ${S}_resid_simple.parquet)
  $PY -u $C/merge_predictions.py "${MERGE[@]}" ${extra[@]+"${extra[@]}"} --ref ${S}_merged_norevert.parquet --out ${S}_final_full_raw$T.tmp.parquet \
      --report logs/mergepred_final_full$T.json && mv ${S}_final_full_raw$T.tmp.parquet ${S}_final_full_raw$T.parquet
}
patch_day() {  # 2026-06-06 ADS-B partly missing: contaminated gated rows take the same row's prediction from the fallback version (rule in the docstring of patch_partial_day.py)
  $PY -u $C/patch_partial_day.py --pred ${S}_final_full_raw$T.parquet --base $BASE --crop crop --feats-tm feats_tm --feats-tma feats_tma \
      --ranking raw/ranking.parquet --airports $C/airports.json --margin "$PATCH_MARGIN" --until "$PATCH_UNTIL" \
      --out ${S}_final_full$T.parquet --report logs/partial_day_patch$T.json
}
floor() {  # lower-bound post-processing (FLOOR_Q unset = off): the residual layer is untouched, rows below the bound are raised to it; the fallback version is used for statistics only
  $PY -u $C/floor_clip.py --pred ${S}_final_full$T.parquet --train-dir raw --ranking raw/ranking.parquet --base $BASE --q "$FLOOR_Q" \
      --out ${S}_final_full_floor$T.parquet --report logs/floor_clip$T.json
}
check_write_full() {
  $PY -u $C/check_submission.py --pred $CHECK_IN --tmpl raw/submitting.parquet --earlypred $M/sub/upstanding-firefly_v22.parquet \
      --basefile n2/baseline_used.parquet --oldbase $M/sub/upstanding-firefly_v8.parquet --ranking raw/ranking.parquet --train-dir raw \
      --out $CHECK_OUT --report logs/submission_check_full$T.json || return 1
  local fl=(); [ -n "$FLOOR_Q" ] && fl=($CHECK_IN)
  md5 -r submit/*.parquet ${S}_final_full_raw$T.parquet ${S}_final_full$T.parquet ${fl[@]+"${fl[@]}"} ${S}_merged_norevert.parquet ${S}_resid_merged.parquet | tee submit/md5_full$T.txt
}
promote() {  # only with FULL_TAG: once the first full version is archived and checked, rename the temporary file that already passed the checks over final_full
  [ -f "$FULL_FIRST" ] || { echo "missing $FULL_FIRST"; return 1; }
  local h; h=$(md5 -q "$FULL_FIRST"); [ "$h" = "${FULL_FIRST_MD5:?}" ] || { echo "full_first md5 $h != $FULL_FIRST_MD5"; return 1; }
  mv "$CHECK_OUT" "$FINAL" && md5 -r submit/*.parquet | tee -a submit/md5_full$T.txt
}
cmpb() {
  $PY -u $C/compare_prediction_files.py --full $FINAL --base $BASE --ranking raw/ranking.parquet \
      --old baseline_20261008 --out logs/full_vs_baseline$T.json || return 1
  if [ -n "$T" ]; then  # comparison of the new full version against the reference FULL_FIRST (new - FULL_FIRST), to see where the two changes together land
    $PY -u $C/compare_prediction_files.py --full $FINAL --base $FULL_FIRST --ranking raw/ranking.parquet \
        --old baseline_20261008 --out logs/full_vs_first$T.json
  fi
}
T=${FULL_TAG-}
PATCH_MARGIN=${PATCH_MARGIN:-1800}; PATCH_UNTIL=${PATCH_UNTIL:-2026-06-07T03:00}; FLOOR_Q=${FLOOR_Q-}
FINAL=submit/upstanding-firefly_final_full_eightseed.parquet; FULL_FIRST=submit/full_ref.parquet
if [ -n "$FLOOR_Q" ]; then CHECK_IN=${S}_final_full_floor$T.parquet; else CHECK_IN=${S}_final_full$T.parquet; fi
CHECK_OUT=$FINAL
log "== fin $MODE start"
case $MODE in
  gate)  step merge_nostand 61 merge_nostand; step merge_norevert 62 merge_norevert; step driftgate 63 driftgate ;;
  final) RV=${2-}; log "revert-cells='$RV'"; step merge_final 71 merge_final "$RV"; step check_write 72 check_write ;;
  gate_full)  step merge_nostand 81 merge_nostand; step merge_norevert_full 82 merge_norevert_full; step driftgate_full 83 driftgate_full ;;
  final_full) RV=${2-}; log "revert-cells='$RV' tag='$T' patch_margin=$PATCH_MARGIN until=$PATCH_UNTIL floor_q='$FLOOR_Q'"
    step merge_final_full$T 91 merge_final_full "$RV"; step patch_day$T 92 patch_day
    [ -n "$FLOOR_Q" ] && step floor$T 95 floor
    step check_write_full$T 93 check_write_full
    : no promote step
    step cmpb$T 94 cmpb ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
log "== fin $MODE RUN_FINAL_${MODE}_DONE"
