#!/bin/bash
# run_submission_chain.sh - variant with the grid-cell fallback off (CELLFB=off; output tag "_cellfb_off").
# Takes every setting of the earlier full version (FULL_TAG=_patched: full version on the 3-seed baseline) and only swaps the residual-layer
# (pred_simple / pred_full) output for the CELLFB=off one (the prediction-side grid-cell fallback did not pass the pre-registered CV gate, so it is removed).
# Under CELLFB=pred the residual step already wrote <tag>_off as well (same boosters; test rows whose cell is missing use the off encoding),
# so nothing is retrained. All other steps and parameters are identical to the full version
# (run_merge_chain.sh gate_full + final_full, FULL_TAG=_patched).
# Usage (cwd arbitrary):
#   bash run_submission_chain.sh gate               applysc + merge_sc (resid_merged_sc) + merge_nosc (resid_merged) + merge_norevert + driftgate -> driftgate<TAG>/drift_gate_final_full<TAG>.json
#   bash run_submission_chain.sh final '<cells>'    merge_final (reverted cells = driftgate verdict) -> partial-day patch (64800 / 06-08T00:00) -> floor q .001 -> check_write -> comparison with the full version and the fallback file
# Switches (environment variables; they only affect this script's output names and input choice, no existing file is changed):
#   CELLFB_OFF_CELLFB=off|pred  default off; pred = take the raw residual output, which must reproduce the full version byte for byte (635370b1...) as a script self-check (check_write output goes to resid_pred, not to submit/)
#   CELLFB_OFF_TAG=_cellfb_off  output suffix (suggested for the self-check: _cellfb_offchk)
#   CELLFB_OFF_ROOT=final       working tree (relative to $PRC_ROOT); for the 8-seed baseline point it at that working tree (needs the same structure: resid_pred, v22, n2, raw, crop, feats, gaptab)
#   CELLFB_OFF_OUT=...          final file path (default $CELLFB_OFF_ROOT/submit/upstanding-firefly_final_full<TAG>.parquet; refuses to write if it exists)
# Logs: <root>/logs/cellfb_off<TAG>_<step>.log; intermediate outputs are written as .tmp and then renamed. The fallback file and the full version are read-only.
set -u -o pipefail
MODE=${1:?gate|final}
M=${PRC_ROOT:?set PRC_ROOT to the working root}
F=$M/${CELLFB_OFF_ROOT:-final}
C=$M/final/code
PY=$M/.venv/bin/python
T=${CELLFB_OFF_TAG:-_cellfb_off}
CF=${CELLFB_OFF_CELLFB:-off}
case $CF in off) SX=_off ;; pred) SX= ;; *) echo "CELLFB_OFF_CELLFB must be off|pred"; exit 2 ;; esac
STANDS=ext/xplane-apt/stand_coords_merged.parquet
S=resid_pred/upstanding-firefly
SIMPLE_PRED=${S}_resid_simple$SX
FULL_PRED=${S}_resid_full$SX
KEEP=EGLL-06,EGLL-07,LIRF   # conservative cells, same as run_stage_chain.sh / run_merge_chain.sh
BASE=$M/final/submit/upstanding-firefly_final_baseline_20261008.parquet   # fallback file without ADS-B in Feb/Jun (read-only; source of the partial-day patch values, floor only reported)
FULL_REF=$M/final/submit/upstanding-firefly_final_full.parquet                   # earlier full version (read-only; for comparison)
FULL_REF_MD5=635370b1ed7063f416f95caf25af0859
for v in $(env | grep -oE '^(RESID|TC)_[A-Z0-9_]*'); do unset "$v"; done
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONUNBUFFERED=1
E_RESID=(RESID_EXTRA=art RESID_MONTH=hyb:01 RESID_FULL_THR=0.15 RESID_NG=off RESID_THREADS=4 RESID_STAND=fix
       RESID_STAND_DIR=feats_tma RESID_FEATS=feats_tma RESID_POOL_FEATS=feats_tmafy RESID_CELLFB=$CF
       RESID_TK_FILE=v22/tk_main.parquet RESID_TK_POOL=v22/tk_pool.parquet RESID_CODE_DIR=$C RESID_STANDS=$STANDS
       TC_CODE_DIR=$C TC_CROP=crop TC_STANDS=$STANDS)
E_FIN=(RESID_BASE_SUB=baseline_used.parquet RESID_N26=auto RESID_WINTER_TEST=01,02 RESID_RAW_DIR=raw)
W7=(RESID_WINTER=11+12 RESID_WINTER_APTS=EDDF,EDDM,EGLL,EHAM,LEBL,LSZH,LEMD)
MERGE=(--cand ${S}_resid_merged_sc$T.parquet --refpred n2/baseline_used.parquet --ensavg n2/baseline_used.parquet --res ${FULL_PRED}.res.npy
       --tab gaptab/out/fy26_final_k8.npz --keys raw/ranking.parquet --w 0.25 --n26 auto --n-gated auto)
cd "$F" && mkdir -p logs "driftgate$T/dir" "driftgate$T/sub" submit
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "logs/cellfb_off${T}_chain.log"; }
step() {  # step <name> <failure code> <function> [args...]
  local n=$1 rc=$2; shift 2; log "start $n"
  ( "$@" ) > "logs/cellfb_off${T}_$n.log" 2>&1 || { log "FAIL $n -> exit $rc"; exit "$rc"; }
  log "done $n"
}
need() { for f in "$@"; do [ -e "$f" ] || { log "missing prerequisite $f"; exit 9; }; done; }

applysc() {  # same as run_stage_chain.sh standcorr, step 1
  env "${E_RESID[@]}" "${E_FIN[@]}" "${W7[@]}" $PY -W ignore $C/residual_context.py applysc --sub ${FULL_PRED}.parquet --res ${FULL_PRED}.res.npy \
      --table v22/sc_table_all_full.parquet --keep $KEEP --out ${S}_resid_full_sc$T.tmp.parquet || return 1
  mv ${S}_resid_full_sc$T.tmp.sc.json ${S}_resid_full_sc$T.sc.json && mv ${S}_resid_full_sc$T.tmp.parquet ${S}_resid_full_sc$T.parquet
}
merge_sc() {  # same as run_stage_chain.sh standcorr, step 2
  env "${E_RESID[@]}" "${E_FIN[@]}" $PY -W ignore $C/residual_context.py merge --base ${SIMPLE_PRED}.parquet --new ${S}_resid_full_sc$T.parquet \
      --keep $KEEP --out ${S}_resid_merged_sc$T.tmp.parquet || return 1
  mv ${S}_resid_merged_sc$T.tmp.merge.json ${S}_resid_merged_sc$T.merge.json && mv ${S}_resid_merged_sc$T.tmp.parquet ${S}_resid_merged_sc$T.parquet
}
merge_nosc() {  # same as run_merge_chain.sh merge_nostand
  env "${E_RESID[@]}" "${E_FIN[@]}" $PY -W ignore $C/residual_context.py merge --base ${SIMPLE_PRED}.parquet --new ${FULL_PRED}.parquet \
      --keep $KEEP --out ${S}_resid_merged$T.tmp.parquet || return 1
  mv ${S}_resid_merged$T.tmp.merge.json ${S}_resid_merged$T.merge.json && mv ${S}_resid_merged$T.tmp.parquet ${S}_resid_merged$T.parquet
}
merge_norevert() {  # same as run_merge_chain.sh merge_norevert_full
  $PY -u $C/merge_predictions.py "${MERGE[@]}" --out ${S}_merged_norevert$T.tmp.parquet --report logs/mergepred_merged_norevert$T.json \
    && mv ${S}_merged_norevert$T.tmp.parquet ${S}_merged_norevert$T.parquet
}
driftgate() {  # same as run_merge_chain.sh driftgate_full, with directory and names switched to this tag
  local d=driftgate$T
  ln -sfn ../../v22/tk_main.parquet $d/dir/tk_main.parquet; ln -sfn ../../v22/tk_pool.parquet $d/dir/tk_pool.parquet
  ln -sfn ../../v22/sc_table_all_full.parquet $d/dir/sc_table_all_full.parquet; ln -sfn ../../feats_tma $d/dir/feats_tma
  for x in resid_simple$SX resid_merged$T resid_merged_sc$T merged_norevert$T; do ln -sfn ../../${S}_$x.parquet $d/sub/upstanding-firefly_$x.parquet; done
  ln -sfn ../../${FULL_PRED}.res.npy $d/sub/$(basename ${FULL_PRED}).res.npy; ln -sfn ../../raw/ranking.parquet $d/sub/ranking.parquet
  ln -sfn ../ref/20260926-v22-cv_eval.json $d/20260926-v22-cv_eval.json
  $PY -u $C/drift_gate.py --dir $d/dir --sub $d/sub --out $d/drift_gate_final_full$T.tmp.json \
      --months 01,02,06,07 --ref-map 02:01,06:07 --cv-map 02:01,06:07 \
      --names variant_simple=resid_simple$SX,variant_merged=resid_merged$T,variant_merged_sc=resid_merged_sc$T,variant_norevert=merged_norevert$T,res=resid_full$SX \
    && mv $d/drift_gate_final_full$T.tmp.json $d/drift_gate_final_full$T.json \
    && jq -c '.verdict | to_entries | map(select(.value.verdict == "revert_to_simple_variant")) | map(.key)' $d/drift_gate_final_full$T.json
}
merge_final() {  # same as run_merge_chain.sh merge_final_full
  local rv=$1; local extra=(); [ -n "$rv" ] && extra=(--revert-cells "$rv" --simple_file ${SIMPLE_PRED}.parquet)
  $PY -u $C/merge_predictions.py "${MERGE[@]}" ${extra[@]+"${extra[@]}"} --ref ${S}_merged_norevert$T.parquet --out ${S}_final_full_raw$T.tmp.parquet \
      --report logs/mergepred_final_full$T.json && mv ${S}_final_full_raw$T.tmp.parquet ${S}_final_full_raw$T.parquet
}
patch_day() {  # as in the full version: margin 64800, until 06-08T00:00, patched rows taken from the fallback file
  $PY -u $C/patch_partial_day.py --pred ${S}_final_full_raw$T.parquet --base $BASE --crop crop --feats-tm feats_tm --feats-tma feats_tma \
      --ranking raw/ranking.parquet --airports $C/airports.json --margin 64800 --until 2026-06-08T00:00 \
      --out ${S}_final_full$T.parquet --report logs/partial_day_patch$T.json
}
floor() {  # as in the full version: FLOOR_Q 0.001 (-> 119)
  $PY -u $C/floor_clip.py --pred ${S}_final_full$T.parquet --train-dir raw --ranking raw/ranking.parquet --base $BASE --q 0.001 \
      --out ${S}_final_full_floor$T.parquet --report logs/floor_clip$T.json
}
check_write() {
  $PY -u $C/check_submission.py --pred ${S}_final_full_floor$T.parquet --tmpl raw/submitting.parquet --earlypred $M/sub/upstanding-firefly_v22.parquet \
      --basefile n2/baseline_used.parquet --oldbase $M/sub/upstanding-firefly_v8.parquet --ranking raw/ranking.parquet --train-dir raw \
      --out $CHECK_OUT --report logs/submission_check_full$T.json
}
promote() {  # off: rename the temp file that passed check_write to the final name (refuse to write if it exists); pred (self-check): only compare the md5 with the full version
  local h; h=$(md5 -q "$CHECK_OUT")
  if [ "$CF" = pred ]; then
    echo "selfcheck md5 $h vs full version $FULL_REF_MD5 -> $([ "$h" = "$FULL_REF_MD5" ] && echo EQUAL || echo DIFF)"; [ "$h" = "$FULL_REF_MD5" ]; return
  fi
  [ -e "$FINAL" ] && { echo "refusing to write: $FINAL already exists"; return 1; }
  mv "$CHECK_OUT" "$FINAL" && md5 -r "$FINAL" ${S}_final_full_raw$T.parquet ${S}_final_full$T.parquet ${S}_final_full_floor$T.parquet \
      ${S}_resid_merged_sc$T.parquet ${S}_resid_merged$T.parquet ${S}_merged_norevert$T.parquet "$FULL_REF" "$BASE" | tee submit/md5_full$T.txt
}
cmpb() {  # comparison: against the full version and against the fallback file (per month x airport, cells of full_vs_baseline)
  local fin=$FINAL; [ "$CF" = pred ] && fin=$CHECK_OUT
  $PY -u $C/compare_prediction_files.py --full $fin --base $FULL_REF --ranking raw/ranking.parquet --old baseline_20261008 --out logs/cellfb_off_vs_fullref$T.json || return 1
  $PY -u $C/compare_prediction_files.py --full $fin --base $BASE --ranking raw/ranking.parquet --old baseline_20261008 --out logs/cellfb_off_vs_baseline$T.json
}
FINAL=${CELLFB_OFF_OUT:-submit/upstanding-firefly_final_full$T.parquet}
if [ "$CF" = pred ]; then CHECK_OUT=${S}_check_write$T.parquet; else CHECK_OUT=submit/upstanding-firefly_final_full$T.tmp.parquet; fi
log "== cellfb_off $MODE start cellfb=$CF tag=$T root=$F simple_pred=$SIMPLE_PRED full_pred=$FULL_PRED"
need ${SIMPLE_PRED}.parquet ${SIMPLE_PRED}.res.npy ${FULL_PRED}.parquet ${FULL_PRED}.res.npy v22/sc_table_all_full.parquet n2/baseline_used.parquet gaptab/out/fy26_final_k8.npz "$BASE" "$FULL_REF"
case $MODE in
  gate)  step applysc 61 applysc; step merge_sc 62 merge_sc; step merge_nosc 63 merge_nosc; step merge_norevert 64 merge_norevert; step driftgate 65 driftgate
         cat "logs/cellfb_off${T}_driftgate.log" | tail -1 | tee -a "logs/cellfb_off${T}_chain.log" ;;
  final) RV=${2-}; log "revert-cells='$RV'"; need ${S}_resid_merged_sc$T.parquet ${S}_merged_norevert$T.parquet
         step merge_final 71 merge_final "$RV"; step patch_day 72 patch_day; step floor 73 floor; step check_write 74 check_write; step promote 75 promote; step cmpb 76 cmpb ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
log "== cellfb_off $MODE RUN_CELLFB_OFF_${MODE}_DONE"
