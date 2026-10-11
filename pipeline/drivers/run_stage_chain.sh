#!/bin/bash
# run_stage_chain.sh - final-phase chain of the 2026 ground features, the 2025-side assets, the residual predictions, the stand correction and the tabular arms
# (steps align_feat to tab_arms below), run on a single compute machine. Algorithm, parameters, seeds and conventions are those of the original runs; only paths and the shell wrapper differ.
# Usage (any cwd): bash run_stage_chain.sh smoke|adsb|tab
#   smoke: feature-store alignment (final/n2/feat.parquet) + build_feats on one day + import self-check of the scripts (measures peak memory)
#   adsb : 2026 ground features (2026-01/02/06/07 feats_tm/tma; the full version includes February and June) -> 2025 ground features (feats_tm/tma 62 days each, feats_tmafy 80 days)
#          -> track-context rows (track_rows -> tk_main)
#          -> 2025-side assets recomputed on the old row order (tk_pool, cv_tm_s8, cv_anchor_s8, cv_tkall_s8, evalx -> sc_table_all_full; reproduction check)
#          -> residual predictions (resid_simple = simpler variant, resid_full = full variant with track context) -> stand correction (applysc -> resid_full_sc, merge -> resid_merged_sc)
#   tab  : tabular features (build_tabular_features.py -> pf.parquet) -> tabular arms (tabular_term.py, 8 seeds -> fy26_final_k8)
# Conventions: each successful step writes final/_done/<step> and finished steps are skipped (the chain can be restarted from the top); outputs are written to .tmp first and renamed;
#   logs go to final/logs/<step>.log, the chain summary to final/logs/chain_<mode>.log.
# Work dirs: final/ (its n2/ = feature store of the baseline, aligned to the old row order); final/r25/ (its n2/ = the early-phase feature store, used only to recompute the 2025-side assets and reproduce the original outputs).
set -u -o pipefail
MODE=${1:?smoke|adsb|tab}
M=${PRC_ROOT:?set PRC_ROOT to the working root}
F=$M/final
RUN3SEED=$M/rerun/final_20261008
PY=$M/.venv/bin/python
C=$F/code
STANDS=ext/xplane-apt/stand_coords_merged.parquet
for v in $(env | grep -oE '^(RESID|TC)_[A-Z0-9_]*'); do unset "$v"; done   # clear leftover switches
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONUNBUFFERED=1
mkdir -p $F/logs $F/_done
LOG=$F/logs/chain_$MODE.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

run_step() {  # run_step <name> <exit code on failure> <function> [args...]
  local n=$1 rc=$2; shift 2
  if [ -f "$F/_done/$n" ]; then log "skip ${n} (already done)"; return 0; fi
  log "start $n"; local t0; t0=$(date +%s)
  ( "$@" ) > "$F/logs/$n.log" 2>&1
  local r=$?
  if [ $r -ne 0 ]; then log "FAIL $n rc=$r -> exit $rc"; exit "$rc"; fi
  printf '{"step":"%s","sec":%d,"at":"%s"}\n' "$n" $(( $(date +%s) - t0 )) "$(date -u +%FT%TZ)" > "$F/_done/$n.tmp" && mv "$F/_done/$n.tmp" "$F/_done/$n"
  log "done ${n} ($(( $(date +%s) - t0 ))s)"
}
need() { for f in "$@"; do [ -e "$f" ] || { log "missing prerequisite $f"; exit 9; }; done; }
dates() { local ym=$1 d; for d in $(seq -f %02g "$2" "$3"); do printf '%s-%s ' "$ym" "$d"; done; }
D26="$(dates 2026-01 1 31)$(dates 2026-02 1 28)$(dates 2026-06 1 30)$(dates 2026-07 1 31)"   # full version: adds February and June (the fallback version had only January and July); 06-30 is included, so 07-01 is rebuilt with it
D25="$(dates 2025-01 1 31)$(dates 2025-07 1 31)"
DFY="$(dates 2025-06 11 30)$(dates 2025-11 1 30)$(dates 2025-12 1 30)"   # pool days: 06-11..30, November, 12-01..30

# Environment: residual-layer configuration of the submitted predictions. With RESID_CELLFB=pred, train_resid.py writes two outputs from the same
# boosters: <tag> (grid-cell fallback on) and <tag>_off (fallback off); run_submission_chain.sh reads the _off outputs (appendix B).
E_RESID=(RESID_EXTRA=art RESID_MONTH=hyb:01 RESID_FULL_THR=0.15 RESID_NG=off RESID_THREADS=4 RESID_STAND=fix
       RESID_STAND_DIR=feats_tma RESID_FEATS=feats_tma RESID_POOL_FEATS=feats_tmafy RESID_CELLFB=pred
       RESID_TK_FILE=v22/tk_main.parquet RESID_TK_POOL=v22/tk_pool.parquet RESID_CODE_DIR=$C RESID_STANDS=$STANDS
       TC_CODE_DIR=$C TC_CROP=crop TC_STANDS=$STANDS)
E_FIN=(RESID_BASE_SUB=baseline_used.parquet RESID_N26=auto RESID_WINTER_TEST=01,02 RESID_RAW_DIR=raw)   # final-phase inputs (baseline file, test-row count, winter-pool months, raw dir)
W7=(RESID_WINTER=11+12 RESID_WINTER_APTS=EDDF,EDDM,EGLL,EHAM,LEBL,LSZH,LEMD)
KEEP=EGLL-06,EGLL-07,LIRF   # conservative cells; February maps to January and June to July

bf_into() {  # bf_into <dest dir> <tmp dir> <anchor 0|1> <dates...>: build_feats -> check the file count -> merge into dest
  local dest=$1 tmp=$2 anc=$3; shift 3
  local af=""; [ "$anc" = 1 ] && af="--anchor"
  rm -rf "$tmp"
  $PY -W ignore $C/build_feats.py --crop crop --raw raw --out "$tmp" --redo --stands $STANDS --typefilter $af --dates "$@" || return 1
  local n; n=$(ls "$tmp"/*.parquet | wc -l | tr -d ' ')
  [ "$n" -eq $# ] || { echo "daily file count $n != $#"; return 1; }
  mkdir -p "$dest" && mv "$tmp"/* "$dest"/ && rmdir "$tmp"
}
par() {  # par <log prefix> -- run several "dest tmp anc dates" groups in parallel; return 0 only if all succeed
  local pre=$1; shift; local pids="" i=0 r=0 p
  while [ $# -gt 0 ]; do
    local spec=$1; shift; i=$((i+1))
    # shellcheck disable=SC2086
    bf_into $spec > "$F/logs/${pre}_$i.log" 2>&1 & pids="$pids $!"
  done
  for p in $pids; do wait "$p" || r=1; done
  return $r
}

# ---------------- smoke
align_feat() {  # reorder the training rows of the baseline feature file to the old feature-store row order -> final/n2/feat.parquet
  # The old store has four extra columns (dwp/relh/spread/frost, a METAR extension) that no downstream script reads (checked by grep), hence --allow-missing-cols
  cd $F && need $RUN3SEED/ensb/feat.parquet inputs/night2_feat.parquet
  $PY -u $C/feat_align.py --new $RUN3SEED/ensb/feat.parquet --old inputs/night2_feat.parquet --out n2/feat.parquet --report logs/feat_align_report.json --allow-missing-cols
}
smoke_bf() { cd $F && bf_into tmp_smoke_out .tmp_smoke 1 2026-01-15 && ls -la tmp_smoke_out; }
smoke_imp() {
  cd $F && env "${E_RESID[@]}" $PY $C/build_track_context.py --help >/dev/null && env "${E_RESID[@]}" $PY $C/residual_context.py --help >/dev/null \
    && env "${E_RESID[@]}" $PY $C/residual_cv.py --help >/dev/null && (cd $C && $PY tabular_term.py --help >/dev/null) \
    && $PY -c "import scipy, lightgbm, polars; print('imp ok', lightgbm.__version__, polars.__version__)"
}

# ---------------- adsb
ground26() { cd $F && par ground26 "feats_tm .tmp_ground26_tm 0 $D26" "feats_tma .tmp_ground26_tma 1 $D26"; }
ground25() { cd $F && par ground25 "feats_tm .tmp_ground25_tm 0 $D25" "feats_tma .tmp_ground25_tma 1 $D25" "feats_tmafy .tmp_ground25_fy 1 $DFY"; }
tc_rows() {  # track-context rows: environment of the earlier configuration, TC_TMA=feats_tma
  cd $F && need n2/feat.parquet cvtg_tm.oof.npy
  env RESID_EXTRA=art RESID_MONTH=hyb:01 RESID_FULL_THR=0.15 RESID_NG=off RESID_THREADS=2 RESID_STAND=fix RESID_STAND_DIR=feats_tm \
      RESID_FEATS=feats_tm TC_CODE_DIR=$C TC_CROP=crop TC_STANDS=$STANDS RESID_STANDS=$STANDS \
      $PY -W ignore $C/track_context_rows.py rows --out gap/track_rows.tmp.parquet && mv gap/track_rows.tmp.parquet gap/track_rows.parquet
}
tc_table() {
  cd $F && env "${E_RESID[@]}" $PY -W ignore $C/build_track_context.py main --rows gap/track_rows.parquet --feats feats_tma --out v22/tk_main.tmp.parquet \
    && mv v22/tk_main.tmp.json v22/tk_main.json && mv v22/tk_main.tmp.parquet v22/tk_main.parquet
}
asset25_pool() {  # winter-pool track-context table
  cd $F/r25 && env "${E_RESID[@]}" $PY -W ignore $C/build_track_context.py pool --feats feats_tmafy --months 11,12 --out v22/tk_pool.tmp.parquet \
    && mv v22/tk_pool.tmp.json v22/tk_pool.json && mv v22/tk_pool.tmp.parquet v22/tk_pool.parquet
}
cvmv() { mv "$1.tmp.oof.npy" "$1.oof.npy" && mv "$1.tmp.json" "$1.json"; }
asset25_cv_plain() {  # residual CV OOF without anchoring (default residual environment + CELLFB=off, feats_tm, 8 seeds, reference cvtg_tm)
  cd $F/r25 && env RESID_EXTRA=art RESID_MONTH=hyb:01 RESID_FULL_THR=0.15 RESID_NG=off RESID_THREADS=4 RESID_STAND=fix \
      RESID_STAND_DIR=feats_tm RESID_FEATS=feats_tm RESID_CELLFB=off \
      $PY -W ignore $C/residual_cv.py cv --seeds 8 --out resid_cv/cv_tm_s8.tmp.json && cvmv resid_cv/cv_tm_s8
}
asset25_cv_anchor() {  # the same with anchoring (feats_tma, reference cv_tm_s8)
  cd $F/r25 && env RESID_EXTRA=art RESID_MONTH=hyb:01 RESID_FULL_THR=0.15 RESID_NG=off RESID_THREADS=4 RESID_STAND=fix \
      RESID_STAND_DIR=feats_tma RESID_FEATS=feats_tma RESID_CELLFB=off \
      $PY -W ignore $C/residual_cv.py cv --seeds 8 --oof resid_cv/cv_tm_s8.oof.npy --out resid_cv/cv_anchor_s8.tmp.json && cvmv resid_cv/cv_anchor_s8
}
asset25_cv_tk() {  # the same with the 17 track-context columns (8 seeds, CELLFB=off, no winter pool setting)
  cd $F/r25 && env "${E_RESID[@]}" RESID_CELLFB=off \
      $PY -W ignore $C/residual_context.py cv --tk all --seeds 8 --oof resid_cv/cv_anchor_s8.oof.npy --out v22/cv_tkall_s8.tmp.json && cvmv v22/cv_tkall_s8
}
asset25_evalx() {  # evaluation that also builds the stand table (the gap-family OOF is absent and skipped automatically, only "all" is reported); writes v22/sc_table_all_{full,cons}.parquet
  cd $F/r25 && env "${E_RESID[@]}" RESID_CELLFB=off \
      $PY -W ignore $C/residual_context.py evalx --ref resid_cv/cv_anchor_s8.oof.npy --new all=v22/cv_tkall_s8.oof.npy --keep EGLL-07,LIRF --out v22/cv_eval.tmp.json \
    && test -s v22/sc_table_all_full.parquet && mv v22/cv_eval.tmp.json v22/cv_eval.json
}
repro() {  # reproduction check (non-blocking): recomputed 2025-side outputs vs the json of the original outputs (copies in final/ref/)
  cd $F
  echo "== tk_pool (new | original)"; jq -c '{rows,seg_rows,track_ok,first_vs_feats_gt1s_unanchored,n_check,stop_share}' r25/v22/tk_pool.json ref/20260926-v22-tk_pool.json
  echo "== tk_main (new, includes the final 2026 rows; look only at mismatch and consistency)"; jq -c '{rows,feats_files,rows_in_feats,mismatch,mismatch_by_yr,first_vs_tma_gt1s_unanchored,n_check}' v22/tk_main.json ref/20260926-v22-tk_main.json
  echo "== cv_anchor_s8"; jq -c '{n,gate_cover,all,jan,jul,anch:.anchored.rows}' r25/resid_cv/cv_anchor_s8.json ref/20260926-v21-cv_anchor_s8.json
  echo "== cv_tkall_s8"; jq -c '{max_abs_diff,changed_rows,all,excl_lirf07,jan,gated_red_rel_excl_lirf07}' r25/v22/cv_tkall_s8.json ref/20260926-v22-cv_tkall_s8.json
  echo "== evalx all (the stand correction itself)"; jq -c '.var.all.full_sc_vs_full | {excl_lirf07, n_keys}' r25/v22/cv_eval.json ref/20260926-v22-cv_eval.json
  md5 -r r25/v22/sc_table_all_full.parquet r25/v22/tk_pool.parquet v22/tk_main.parquet
}
pred() {  # pred <tag> [TK]: residual predictions in the final configuration (E_FIN added); output goes to a temporary dir first
  local tag=$1 tk=${2:-}; local tmp=.tmp_pred_$tag
  cd $F && need n2/baseline_used.parquet v22/tk_main.parquet v22/tk_pool.parquet && rm -rf $tmp
  local etk=(RESID_DUMMY_NOTK=1); [ -n "$tk" ] && etk=(RESID_TK=$tk)   # empty = unset (RESID_TK=)
  env "${E_RESID[@]}" "${E_FIN[@]}" "${W7[@]}" "${etk[@]}" \
      $PY -W ignore $C/train_resid.py --storedir n2 --feats feats_tma --predict --seeds 8 --tag $tag --out-dir $tmp \
    && mkdir -p resid_pred && mv $tmp/* resid_pred/ && rmdir $tmp
}
pred_simple() { pred resid_simple; }
pred_full() { pred resid_full all; }
standcorr() {  # applysc (sc_table_all_full) -> merge (conservative cells taken from resid_simple); keep cells follow the February->January / June->July mapping
  cd $F && need v22/sc_table_all_full.parquet && local s=resid_pred/upstanding-firefly
  env "${E_RESID[@]}" "${E_FIN[@]}" "${W7[@]}" \
      $PY -W ignore $C/residual_context.py applysc --sub ${s}_resid_full.parquet --res ${s}_resid_full.res.npy --table v22/sc_table_all_full.parquet \
      --keep $KEEP --out ${s}_resid_full_sc.tmp.parquet || return 1
  mv ${s}_resid_full_sc.tmp.sc.json ${s}_resid_full_sc.sc.json && mv ${s}_resid_full_sc.tmp.parquet ${s}_resid_full_sc.parquet
  env "${E_RESID[@]}" "${E_FIN[@]}" $PY -W ignore $C/residual_context.py merge --base ${s}_resid_simple.parquet --new ${s}_resid_full_sc.parquet \
      --keep $KEEP --out ${s}_resid_merged_sc.tmp.parquet || return 1
  mv ${s}_resid_merged_sc.tmp.merge.json ${s}_resid_merged_sc.merge.json && mv ${s}_resid_merged_sc.tmp.parquet ${s}_resid_merged_sc.parquet
  env "${E_RESID[@]}" $PY -W ignore $C/residual_context.py diff --a ${s}_resid_simple.parquet --b ${s}_resid_merged_sc.parquet --out v22/diff_resid_simple_resid_merged_sc.json || true
  md5 -r resid_pred/*.parquet n2/baseline_used.parquet | tee v22/cand_md5_final.txt
}

# ---------------- tab
tab_feats() {  # tabular feature store from the baseline feature file (n2/feat_3seed.parquet) and ext/xplane-apt
  cd $F && need n2/feat_3seed.parquet && mkdir -p gaptab/out \
    && $PY -u $C/build_tabular_features.py raw n2/feat_3seed.parquet ext/xplane-apt gaptab/out/pf.tmp.parquet && mv gaptab/out/pf.tmp.parquet gaptab/out/pf.parquet
}
tab_arms() {  # tabular arms, 8 seeds (no --ref: with the final data, seed 0 no longer equals the old npz bit for bit)
  cd $F && need n2/feat.parquet gaptab/out/pf.parquet gaptab/out/fcv_all_te2.json n2/oof12/srp_g5_avg.npy
  (cd $C && POLARS_MAX_THREADS=4 $PY -u tabular_term.py $F/n2 $F/gaptab/out/pf.parquet $F/gaptab/out/fcv_all_te2.json $F/gaptab/out/fy26_final_k8.tmp \
      --fam a,b,c,d,e --tekeys te_alst,te_strw,te_tyst,te_alde --threads 4 --lr 0.1 --seeds 0,1,2,3,4,5,6,7) || return 1
  mv gaptab/out/fy26_final_k8.tmp.json gaptab/out/fy26_final_k8.json && mv gaptab/out/fy26_final_k8.tmp.npz gaptab/out/fy26_final_k8.npz
}

log "== chain $MODE start"
case $MODE in
  smoke) run_step align_feat 11 align_feat; run_step smoke_bf 12 smoke_bf; run_step smoke_imp 13 smoke_imp ;;
  adsb)  need $F/_done/align_feat
         run_step ground26 21 ground26; run_step ground25 22 ground25; run_step tc_rows 23 tc_rows; run_step tc_table 24 tc_table
         run_step asset25_pool 31 asset25_pool; run_step asset25_cv_plain 32 asset25_cv_plain; run_step asset25_cv_anchor 33 asset25_cv_anchor
         run_step asset25_cv_tk 34 asset25_cv_tk; run_step asset25_evalx 35 asset25_evalx
         ( repro ) > $F/logs/repro_check.txt 2>&1 || log "repro check failed (non-blocking), see logs/repro_check.txt"
         run_step pred_simple 41 pred_simple; run_step pred_full 42 pred_full; run_step standcorr 43 standcorr ;;
  tab)   need $F/_done/align_feat
         run_step tab_feats 51 tab_feats; run_step tab_arms 52 tab_arms ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
log "== chain $MODE RUN_CHAIN_${MODE}_DONE"
