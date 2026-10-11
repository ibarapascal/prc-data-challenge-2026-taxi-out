#!/bin/bash
# run_8seed_downstream.sh - downstream chain for the 8-seed-baseline candidate. Separate directory final_eightseed/; inputs are referenced read-only from final/ (symlinks) or snapshotted (code, feats); all outputs go to the new directory.
# Usage: bash run_8seed_downstream.sh gate            build the tree + check that the earlier inputs are unchanged -> pred_simple, pred_full -> standcorr -> merge_nostand (resid_merged) -> merge_norevert_full -> driftgate_full (drift_gate_final_full.json)
#        bash run_8seed_downstream.sh final '<cells>'  merge_final_full (revert = verdict of the new driftgate_full) -> partial-day patch (margin 64800, until 06-08T00:00) -> floor q .001 -> check_write_full -> submit/..._eightseed -> comparison with the fallback file and the earlier full version
set -eu -o pipefail
MODE=${1:?gate|final}
M=${PRC_ROOT:?set PRC_ROOT to the working root}; F=$M/final; FW8=$M/final_eightseed; V=$M/eightseed; WORK8=$M/rerun/final_eightseed_baseline
ENSAVG_8SEED=$WORK8/sub/upstanding-firefly_ensavg_8seed.parquet
setup() {
  mkdir -p $FW8/n2 $FW8/v22 $FW8/gaptab/out $FW8/resid_pred $FW8/driftgate/dir $FW8/driftgate/sub $FW8/submit $FW8/logs $FW8/_done
  [ -d $FW8/code ] || { rm -rf $FW8/code.tmp; rsync -a --exclude __pycache__ $F/code/ $FW8/code.tmp/ && mv $FW8/code.tmp $FW8/code; }
  for d in feats_tm feats_tma feats_tmafy; do [ -d $FW8/$d ] || { rm -rf $FW8/$d.tmp; rsync -a $F/$d/ $FW8/$d.tmp/ && mv $FW8/$d.tmp $FW8/$d; }; done
  for d in raw crop ext ref inputs r25 baseline_20261008; do ln -sfn $F/$d $FW8/$d; done
  ln -sfn $F/n2/feat.parquet $FW8/n2/feat.parquet; ln -sfn ../inputs/oof $FW8/n2/oof; ln -sfn ../inputs/oof12 $FW8/n2/oof12
  ln -sfn $F/v22/tk_main.parquet $FW8/v22/tk_main.parquet; ln -sfn $F/v22/tk_main.json $FW8/v22/tk_main.json
  ln -sfn ../r25/v22/tk_pool.parquet $FW8/v22/tk_pool.parquet; ln -sfn ../r25/v22/sc_table_all_full.parquet $FW8/v22/sc_table_all_full.parquet
  ln -sfn $F/gaptab/out/fy26_final_k8.npz $FW8/gaptab/out/fy26_final_k8.npz; ln -sfn $F/gaptab/out/fy26_final_k8.json $FW8/gaptab/out/fy26_final_k8.json
  ln -sfn $F/submit/upstanding-firefly_final_baseline_20261008.parquet $FW8/submit/upstanding-firefly_final_baseline_20261008.parquet
  ln -sfn $F/submit/upstanding-firefly_final_full.parquet $FW8/submit/full_ref.parquet
  # Inputs of the earlier full version (feats, n2, oof, tk, sc_table, tabular terms, fallback file and full version, key scripts): the md5 of each must equal the recorded list
  (cd $FW8 && md5 -r $(awk '{print $2}' $V/v2_inputs_md5.txt | sed 's#^submit/upstanding-firefly_final_full.parquet$#submit/full_ref.parquet#')) \
    | sed 's#submit/full_ref.parquet#submit/upstanding-firefly_final_full.parquet#' | diff - $V/v2_inputs_md5.txt
  test -s $ENSAVG_8SEED
  if [ ! -e $FW8/n2/baseline_used.parquet ]; then cp -c $ENSAVG_8SEED $FW8/n2/baseline_used.tmp.parquet && mv $FW8/n2/baseline_used.tmp.parquet $FW8/n2/baseline_used.parquet; fi
  [ "$(md5 -q $FW8/n2/baseline_used.parquet)" = "$(md5 -q $ENSAVG_8SEED)" ]
  md5 -r $FW8/n2/baseline_used.parquet $ENSAVG_8SEED
}
case $MODE in
  gate)  setup
         bash $V/code/run_stage_chain_8seed.sh resid8
         bash $V/code/run_merge_chain_8seed.sh gate_full ;;
  final) RV=${2?revert cells}
         FULL_TAG=_eightseed PATCH_MARGIN=64800 PATCH_UNTIL=2026-06-08T00:00 FLOOR_Q=0.001 bash $V/code/run_merge_chain_8seed.sh final_full "$RV"
         grep -q '^ "ok": true' $FW8/logs/submission_check_full_eightseed.json || { echo "submission checks did not fully pass"; exit 7; }
         md5 -r $FW8/submit/upstanding-firefly_final_full_eightseed.parquet ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
echo "EIGHTSEED_DOWN_${MODE}_DONE"
