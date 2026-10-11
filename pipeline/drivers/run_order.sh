#!/bin/bash
# run_order.sh - the order in which the drivers were run to produce the submitted predictions.
#
# This file records the as-run sequence; it is NOT a tested one-click script. Run the steps one at a time
# and read each driver's header first. Several drivers compare MD5 sums of our own intermediate files;
# a rerun on another machine produces slightly different bytes and will stop at those checks
# (see README, "Reproduction caveats"). Directory names, file-name suffixes and environment-variable
# names below are those hard-coded in the drivers. A few short directory and file names (night2/, n2/,
# r25/, v22/, feats_tm*/, gaptab/, fy26_final_k8, v2_inputs_md5.txt) are kept as produced by earlier
# stages because MD5 lists or earlier-phase inputs refer to them; README, "Reproduction caveats", lists them.
#
# Working tree under $PRC_ROOT (as expected by the drivers):
#   .venv/                               Python 3.12 environment (requirements.txt)
#   night2/                              2025 artifacts from earlier phases (README, "Inputs from earlier phases")
#   rerun/code_final_20261008/           <- copy of pipeline/baseline/ (keep the runner/ subdirectory)
#   rerun/final_20261008/                work dir of the 3-seed baseline (step 2)
#   rerun/final_eightseed_baseline/      work dir of the 8-seed baseline (step 7)
#   final/code/                          <- flat copy of: pipeline/baseline/feat_align.py, pipeline/adsb_residual/*.py,
#                                           pipeline/tabular_blend/*.py, pipeline/drift_guard/*.py, pipeline/finalize/*.py,
#                                           pipeline/data/airports.json, experiments/not_adopted/*.py, evaluation/*.py,
#                                           pipeline/drivers/{run_stage_chain,run_merge_chain,run_submission_chain,
#                                           run_postprocessing_options}.sh
#   final/raw/                           links: training_2025-*.parquet, ranking.parquet -> final_ranking.parquet,
#                                           submitting.parquet -> final_submitting.parquet
#   final/crop/<yyyy>/<date>.parquet     cropped ADS-B days (step 1)
#   final/ext/xplane-apt/                merged stand table and runway table (README, "Data")
#   final/inputs/, final/n2/, final/r25/, final/ref/, final/gaptab/out/
#                                        inputs from earlier phases and links, as listed in the header of
#                                           run_stage_chain.sh and in README
#   eightseed/code/                      <- pipeline/drivers/{run_baseline_8seed,run_8seed_downstream,
#                                           run_stage_chain_8seed,run_merge_chain_8seed}.sh,
#                                           pipeline/baseline/verify_baseline_8seed.py
#   eightseed/configs/baseline_8seed.json <- pipeline/baseline/runner/configs/baseline_8seed.json (paths edited)
#   eightseed/v2_inputs_md5.txt          MD5 list checked by run_8seed_downstream.sh (not in this repository)
#   sub/                                 two earlier-phase prediction files read only for report statistics
#                                           by check_submission.py (any files with the same ID column work)
set -euo pipefail
: "${PRC_ROOT:?set PRC_ROOT}"
PY=$PRC_ROOT/.venv/bin/python
F=$PRC_ROOT/final/code
V=$PRC_ROOT/eightseed/code

# 1. ADS-B daily archives: download (manifest.json next to the script lists the files and sizes),
#    crop each day to the ten airport boxes, then place crop/<yyyy>/<date>.parquet under $PRC_ROOT/final/crop/.
#    $PY pipeline/data/download_adsb.py
#    $PY -u pipeline/data/crop_ready_days.py --src <archive dir> --out <crop dir> \
#        --crop-day pipeline/data/crop_day.py --airports pipeline/data/airports.json --workers 2

# 2. Baseline with 3 CatBoost seeds in ensemble B (edit every /path/to/prc-root entry of the config first).
cd "$PRC_ROOT/rerun/code_final_20261008"
$PY -u runner/run_baseline.py --config runner/configs/baseline_3seed.json --to ensavg
#    then, as prepared in our run:
#    cp rerun/final_20261008/sub/upstanding-firefly_ensavg_3seed.parquet  final/n2/baseline_used.parquet
#    ln -s rerun/final_20261008/feat/feat.parquet               final/n2/feat_3seed.parquet

# 3. First pass on the 3-seed baseline. In this pass the D26 line of run_stage_chain.sh listed
#    2026-01 and 2026-07 only (edit that line). According to our change log this line was the only later
#    change to the driver; the exact first-pass revision is not included in this repository.
bash "$F/run_stage_chain.sh" smoke
bash "$F/run_stage_chain.sh" adsb
bash "$F/run_stage_chain.sh" tab

# 4. Fallback file without ADS-B in February and June. Step 9 takes the values of the rows it patches
#    on 2026-06-06/07 from this file. The cell list is the drift-gate verdict printed by the gate step.
bash "$F/run_merge_chain.sh" gate
bash "$F/run_merge_chain.sh" final 'EDDF:07,EDDF:01>=29,LSZH:07'

# 5. Restore the full D26 line (2026-01, -02, -06, -07), drop the completion markers of the steps
#    that depend on the 2026 ADS-B days, and rerun the ADS-B part (2025-side steps are skipped).
rm -f "$PRC_ROOT"/final/_done/{ground26,tc_rows,tc_table,pred_simple,pred_full,standcorr}
bash "$F/run_stage_chain.sh" adsb

# 6. Full version on the 3-seed baseline. Not used for any predicted value of the submission;
#    step 9 requires this file to exist and reports differences against it.
bash "$F/run_merge_chain.sh" gate_full
bash "$F/run_merge_chain.sh" final_full 'EDDF:07,EDDF:01>=29,LSZH:07,EDDF:02,LSZH:06'
cp -p "$PRC_ROOT/final/submit/upstanding-firefly_final_full.parquet" "$PRC_ROOT/final/submit/upstanding-firefly_final_full_first.parquet"
FULL_TAG=_patched FULL_FIRST_MD5=$(md5 -q "$PRC_ROOT/final/submit/upstanding-firefly_final_full_first.parquet") \
  PATCH_MARGIN=64800 PATCH_UNTIL=2026-06-08T00:00 FLOOR_Q=0.001 \
  bash "$F/run_merge_chain.sh" final_full 'EDDF:07,EDDF:01>=29,LSZH:07,EDDF:02,LSZH:06'

# 7. Baseline with 8 CatBoost seeds in ensemble B (seeds 0-2 are taken from the step-2 model cache).
bash "$V/run_baseline_8seed.sh"

# 8. Residual layer on the 8-seed baseline in a separate work dir (final_eightseed/).
bash "$V/run_8seed_downstream.sh" gate
#    As run, `run_8seed_downstream.sh final '<cells>'` was also executed; step 9 does not read its output.

# 9. Submission chain: residual outputs of step 8 with the grid-cell fallback off, stand correction,
#    conservative cells, drift gate, merge with the tabular term, partial-day patch, floor, checks.
#    The cell list of the final call is the drift-gate verdict printed by the gate call.
CELLFB_OFF_CELLFB=off CELLFB_OFF_ROOT=final_eightseed CELLFB_OFF_TAG=_eightseed_cellfb_off bash "$F/run_submission_chain.sh" gate
CELLFB_OFF_CELLFB=off CELLFB_OFF_ROOT=final_eightseed CELLFB_OFF_TAG=_eightseed_cellfb_off bash "$F/run_submission_chain.sh" final 'EDDF:07,EDDF:01>=29,LSZH:07,EDDF:02,LSZH:06'
#    Output: $PRC_ROOT/final_eightseed/submit/upstanding-firefly_final_full_eightseed_cellfb_off.parquet = the submitted file.

# 10. Optional, not part of the submission path: the competition-specific options (docs/appendix-a) with both
#     switches off. In our run this reproduced the step-9 file byte for byte.
#     BASE=eightseed_cellfb_off RULE1=off RULE2=off bash "$F/run_postprocessing_options.sh"
