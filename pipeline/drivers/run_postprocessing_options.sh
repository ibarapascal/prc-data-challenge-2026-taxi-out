#!/bin/bash
# run_postprocessing_options.sh - final assembly with the optional post-processing rules (appendix A):
#   base (final file of the cellfb_off or eightseed_cellfb_off chain) -> rule 1 (RULE1) -> rule 2 (RULE2) -> floor replay -> check_write -> final file -> comparison.
# Not offered as switches (not adopted): the forward-validation switch to the fallback (verdict KEEP in both Feb and Jun) and a baseline-retrain correction (the retrain difference is not a bug).
# Placement: the main flow is merge_final -> partial-day patch (patched rows from the fallback file) -> floor (FLOOR_Q .001 -> 119) -> check_write.
#   Rules 1 and 2 go after the partial-day patch and before the floor (as in the 2025 CV definition: base = chain-proxy OOF, rule applied, both arms
#   get the floor last). Taking the final file as the base is equivalent: the new value of a hit row does not depend on the original prediction,
#   so floor(D(floor(x))) = floor(D(x)).
#   Rule 1 (RULE1, flt_null=0) and rule 2 (RULE2, flt_null=1) have disjoint hit sets, so their order does not matter.
#   Rules and implementation: see the docstring of postprocessing_options.py.
# Usage (cwd arbitrary):
#   BASE=cellfb_off RULE1=on RULE2=on bash run_postprocessing_options.sh
#   BASE=eightseed_cellfb_off RULE1=on RULE2=on bash run_postprocessing_options.sh      (once the 8-seed-baseline line is accepted and its _eightseed_cellfb_off final file exists)
# Switches: BASE=cellfb_off|eightseed_cellfb_off; RULE1=on|off; RULE2=on|off (all must be given explicitly).
#   cellfb_off: final/submit/upstanding-firefly_final_full_cellfb_off.parquet, md5 must be 377dec38...
#   eightseed_cellfb_off: looks for upstanding-firefly_final_full_eightseed_cellfb_off.parquet in final/submit/, then final_eightseed/submit/; none found -> exit 10; md5 differs between the two -> exit 11
# Output: final file final/submit/upstanding-firefly_final_full_asm_<BASE>_rule1<RULE1>_rule2<RULE2>.parquet (refuses to write if it exists, exit 12);
#   intermediates and reports in final/asm/<tag>/ (pre_floor, floor, apply.json, floor_clip.json, submission_check.json, vs_fullref.json, vs_base.json, vs_baseline.json, md5.txt).
# Self-check: with RULE1=off RULE2=off the final file must equal the base row by row (vs_base changed = 0 and same ID set), otherwise exit 77; also reports whether the md5 is byte-identical.
# The fallback file (49b64263...) and the earlier full version (635370b1...) have their md5 checked at the start and at the end; a change exits 13.
set -u -o pipefail
M=${PRC_ROOT:?set PRC_ROOT to the working root}
F=$M/final
C=$F/code
PY=$M/.venv/bin/python
B=${BASE:?BASE=cellfb_off|eightseed_cellfb_off}; X1=${RULE1:?RULE1=on|off}; X11=${RULE2:?RULE2=on|off}
case $X1 in on|off) ;; *) echo "RULE1 must be on|off"; exit 2 ;; esac
case $X11 in on|off) ;; *) echo "RULE2 must be on|off"; exit 2 ;; esac
SAVE=$F/submit/upstanding-firefly_final_baseline_20261008.parquet; SAVE_MD5=49b64263bbdf368c835e77690d082743
SAVE2=$F/submit/upstanding-firefly_final.parquet
FULL_REF=$F/submit/upstanding-firefly_final_full.parquet; FULL_REF_MD5=635370b1ed7063f416f95caf25af0859
CELLFB_OFF_MD5=377dec385933037042136d07d63e38cd
case $B in
  cellfb_off) BF=$F/submit/upstanding-firefly_final_full_cellfb_off.parquet
        [ -e "$BF" ] || { echo "missing base $BF"; exit 10; }
        [ "$(md5 -q "$BF")" = $CELLFB_OFF_MD5 ] || { echo "base md5 is not $CELLFB_OFF_MD5"; exit 11; } ;;
  eightseed_cellfb_off) c1=$F/submit/upstanding-firefly_final_full_eightseed_cellfb_off.parquet; c2=$M/final_eightseed/submit/upstanding-firefly_final_full_eightseed_cellfb_off.parquet
        if [ -e "$c1" ] && [ -e "$c2" ]; then [ "$(md5 -q "$c1")" = "$(md5 -q "$c2")" ] || { echo "the two _eightseed_cellfb_off files differ in md5"; exit 11; }; BF=$c1
        elif [ -e "$c1" ]; then BF=$c1; elif [ -e "$c2" ]; then BF=$c2
        else echo "_eightseed_cellfb_off final file not found ($c1 / ${c2})"; exit 10; fi ;;
  *) echo "BASE must be cellfb_off|eightseed_cellfb_off"; exit 2 ;;
esac
TAG=asm_${B}_rule1${X1}_rule2${X11}
FINAL=submit/upstanding-firefly_final_full_$TAG.parquet
W=asm/$TAG
cd "$F" || exit 3
[ -e "$FINAL" ] && { echo "refusing to write: $FINAL already exists"; exit 12; }
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONUNBUFFERED=1
mkdir -p "$W"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$W/chain.log"; }
step() { local n=$1 rc=$2; shift 2; log "start $n"; ( "$@" ) > "$W/$n.log" 2>&1 || { log "FAIL $n -> exit $rc"; tail -5 "$W/$n.log"; exit "$rc"; }; log "done $n"; }
guard() { for p in "$SAVE:$SAVE_MD5" "$SAVE2:$SAVE_MD5" "$FULL_REF:$FULL_REF_MD5"; do
  [ "$(md5 -q "${p%%:*}")" = "${p##*:}" ] || { log "protected file md5 changed: ${p%%:*}"; exit 13; }; done; log "guard ok (fallback file x2 ${SAVE_MD5}, full version ${FULL_REF_MD5})"; }

log "== asm start BASE=$B ($BF md5 $(md5 -q "$BF")) RULE1=$X1 RULE2=$X11 tag=$TAG"
guard
step apply 21 $PY -u $C/postprocessing_options.py apply --base "$BF" --feat n2/feat.parquet --ranking raw/ranking.parquet --rule1 $X1 --rule2 $X11 \
     --out $W/pre_floor.parquet --report $W/apply.json
tail -1 $W/apply.log | tee -a $W/chain.log
step floor 22 $PY -u $C/floor_clip.py --pred $W/pre_floor.parquet --train-dir raw --ranking raw/ranking.parquet --base "$SAVE" --q 0.001 \
     --out $W/floor.parquet --report $W/floor_clip.json
fl=$(jq -r .floor $W/floor_clip.json); log "floor=$fl lt_floor=$(jq -r .apply.lt_floor $W/floor_clip.json)"
[ "$fl" = 119 ] || { log "floor is not 119 (${fl}) -> exit 23"; exit 23; }
step check_write 24 $PY -u $C/check_submission.py --pred $W/floor.parquet --tmpl raw/submitting.parquet --earlypred $M/sub/upstanding-firefly_v22.parquet \
     --basefile n2/baseline_used.parquet --oldbase $M/sub/upstanding-firefly_v8.parquet --ranking raw/ranking.parquet --train-dir raw \
     --out submit/upstanding-firefly_final_full_$TAG.tmp.parquet --report $W/submission_check.json
log "checks ok=$(jq -r .ok $W/submission_check.json) fail=$(jq -c .fail $W/submission_check.json) range=$(jq -c .range $W/submission_check.json)"
[ -e "$FINAL" ] && { log "refusing to write: $FINAL already exists"; exit 12; }
mv submit/upstanding-firefly_final_full_$TAG.tmp.parquet "$FINAL" || exit 25
md5 -r "$FINAL" $W/pre_floor.parquet $W/floor.parquet "$BF" "$FULL_REF" "$SAVE" $C/postprocessing_options.py $C/run_postprocessing_options.sh | tee $W/md5.txt | tee -a $W/chain.log
step vs_fullref 26 $PY -u $C/postprocessing_options.py diff --new "$FINAL" --ref "$FULL_REF" --ranking raw/ranking.parquet --out $W/vs_fullref.json
step vs_base 27 $PY -u $C/postprocessing_options.py diff --new "$FINAL" --ref "$BF" --ranking raw/ranking.parquet --out $W/vs_base.json
step vs_baseline 28 $PY -u $C/postprocessing_options.py diff --new "$FINAL" --ref "$SAVE" --ranking raw/ranking.parquet --out $W/vs_baseline.json
for k in vs_fullref vs_base vs_baseline; do log "$k $(jq -c '{only_new,only_ref,all}' $W/$k.json)"; done
if [ $X1 = off ] && [ $X11 = off ]; then
  ch=$(jq -r '.all.changed' $W/vs_base.json); on=$(jq -r '.only_new + .only_ref' $W/vs_base.json)
  same=$([ "$(md5 -q "$FINAL")" = "$(md5 -q "$BF")" ] && echo BYTE_EQUAL || echo BYTE_DIFF)
  log "selfcheck vs base: changed=$ch id_mismatch=$on md5=$same"
  [ "$ch" = 0 ] && [ "$on" = 0 ] || { log "self-check failed -> exit 77"; exit 77; }
fi
guard
log "== asm RUN_ASM_DONE $TAG $(md5 -q "$FINAL")"
