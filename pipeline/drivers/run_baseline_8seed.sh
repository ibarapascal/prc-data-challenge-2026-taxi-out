#!/bin/bash
# run_baseline_8seed.sh -- 2026-side 8-seed baseline (only the CatBoost seeds of ensemble B go from 3 to 8) recomputed in a separate work directory rerun/final_eightseed_baseline; the 3-seed baseline run (rerun/final_20261008, RUN3SEED below) is read-only.
# Copy (clonefile) the ensemble-B feature store, the reused ensemble-B members, the ensemble-A files and cb_cache of the 3-seed run -> run_baseline.py runs only ensb_cat_taxi, ensb_cat_delta, ensb_assemble, ensavg -> verify_baseline_8seed.py checks.
# Can be rerun from scratch: files already copied are skipped; run_baseline.py skips finished stages via _done; CatBoost parts hit cb_cache.
set -eu -o pipefail
M=${PRC_ROOT:?set PRC_ROOT to the working root}; V=$M/eightseed; PY=$M/.venv/bin/python
RUN3SEED=$M/rerun/final_20261008; W=$M/rerun/final_eightseed_baseline; CODE=$M/rerun/code_final_20261008
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONUNBUFFERED=1
mkdir -p "$W/ensa" "$W/ensb" "$W/sub" "$W/models" "$W/report"
[ -e "$W/raw" ] || cp -a "$RUN3SEED/raw" "$W/raw"          # symlinks are copied as they are
[ -e "$W/ext" ] || cp -a "$RUN3SEED/ext" "$W/ext"
if [ ! -d "$W/models/cb_cache" ]; then rm -rf "$W/models/cb_cache.tmp"; cp -cR "$RUN3SEED/models/cb_cache" "$W/models/cb_cache.tmp"; mv "$W/models/cb_cache.tmp" "$W/models/cb_cache"; fi
for f in feat.parquet results.tsv full_srp_g5_delta.npy full_anc_iobt.npy full_apt_delta.npy; do [ -e "$W/ensb/$f" ] || cp -c "$RUN3SEED/ensb/$f" "$W/ensb/$f"; done
for f in full_anc_iobt.npy full_apt_delta.npy; do [ -e "$W/ensa/$f" ] || cp -c "$RUN3SEED/ensa/$f" "$W/ensa/$f"; done
[ -e "$W/sub/upstanding-firefly_ensa_final.parquet" ] || cp -c "$RUN3SEED/sub/upstanding-firefly_ensa_final.parquet" "$W/sub/"
for f in full_cat_delta.npy cat_pt_ensb_3seed.npy cat_pd_ensb_3seed.npy; do
  if [ -e "$W/ensb/$f" ] && [ ! -e "$W/_done/ensb_assemble.json" ]; then echo "the new directory must not contain $f of RUN3SEED"; exit 3; fi
done
h=$(md5 -q "$W/ensb/feat.parquet"); [ "$h" = 6a65e24bd23c0505d1adf251609021b3 ] || { echo "ensb feat md5 $h"; exit 4; }
h=$(md5 -q "$W/sub/upstanding-firefly_ensa_final.parquet"); h1=$(md5 -q "$RUN3SEED/sub/upstanding-firefly_ensa_final.parquet"); [ "$h" = "$h1" ] || { echo "ensa_final md5"; exit 4; }
cd "$CODE"
$PY -u runner/run_baseline.py --config $V/configs/baseline_8seed.json --stages ensb_cat_taxi,ensb_cat_delta,ensb_assemble,ensavg
$PY -u $V/code/verify_baseline_8seed.py --run3seed "$RUN3SEED" --w "$W" --out "$W/report/ensavg_8seed_verify.json"
md5 -r "$W/sub/"*.parquet
echo ENSAVG_8SEED_DONE
