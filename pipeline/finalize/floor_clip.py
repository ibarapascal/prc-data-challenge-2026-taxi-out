"""floor_clip: lower bound on the predictions before the full version is written out.

Background: the full version has 5 rows below 30 s (minimum 0), coming from the clip(base+res, 0) already present in the gated residual layer.
The residual-layer code is not changed; the lower bound is a post-processing step at the end of the final chain.
Rule (fixed before seeing results):
1. Lower bound = global 0.1 % quantile of TAXITIME_SEC_mvt over the DEP rows of the 2025 training months (training_2025-*.parquet under --train-dir, all 12 files)
   (numpy linear interpolation; if MVT_ID_mvt is present, deduplicate by ID first so that duplicate rows across file boundaries count once; null / non-finite
   values dropped), rounded to an integer (floor(x+0.5)).
2. Only predictions below the bound are raised to it; all other rows keep their value exactly; column names, column order, dtype and row order are unchanged.
3. Counted only, not changed: the number of rows below the bound in --base (the fallback file).
Report: bound value and source (number of files, rows, before / after deduplication, unrounded value, plus the p0.01 / p1 values and the same-definition value
including ARR rows for reference), number of changed rows, minimum before / after, changed rows by month x airport.
Usage (seconds):
  python floor_clip.py --pred resid_pred/..._final_full_patched.parquet --train-dir raw --ranking raw/ranking.parquet \
    --base submit/upstanding-firefly_final_baseline_20261008.parquet --q 0.001 --out resid_pred/..._final_full_floor_patched.parquet --report logs/floor_clip_patched.json
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

IDC = 'MVT_ID_mvt'
Y = 'TAXITIME_SEC_mvt'


def sub(p):
    t = pq.read_table(p)
    yc = [c for c in t.column_names if c != IDC][0]
    return t, yc


def main():
    ap = argparse.ArgumentParser()
    for k in ('pred', 'train_dir', 'ranking', 'base', 'out', 'report'):
        ap.add_argument('--' + k.replace('_', '-'), dest=k, required=True)
    ap.add_argument('--q', type=float, default=0.001)
    a = ap.parse_args()
    rep = dict(rule=__doc__.split('Rule (fixed before seeing results):')[1].split('Report:')[0].strip(), q=a.q)

    # 1. Lower bound
    files = sorted(glob.glob(os.path.join(a.train_dir, 'training_2025-*.parquet')))
    assert files, 'no training files'
    parts = []
    for p in files:
        names = pq.read_schema(p).names
        cols = [c for c in (IDC, 'PHASE_mvt', Y) if c in names]
        t = pl.read_parquet(p, columns=cols)
        if IDC in cols:
            t = t.with_columns(pl.col(IDC).cast(pl.Float64))
        parts.append(t)
    tr = pl.concat(parts, how='diagonal_relaxed')
    src = dict(files=[os.path.basename(p) for p in files], rows_all=tr.height, has_id=IDC in tr.columns,
               phase_counts={str(k): int(v) for k, v in tr.group_by('PHASE_mvt').len().iter_rows()})

    def qv(df, qs):
        v = df[Y].cast(pl.Float64).to_numpy()
        v = v[np.isfinite(v)]
        return int(len(v)), [float(x) for x in np.quantile(v, qs)]

    dep = tr.filter(pl.col('PHASE_mvt') == 'DEP').filter(pl.col(Y).is_not_null())
    src['dep_rows'] = dep.height
    if IDC in dep.columns:
        dep = dep.unique(IDC, keep='first')
        src['dep_rows_dedup'] = dep.height
    n, (q01, q, q1) = qv(dep, [0.0001, a.q, 0.01])
    floor = int(math.floor(q + 0.5))
    alld = tr.filter(pl.col(Y).is_not_null())
    if IDC in alld.columns:
        alld = alld.unique(IDC, keep='first')
    n_all, (q_all,) = qv(alld, [a.q])
    src.update(n_used=n, q_raw=round(q, 4), p0_01=round(q01, 2), p1=round(q1, 2), min=float(dep[Y].min()),
               ref_all_phases=dict(n=n_all, q_raw=round(q_all, 4)))
    rep['floor'] = floor
    rep['floor_source'] = src

    # 2. Apply to the full version
    pt, pyc = sub(a.pred)
    pid = pt.column(IDC).to_numpy().astype(np.float64)
    pv = pt.column(pyc).to_numpy(zero_copy_only=False).astype(np.float64)
    m = pv < floor
    nv = np.where(m, float(floor), pv)
    k = (pl.read_parquet(a.ranking, columns=[IDC, 'ADEP_mvt', 'MVT_TIME_UTC_mvt'])
         .with_columns(pl.col(IDC).cast(pl.Float64), pl.col('MVT_TIME_UTC_mvt').dt.strftime('%m').alias('mo')).select(IDC, 'ADEP_mvt', 'mo'))
    ch = pl.DataFrame({IDC: pid[m], 'old': pv[m]}).join(k, on=IDC, how='left')
    rep['apply'] = dict(rows=int(len(pv)), lt_floor=int(m.sum()), lt30=int((pv < 30).sum()), min_before=float(pv.min()), min_after=float(nv.min()),
                        mean_lift=round(float((floor - pv[m]).mean()), 3) if m.any() else 0.0,
                        changed=[dict(id=float(r[IDC]), cell=f"{r['ADEP_mvt']}-{r['mo']}", old=float(r['old'])) for r in ch.iter_rows(named=True)],
                        unchanged_rows_eq=bool((nv[~m] == pv[~m]).all()))
    cols = [pt.column(c) if c != pyc else pa.array(nv).cast(pt.schema.field(pyc).type) for c in pt.column_names]
    out = pa.Table.from_arrays(cols, schema=pt.schema)
    tmp = a.out + '.tmp'
    pq.write_table(out, tmp)
    os.replace(tmp, a.out)
    rep['out'] = a.out

    # 3. Fallback file: count only
    bt, byc = sub(a.base)
    bv = bt.column(byc).to_numpy(zero_copy_only=False).astype(np.float64)
    rep['base_stats'] = dict(path=a.base, rows=int(len(bv)), lt_floor=int((bv < floor).sum()), lt30=int((bv < 30).sum()), min=float(bv.min()))

    json.dump(rep, open(a.report + '.tmp', 'w'), ensure_ascii=False, indent=1)
    os.replace(a.report + '.tmp', a.report)
    print(json.dumps(dict(floor=floor, floor_source={x: src[x] for x in ('n_used', 'q_raw', 'p0_01', 'p1', 'min')},
                          apply={x: rep['apply'][x] for x in ('lt_floor', 'lt30', 'min_before', 'min_after', 'mean_lift', 'unchanged_rows_eq')},
                          base=rep['base_stats']), ensure_ascii=False))


if __name__ == '__main__':
    main()
