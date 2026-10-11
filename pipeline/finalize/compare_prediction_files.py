"""compare_prediction_files: cell-by-cell comparison of the full version (ADS-B also in February and June) against the fallback file (no ADS-B in February and June), and upstream consistency checks.

Comparison: per month x airport the number of rows, number of changed rows, mean difference (full - base), median and P95 of |difference|, maximum |difference| and the mean of both versions;
also a per-month summary and a single-day summary for 2026-07-01.
Upstream checks (explain why January / July changed or not):
- md5 of the daily January and July 2026 files of feats_tm / feats_tma, old vs new work directory (expected: only 07-01 changed because the 06-30 crop was added); how many rows of 07-01 changed and how many changed their gating.
- whether the 2025 rows of tk_main (non-ranking IDs) are value-for-value identical old vs new (the 2025 side must not depend on 2026 features).
Usage:
  python compare_prediction_files.py --full submit/upstanding-firefly_final_full.parquet --base submit/upstanding-firefly_final_baseline_20261008.parquet \
    --ranking raw/ranking.parquet --old baseline_20261008 --out logs/full_vs_baseline.json
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np
import polars as pl
import pyarrow.parquet as pq

IDC = 'MVT_ID_mvt'


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sub(p):
    t = pq.read_table(p)
    yc = [c for c in t.column_names if c != IDC][0]
    return pl.DataFrame({IDC: t.column(IDC).to_numpy().astype(np.float64), 'y': t.column(yc).to_numpy(zero_copy_only=False).astype(np.float64)})


def stats(g):
    d = (g['full'] - g['base']).to_numpy()
    ad = np.abs(d)
    return dict(n=int(len(d)), changed=int((d != 0).sum()), mean_diff=round(float(d.mean()), 3),
                med_abs=round(float(np.median(ad)), 2), p95_abs=round(float(np.quantile(ad, .95)), 2), max_abs=round(float(ad.max()), 1),
                mean_full=round(float(g['full'].mean()), 2), mean_base=round(float(g['base'].mean()), 2),
                med_abs_changed=round(float(np.median(ad[d != 0])), 2) if (d != 0).any() else 0.0)


def main():
    ap = argparse.ArgumentParser()
    for k in ('full', 'base', 'ranking', 'old', 'out'):
        ap.add_argument('--' + k, required=True)
    a = ap.parse_args()
    rep = dict(files={k: dict(path=getattr(a, k), md5=md5(getattr(a, k))) for k in ('full', 'base')})
    k = (pl.read_parquet(a.ranking, columns=[IDC, 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
         .with_columns(pl.col(IDC).cast(pl.Float64), pl.col('MVT_TIME_UTC_mvt').dt.month().alias('mo'),
                       pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d').alias('date')).select(IDC, 'ADEP_mvt', 'mo', 'date'))
    f = sub(a.full).rename({'y': 'full'}).join(sub(a.base).rename({'y': 'base'}), on=IDC, how='inner').join(k, on=IDC, how='left')
    rep['rows'] = dict(joined=f.height, missing_meta=int(f['mo'].null_count()))
    rep['all'] = stats(f)
    rep['by_month'] = {str(m): stats(g) for (m,), g in f.sort('mo').group_by('mo', maintain_order=True)}
    rep['cells'] = {f'{x}-{m:02d}': stats(g) for (m, x), g in f.sort('mo', 'ADEP_mvt').group_by('mo', 'ADEP_mvt', maintain_order=True)}
    rep['day_2026-07-01'] = stats(f.filter(pl.col('date') == '2026-07-01'))
    rep['jan_jul_excl_0701'] = stats(f.filter(pl.col('mo').is_in([1, 7]) & (pl.col('date') != '2026-07-01')))
    rep['day_2026-06-06'] = stats(f.filter(pl.col('date') == '2026-06-06'))

    # Upstream: md5 of the feats files (January / July)
    fe = {}
    for dn in ('feats_tm', 'feats_tma'):
        diff = []
        for p in sorted(glob.glob(f'{a.old}/{dn}/2026-0[17]-*.parquet')):
            q = f'{dn}/{os.path.basename(p)}'
            if md5(p) != md5(q):
                diff.append(os.path.basename(p))
        fe[dn] = dict(compared=len(glob.glob(f'{a.old}/{dn}/2026-0[17]-*.parquet')), changed_files=diff)
        if '2026-07-01.parquet' in diff:
            o = pl.read_parquet(f'{a.old}/{dn}/2026-07-01.parquet').drop('date')
            n = pl.read_parquet(f'{dn}/2026-07-01.parquet').drop('date')
            j = o.join(n, on=IDC, how='full', suffix='_n', coalesce=True)
            cols = [c for c in o.columns if c != IDC and f'{c}_n' in j.columns]
            ch = j.select(pl.any_horizontal([~pl.col(c).eq_missing(pl.col(f'{c}_n')) for c in cols]).alias('x'))['x']
            fe[dn]['0701'] = dict(rows=j.height, rows_changed=int(ch.sum()),
                                  gate_old=int(j['ab_obs_first'].is_not_null().sum()), gate_new=int(j['ab_obs_first_n'].is_not_null().sum()))
    rep['feats_jan_jul'] = fe

    # Upstream: 2025 rows of tk_main
    ids26 = k[IDC]
    to = pl.read_parquet(f'{a.old}/v22/tk_main.parquet').with_columns(pl.col(IDC).cast(pl.Float64))
    tn = pl.read_parquet('v22/tk_main.parquet').with_columns(pl.col(IDC).cast(pl.Float64))
    o25 = to.filter(~pl.col(IDC).is_in(ids26)).sort(IDC)
    n25 = tn.filter(~pl.col(IDC).is_in(ids26)).sort(IDC)
    same_cols = o25.columns == n25.columns
    rep['tk_main'] = dict(rows_old=to.height, rows_new=tn.height, rows25_old=o25.height, rows25_new=n25.height, same_cols=same_cols,
                          rows25_equal=bool(same_cols and o25.height == n25.height and o25.equals(n25, null_equal=True)),
                          rows26_old=to.height - o25.height, rows26_new=tn.height - n25.height)
    json.dump(rep, open(a.out + '.tmp', 'w'), ensure_ascii=False, indent=1)
    os.replace(a.out + '.tmp', a.out)
    print(json.dumps(dict(all=rep['all'], by_month=rep['by_month'], d0701=rep['day_2026-07-01'], feats=fe, tk=rep['tk_main']), ensure_ascii=False))


if __name__ == '__main__':
    main()
