"""Rule 1 (appendix A, key RULE1): among LIRF candidate rows that have a network-manager (NM) match and an AOBT more than 20 minutes later than scheduled,
the BLOCK ~ SCHED artefact rate is determined by the airline. Hits are rows of airline or airline x destination groups whose artefact rate is >= the threshold
with n >= 30; their prediction is replaced by MVT - SCHED (= mvt_delay).

Validation: leave-one-month-out on January / July 2025 (group rates use only the other 11 months), stacked on the predictions of an earlier residual iteration
(baseline OOF + gated ADS-B residual), looking at the SSE on non-extreme rows.
Application: group rates from all of 2025; hits among the 2026 ranking rows overwrite the prediction.
Usage: python lirf_airline_rule.py --oof resid_eval_26d.oof.npy [--thr 0.8] [--apply sub/upstanding-firefly_v13.parquet --out sub/x.parquet]
"""
import argparse
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, 'code')
import train_resid as T

COLS = ['MVT_ID_mvt', 'ADEP_mvt', 'ADES_mvt', 'al', 'flt_null', 'mvt_delay', 'mvt_aobt3', 'ym', 'src', 'TAXITIME_SEC_mvt']


def cand(df):
    return df.filter((pl.col('ADEP_mvt') == 'LIRF') & (pl.col('flt_null') == 0) & pl.col('mvt_aobt3').is_not_null()
                     & ((pl.col('mvt_delay') - pl.col('mvt_aobt3')) > 1200))


def rates(c, thr, nmin=30):
    c = c.with_columns(((pl.col('TAXITIME_SEC_mvt') - pl.col('mvt_delay')).abs() <= 60).cast(pl.Float64).alias('art'))
    g1 = c.group_by('al').agg(pl.col('art').mean().alias('r'), pl.len().alias('n')).filter((pl.col('n') >= nmin) & (pl.col('r') >= thr))
    g2 = c.group_by('al', 'ADES_mvt').agg(pl.col('art').mean().alias('r'), pl.len().alias('n')).filter((pl.col('n') >= nmin) & (pl.col('r') >= thr))
    return set(g1['al'].to_list()), set(map(tuple, g2.select('al', 'ADES_mvt').rows()))


def hits(df, s1, s2):
    return np.array([(a in s1) or ((a, d) in s2) for a, d in df.select('al', 'ADES_mvt').rows()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', required=True)
    ap.add_argument('--thr', type=float, default=0.8)
    ap.add_argument('--apply')
    ap.add_argument('--out')
    a = ap.parse_args()
    feat = pl.read_parquet('n2/feat.parquet', columns=COLS)
    tr_all = feat.filter(pl.col('src') != 99)
    # Validation: leave-one-month-out on the predictions of the earlier residual iteration
    v = T.load_train('n2', 'feats')
    res = np.load(a.oof)
    p13 = v['base'].to_numpy() + res * v['gate'].to_numpy()
    y = v['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    nA = ~v['isA'].to_numpy()
    vv = v.select('MVT_ID_mvt', 'ym').join(feat.select([c for c in COLS if c not in ('ym', 'src', 'TAXITIME_SEC_mvt')]),
                                            on='MVT_ID_mvt', how='left')
    assert vv['MVT_ID_mvt'].to_list() == v['MVT_ID_mvt'].to_list()
    new = p13.copy()
    out = {}
    for mo in (1, 7):
        s1, s2 = rates(cand(tr_all.filter(pl.col('ym') != 202500 + mo)), a.thr)
        m = (vv['ym'] == 202500 + mo).to_numpy()
        cm = cand(vv.filter(pl.Series(m)).with_columns(pl.lit(0).alias('TAXITIME_SEC_mvt')))
        ids = set(np.array(cm['MVT_ID_mvt'].to_list())[hits(cm, s1, s2)]) if cm.height else set()
        h = np.isin(v['MVT_ID_mvt'].to_numpy(), list(ids))
        new[h] = vv['mvt_delay'].to_numpy()[h]
        out[mo] = dict(hit=int(h.sum()), d_sse=float(((new - y) ** 2)[h & nA].sum() - ((p13 - y) ** 2)[h & nA].sum()))
    tot_sse = float(((p13 - y) ** 2)[nA].sum())
    red = -sum(o['d_sse'] for o in out.values()) / tot_sse
    print(json.dumps(dict(thr=a.thr, per_month=out, rel_sse_red_on_train_days=round(red, 5),
                          train_days=int(v['date'].n_unique())), indent=1))
    if a.apply:
        s1, s2 = rates(cand(tr_all), a.thr)
        rk = feat.filter(pl.col('src') == 99)
        cr = cand(rk)
        hid = np.array(cr['MVT_ID_mvt'].to_list())[hits(cr, s1, s2)] if cr.height else np.array([])
        sub = pl.read_parquet(a.apply)
        repl = rk.filter(pl.col('MVT_ID_mvt').is_in(hid.tolist())).select('MVT_ID_mvt', pl.col('mvt_delay').round().cast(pl.Int32).alias('r'))
        sub2 = sub.join(repl, on='MVT_ID_mvt', how='left').with_columns(
            pl.coalesce('r', 'TAXITIME_SEC_mvt').alias('TAXITIME_SEC_mvt')).drop('r')
        d = (sub2['TAXITIME_SEC_mvt'] - sub['TAXITIME_SEC_mvt']).cast(pl.Float64)
        print('2026 hits', len(hid), 'changed', int((d != 0).sum()), 'mean d', float(d.mean()), 'mean d2 over all rows', float((d ** 2).mean()))
        if a.out:
            sub2.write_parquet(a.out)


if __name__ == '__main__':
    main()
