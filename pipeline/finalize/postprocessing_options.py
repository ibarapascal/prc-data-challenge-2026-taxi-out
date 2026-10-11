"""postprocessing_options: stacking of the optional post-processing rules 1 and 2 (appendix A; keys RULE1 / RULE2) on the final file, and comparison (orchestrated by run_postprocessing_options.sh).

Position: the base is the finished file that has already passed the partial-day patch and the floor (_cellfb_off or _eightseed_cellfb_off). The new value of a hit row of
rule 1 / rule 2 depends only on the row's features, not on the original prediction, so floor(D(floor(x))) = floor(D(x)); this is equivalent to inserting the rules
after the partial-day patch and before the floor (as in the 2025 CV definition: both arms get the floor 119 last).
This script only changes hit rows; run_postprocessing_options.sh then replays the floor with floor_clip.py.
Rule (fixed before seeing results; thresholds unchanged, no extra gate added):
- RULE1 = rule 1 = the --apply branch of experiments/not_adopted/lirf_airline_rule.py: candidates LIRF & flt_null=0 & mvt_aobt3 not null & mvt_delay - mvt_aobt3 > 1200;
  group rates from all 2025 training rows (src != 99); an airline or airline x destination qualifies with n >= 30 and an artefact rate (|y - mvt_delay| <= 60) >= .8;
  hits among the 2026 rows (src == 99) are predicted as rint(mvt_delay). The cand / rates expressions are copied verbatim from lirf_airline_rule.py; the row-wise
  hits() is replaced by is_in (null keys are filled with a sentinel first, equivalent to None == None in a Python set).
- RULE2 = rule 2 = global-P version: hits LIRF & flt_null=1 & mvt_delay > 36,000; ms <= 50,000 -> ms, otherwise rint((1 - P) * ms + P * 87,300);
  when applied to 2026, P is the share of 2025 rows of the same kind (LIRF & flt_null=1 & ms > 50,000) with "y >= 86,400 and |y - ms| > 600".
- The hit sets of RULE1 and RULE2 are disjoint (flt_null 0 / 1), which is asserted; their order does not affect the result.
Usage (seconds):
  python postprocessing_options.py apply --base B --feat n2/feat.parquet --ranking raw/ranking.parquet --rule1 on --rule2 on --out O --report R
  python postprocessing_options.py diff --new N --ref R --ranking raw/ranking.parquet --out J
"""
import argparse
import json
import os

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

IDC = 'MVT_ID_mvt'
COLS = ['MVT_ID_mvt', 'ADEP_mvt', 'ADES_mvt', 'al', 'flt_null', 'mvt_delay', 'mvt_aobt3', 'ym', 'src', 'TAXITIME_SEC_mvt']
NUL = '\x00NULL'


def cand(df):  # verbatim the same as cand in lirf_airline_rule.py
    return df.filter((pl.col('ADEP_mvt') == 'LIRF') & (pl.col('flt_null') == 0) & pl.col('mvt_aobt3').is_not_null()
                     & ((pl.col('mvt_delay') - pl.col('mvt_aobt3')) > 1200))


def rates(c, thr, nmin=30):  # verbatim the same as rates in lirf_airline_rule.py; returns lists (keys include the sentinel)
    c = c.with_columns(((pl.col('TAXITIME_SEC_mvt') - pl.col('mvt_delay')).abs() <= 60).cast(pl.Float64).alias('art'))
    g1 = c.group_by('al').agg(pl.col('art').mean().alias('r'), pl.len().alias('n')).filter((pl.col('n') >= nmin) & (pl.col('r') >= thr))
    g2 = c.group_by('al', 'ADES_mvt').agg(pl.col('art').mean().alias('r'), pl.len().alias('n')).filter((pl.col('n') >= nmin) & (pl.col('r') >= thr))
    return g1['al'].to_list(), g2.select(pl.concat_str('al', pl.lit('|'), 'ADES_mvt'))[:, 0].to_list(), g1.sort('al').rows(), g2.sort('al', 'ADES_mvt').rows()


def read_sub(p):
    t = pq.read_table(p)
    yc = [c for c in t.column_names if c != IDC][0]
    return t, yc


def months(ranking):
    return (pl.read_parquet(ranking, columns=[IDC, 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
            .select(pl.col(IDC).cast(pl.Float64), pl.col('ADEP_mvt').cast(pl.String).alias('apt'),
                    pl.col('MVT_TIME_UTC_mvt').dt.month().alias('mo'), pl.col('MVT_TIME_UTC_mvt').dt.strftime('%m-%d').alias('md')))


def apply(a):
    rep = dict(rule=__doc__.split('Rule (fixed before seeing results; thresholds unchanged, no extra gate added):')[1].split('Usage')[0].strip(),
               rule1=a.rule1, rule2=a.rule2, base=a.base, feat=a.feat)
    feat = (pl.scan_parquet(a.feat).select(COLS)
            .with_columns(pl.col(IDC).cast(pl.Float64), *(pl.col(c).cast(pl.String).fill_null(NUL) for c in ('ADEP_mvt', 'ADES_mvt', 'al')))
            .collect())
    tr = feat.filter(pl.col('src') != 99)
    rk = feat.filter(pl.col('src') == 99)
    rep['feat_rows'] = dict(all=feat.height, train=tr.height, ranking=rk.height)
    del feat
    upd = []  # (ID, new value, rule)

    if a.rule1 == 'on':
        s1, s2, g1, g2 = rates(cand(tr), 0.8)
        cr = cand(rk).with_columns(pl.concat_str('al', pl.lit('|'), 'ADES_mvt').alias('k2'))
        h = cr.filter(pl.col('al').is_in(pl.Series(s1, dtype=pl.String)) | pl.col('k2').is_in(pl.Series(s2, dtype=pl.String)))
        md = h['mvt_delay'].cast(pl.Float64).to_numpy()
        upd.append(pl.DataFrame({IDC: h[IDC], 'new': np.rint(md)}).with_columns(pl.lit('RULE1', dtype=pl.String).alias('rule')))
        rep['RULE1'] = dict(cand_train=cand(tr).height, cand_2026=cr.height, hit=h.height, groups_al=[list(r) for r in g1],
                         groups_al_ades=[list(r) for r in g2], null_key_in_groups=bool(NUL in s1 or any(NUL in k for k in s2)),
                         mvt_delay_all_integral=bool((md == np.rint(md)).all()))
    if a.rule2 == 'on':
        t = tr.with_columns(pl.col('mvt_delay').cast(pl.Float64).alias('ms'), pl.col('TAXITIME_SEC_mvt').cast(pl.Float64).alias('y'))
        big = t.filter((pl.col('ADEP_mvt') == 'LIRF') & (pl.col('flt_null') == 1) & (pl.col('ms') > 50000))
        P = float(((big['y'] >= 86400) & ((big['y'] - big['ms']).abs() > 600)).mean()) if big.height else 0.0
        h = rk.with_columns(pl.col('mvt_delay').cast(pl.Float64).alias('ms')).filter(
            (pl.col('ADEP_mvt') == 'LIRF') & (pl.col('flt_null') == 1) & (pl.col('ms') > 36000))
        ms = h['ms'].to_numpy()
        q = np.where(ms <= 50000, ms, (1 - P) * ms + P * 87300)
        upd.append(pl.DataFrame({IDC: h[IDC], 'new': np.rint(q)}).with_columns(pl.lit('RULE2', dtype=pl.String).alias('rule')))
        rep['RULE2'] = dict(P=round(P, 6), n_big_2025=big.height, hit=h.height, hit_le50k=int((ms <= 50000).sum()))

    bt, byc = read_sub(a.base)
    bid = bt.column(IDC).to_numpy().astype(np.float64)
    bv = bt.column(byc).to_numpy(zero_copy_only=False).astype(np.float64)
    if upd:
        U = pl.concat(upd)
        assert U[IDC].n_unique() == U.height, 'RULE1 and RULE2 hit sets overlap or a hit ID is duplicated'
        inb = U.filter(pl.col(IDC).is_in(pl.Series(bid)))
        rep['hits_not_in_base'] = U.height - inb.height
        newv = pl.DataFrame({IDC: bid, 'v': bv}).select(
            pl.col(IDC).replace_strict(inb[IDC], inb['new'], default=pl.col('v'), return_dtype=pl.Float64))[:, 0].to_numpy()
    else:
        U = pl.DataFrame({IDC: [], 'new': [], 'rule': []}, schema={IDC: pl.Float64, 'new': pl.Float64, 'rule': pl.String})
        rep['hits_not_in_base'] = 0
        newv = bv.copy()
    assert (newv == np.rint(newv)).all()
    ch = newv != bv
    rep['changed_vs_base'] = int(ch.sum())
    k = months(a.ranking)
    det = (pl.DataFrame({IDC: bid, 'old': bv, 'new_v': newv}).join(U.select(IDC, 'rule'), on=IDC, how='inner').join(k, on=IDC, how='left')
           .sort('rule', 'mo', IDC))
    rep['hits_by'] = {r: {f"{x['apt']}-{x['mo']:02d}": int(x['len']) for x in g.group_by('apt', 'mo').len().sort('apt', 'mo').iter_rows(named=True)}
                      for (r,), g in det.group_by('rule', maintain_order=True)}
    rep['hits_in_0606_0607'] = int(det.filter(pl.col('md').is_in(['06-06', '06-07'])).height)
    rep['hit_rows'] = [dict(id=float(r[IDC]), rule=r['rule'], cell=f"{r['apt']}-{r['mo']:02d}", day=r['md'], old=r['old'], new=r['new_v'])
                       for r in det.iter_rows(named=True)]
    ytype = bt.schema.field(byc).type
    cols = [bt.column(c) if c != byc else pa.array(newv).cast(ytype) for c in bt.column_names]
    out = pa.Table.from_arrays(cols, schema=bt.schema)
    tmp = a.out + '.tmp'
    pq.write_table(out, tmp)
    rb = pq.read_table(tmp)
    assert (rb.column(byc).to_numpy(zero_copy_only=False).astype(np.float64) == newv).all()
    os.replace(tmp, a.out)
    dump(a.report, rep)
    print(json.dumps({x: rep.get(x) for x in ('rule1', 'rule2', 'hits_not_in_base', 'changed_vs_base', 'hits_by', 'hits_in_0606_0607')}
                     | {'RULE1': {x: rep['RULE1'][x] for x in ('cand_2026', 'hit', 'null_key_in_groups', 'mvt_delay_all_integral')} if 'RULE1' in rep else None,
                        'RULE2': rep.get('RULE2')}, ensure_ascii=False))


def diff(a):
    nt, nyc = read_sub(a.new)
    rt, ryc = read_sub(a.ref)
    n = pl.DataFrame({IDC: nt.column(IDC).to_numpy().astype(np.float64), 'new': nt.column(nyc).to_numpy(zero_copy_only=False).astype(np.float64)})
    r = pl.DataFrame({IDC: rt.column(IDC).to_numpy().astype(np.float64), 'ref': rt.column(ryc).to_numpy(zero_copy_only=False).astype(np.float64)})
    j = n.join(r, on=IDC, how='full', coalesce=True)
    rep = dict(new=a.new, ref=a.ref, rows_new=n.height, rows_ref=r.height, only_new=int(j['ref'].null_count()), only_ref=int(j['new'].null_count()),
               same_row_order=bool(n.height == r.height and (n[IDC] == r[IDC]).all()))
    j = j.drop_nulls().with_columns((pl.col('new') - pl.col('ref')).alias('d')).join(months(a.ranking), on=IDC, how='left')

    def st(g):
        d = g['d'].to_numpy()
        c = d != 0
        return dict(n=int(len(d)), changed=int(c.sum()), mean=round(float(d.mean()), 4), rms=round(float(np.sqrt((d ** 2).mean())), 4),
                    mean_changed=round(float(d[c].mean()), 3) if c.any() else 0.0, max_abs=float(np.abs(d).max()) if len(d) else 0.0)
    rep['all'] = st(j)
    rep['by_month'] = {f"{m:02d}": st(g) for (m,), g in j.sort('mo').group_by('mo', maintain_order=True)}
    rep['by_cell_changed'] = {f"{ap}-{m:02d}": st(g) for (ap, m), g in j.filter(pl.col('d') != 0).sort('apt', 'mo').group_by('apt', 'mo', maintain_order=True)}
    for kk, v in rep['by_cell_changed'].items():
        v['n_cell'] = int(j.filter((pl.col('apt') == kk[:4]) & (pl.col('mo') == int(kk[5:]))).height)
    dump(a.out, rep)
    print(json.dumps({x: rep[x] for x in ('rows_new', 'rows_ref', 'only_new', 'only_ref', 'same_row_order', 'all')}, ensure_ascii=False))


def dump(p, rep):
    with open(p + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
    os.replace(p + '.tmp', p)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p1 = sp.add_parser('apply')
    for k in ('base', 'feat', 'ranking', 'out', 'report'):
        p1.add_argument('--' + k, required=True)
    p1.add_argument('--rule1', choices=('on', 'off'), required=True)
    p1.add_argument('--rule2', choices=('on', 'off'), required=True)
    p2 = sp.add_parser('diff')
    for k in ('new', 'ref', 'ranking', 'out'):
        p2.add_argument('--' + k, required=True)
    a = ap.parse_args()
    apply(a) if a.cmd == 'apply' else diff(a)


if __name__ == '__main__':
    main()
