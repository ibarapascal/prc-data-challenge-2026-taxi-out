"""check_submission: final checks before submission, and writing of the final file in the layout of the official template.

Reads the merged output of the merge step, the official template final_submitting.parquet, an earlier-phase prediction file (earlypred), the retrained baseline (basefile), an older
baseline file (oldbase), final_ranking and the 2025 training-month labels. The last three files are used only for the report comparisons.
Hard checks (any failure -> nothing is written, exit 2): the template has only an ID plus one prediction column; row count and ID set equal to the template, no
duplicates on either side; no null / NaN / inf in the predictions; column names, column order and dtype of the written file equal to the template (compared without
schema metadata), row order = template row order, values equal after reading the file back.
Report (not judged): value quantiles, number of rows < 30 s and > 20,000 s; RMS / correlation / mean difference against earlypred (common IDs) and basefile (overall, by month);
mean comparison by month x airport (final, basefile, earlypred on common IDs, basefile - oldbase = retrain shift, mean DEP label of the same 2025 month) with flags for unusual cells.
Usage (seconds):
  python check_submission.py --pred P --tmpl raw/submitting.parquet --earlypred EARLYPRED --basefile BASEFILE --oldbase OLDBASE --ranking raw/ranking.parquet \
    --train-dir raw --out submit/upstanding-firefly_final.parquet --report logs/submission_check.json
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

IDC = 'MVT_ID_mvt'
MONTHS = (1, 2, 6, 7)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sub(p):
    t = pq.read_table(p)
    yc = [c for c in t.column_names if c != IDC][0]
    return pl.DataFrame({IDC: t.column(IDC).to_numpy().astype(np.float64), 'y': t.column(yc).to_numpy().astype(np.float64)})


def cmp(a, b):
    d = a - b
    return dict(n=int(len(d)), mean=round(float(d.mean()), 3), rms=round(float(np.sqrt((d ** 2).mean())), 3),
                max_abs=round(float(np.abs(d).max()), 1), corr=round(float(np.corrcoef(a, b)[0, 1]), 6)) if len(d) else None


def main():
    ap = argparse.ArgumentParser()
    for k in ('pred', 'tmpl', 'earlypred', 'basefile', 'oldbase', 'ranking', 'train_dir', 'out', 'report'):
        ap.add_argument('--' + k.replace('_', '-'), dest=k, required=True)
    a = ap.parse_args()
    rep = dict(files={k: dict(path=getattr(a, k), md5=md5(getattr(a, k))) for k in ('pred', 'tmpl', 'earlypred', 'basefile', 'oldbase')}, hard={}, fail=[])

    def hard(name, ok, **info):
        rep['hard'][name] = dict(ok=bool(ok), **info)
        if not ok:
            rep['fail'].append(name)

    # ---- template
    tm = pq.read_table(a.tmpl)
    tsch = tm.schema.remove_metadata()
    yc = [n for n in tsch.names if n != IDC]
    rep['tmpl'] = dict(rows=tm.num_rows, names=tsch.names, types=[str(f.type) for f in tsch],
                       null_counts={c: tm.column(c).null_count for c in tsch.names}, metadata_keys=[k.decode() for k in (tm.schema.metadata or {})])
    hard('tmpl_two_cols', IDC in tsch.names and len(yc) == 1, names=tsch.names)
    if rep['fail']:
        return done(a, rep)
    yc = yc[0]
    tid = tm.column(IDC).to_numpy().astype(np.float64)

    # ---- prediction
    pt = pq.read_table(a.pred)
    rep['pred_schema'] = dict(names=pt.schema.names, types=[str(f.type) for f in pt.schema])
    pyc = [c for c in pt.column_names if c != IDC][0]
    pid = pt.column(IDC).to_numpy().astype(np.float64)
    pv = pt.column(pyc).to_numpy(zero_copy_only=False).astype(np.float64)
    hard('rows', len(pid) == len(tid), pred=int(len(pid)), tmpl=int(len(tid)))
    hard('no_dup', len(np.unique(pid)) == len(pid) and len(np.unique(tid)) == len(tid),
         pred_unique=int(len(np.unique(pid))), tmpl_unique=int(len(np.unique(tid))))
    hard('id_set_eq', set(pid.tolist()) == set(tid.tolist()),
         only_pred=int(len(set(pid.tolist()) - set(tid.tolist()))), only_tmpl=int(len(set(tid.tolist()) - set(pid.tolist()))))
    hard('finite', pt.column(pyc).null_count == 0 and bool(np.isfinite(pv).all()), nulls=pt.column(pyc).null_count,
         nonfinite=int((~np.isfinite(pv)).sum()))
    hard('pred_row_order_eq_tmpl', True, info=bool(len(pid) == len(tid) and (pid == tid).all()))  # only recorded; the output is reordered to the template row order
    if rep['fail']:
        return done(a, rep)

    # ---- write in template row order and schema
    o = np.argsort(pid)
    vals = pv[o[np.searchsorted(pid[o], tid)]]
    ytype = tsch.field(yc).type
    if pa.types.is_integer(ytype):
        assert (vals == np.rint(vals)).all()
    cols = {IDC: tm.column(IDC).combine_chunks(), yc: pa.array(vals).cast(ytype)}
    out = pa.Table.from_arrays([cols[n] for n in tsch.names], schema=tsch)
    tmp = a.out + '.tmp'
    pq.write_table(out, tmp)
    rb = pq.read_table(tmp)
    hard('out_schema_eq_tmpl', rb.schema.remove_metadata().equals(tsch), out=str(rb.schema.remove_metadata()))
    hard('out_ids_eq_tmpl_order', bool((rb.column(IDC).to_numpy().astype(np.float64) == tid).all()))
    hard('out_vals_eq_pred', bool((rb.column(yc).to_numpy().astype(np.float64) == vals).all()))
    if rep['fail']:
        os.remove(tmp)
        return done(a, rep)
    os.replace(tmp, a.out)
    rep['out'] = dict(path=a.out, md5=md5(a.out), bytes=os.path.getsize(a.out))

    # ---- value range
    q = np.quantile(vals, [0, .001, .01, .5, .99, .999, 1])
    rep['range'] = dict(q=dict(zip(['min', 'p0.1', 'p1', 'p50', 'p99', 'p99.9', 'max'], [round(float(x), 1) for x in q])),
                        mean=round(float(vals.mean()), 2), lt30=int((vals < 30).sum()), gt20000=int((vals > 20000).sum()))

    # ---- comparison tables
    k = pl.read_parquet(a.ranking, columns=[IDC, 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
    k = k.with_columns(pl.col(IDC).cast(pl.Float64), pl.col('MVT_TIME_UTC_mvt').dt.month().alias('mo')).select(IDC, 'ADEP_mvt', 'mo')
    f = pl.DataFrame({IDC: tid, 'fin': vals}).join(k, on=IDC, how='left')
    hard('ranking_cover', f['ADEP_mvt'].null_count() == 0, missing=int(f['ADEP_mvt'].null_count()))
    f = (f.join(sub(a.basefile).rename({'y': 'basefile'}), on=IDC, how='left').join(sub(a.earlypred).rename({'y': 'earlypred'}), on=IDC, how='left')
         .join(sub(a.oldbase).rename({'y': 'oldbase'}), on=IDC, how='left'))
    hard('basefile_cover', f['basefile'].null_count() == 0, missing=int(f['basefile'].null_count()))
    c22 = f.filter(pl.col('earlypred').is_not_null())
    rep['vs_earlypred_common'] = dict(all=cmp(c22['fin'].to_numpy(), c22['earlypred'].to_numpy()),
                                by_month={str(m): cmp(g['fin'].to_numpy(), g['earlypred'].to_numpy()) for (m,), g in c22.group_by('mo', maintain_order=True)})
    rep['vs_basefile'] = dict(all=cmp(f['fin'].to_numpy(), f['basefile'].to_numpy()),
                         by_month={str(m): cmp(g['fin'].to_numpy(), g['basefile'].to_numpy()) for (m,), g in f.group_by('mo', maintain_order=True)})
    c8 = f.filter(pl.col('oldbase').is_not_null())
    rep['basefile_vs_oldbase_common'] = cmp(c8['basefile'].to_numpy(), c8['oldbase'].to_numpy())

    lab = []
    for p in sorted(glob.glob(os.path.join(a.train_dir, 'training_2025-*.parquet'))):
        t = pl.read_parquet(p, columns=['ADEP_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt', 'TAXITIME_SEC_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
        t = t.with_columns(pl.col('MVT_TIME_UTC_mvt').dt.month().alias('mo')).filter(pl.col('mo').is_in(MONTHS))
        if t.height:
            lab.append(t.group_by('ADEP_mvt', 'mo').agg(pl.col('TAXITIME_SEC_mvt').sum().alias('s25'), pl.len().alias('n25')))
    lab = pl.concat(lab).group_by('ADEP_mvt', 'mo').agg(pl.col('s25').sum(), pl.col('n25').sum()).with_columns((pl.col('s25') / pl.col('n25')).alias('lab25')).drop('s25')  # rows on month boundaries across files are merged before dividing

    g = (f.group_by('mo', 'ADEP_mvt').agg(
        pl.len().alias('n'), pl.col('fin').mean().alias('fin'), pl.col('fin').std().alias('fin_sd'), pl.col('fin').min().alias('fin_min'),
        pl.col('basefile').mean().alias('basefile'), (pl.col('fin') - pl.col('basefile')).mean().alias('d_basefile'),
        ((pl.col('fin') - pl.col('basefile')) ** 2).mean().sqrt().alias('rms_basefile'),
        pl.col('earlypred').count().alias('n_earlypred'), (pl.col('fin') - pl.col('earlypred')).mean().alias('d_earlypred'),
        ((pl.col('fin') - pl.col('earlypred')) ** 2).mean().sqrt().alias('rms_earlypred'),
        (pl.col('basefile') - pl.col('oldbase')).mean().alias('run3seed_shift'))
         .join(lab, on=['ADEP_mvt', 'mo'], how='left').with_columns((pl.col('fin') / pl.col('lab25')).alias('ratio_lab25'))
         .sort('mo', 'ADEP_mvt'))
    cells, flags = {}, {}
    for r in g.iter_rows(named=True):
        kk = f"{r['ADEP_mvt']}-{r['mo']:02d}"
        cells[kk] = {c: (round(v, 2) if isinstance(v, float) else v) for c, v in r.items() if c not in ('ADEP_mvt', 'mo')}
        fl = []
        if r['d_earlypred'] is not None and abs(r['d_earlypred']) >= 5:
            fl.append(f"final-earlypred mean {r['d_earlypred']:+.1f}s")
        if abs(r['d_basefile']) >= 10:
            fl.append(f"final-basefile mean {r['d_basefile']:+.1f}s")
        if r['ratio_lab25'] is not None and not (0.8 <= r['ratio_lab25'] <= 1.2):
            fl.append(f"final / 2025 same-month label {r['ratio_lab25']:.2f}")
        if r['fin_min'] < 30:
            fl.append(f"min {r['fin_min']:.0f}")
        if fl:
            flags[kk] = fl
    rep['cells'] = cells
    rep['flags'] = dict(rule='for the report, not a criterion: abs(final-earlypred) >= 5 s (common IDs), abs(final-basefile) >= 10 s, final / 2025 same-month DEP label mean outside [0.8, 1.2], min < 30', cells=flags)
    rep['month_means'] = {str(m): dict(n=gg.height, fin=round(gg['fin'].mean(), 2), basefile=round(gg['basefile'].mean(), 2))
                          for (m,), gg in f.group_by('mo', maintain_order=True)}
    return done(a, rep)


def done(a, rep):
    rep['ok'] = not rep['fail']
    with open(a.report, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: rep.get(k) for k in ('ok', 'fail', 'hard', 'out', 'range', 'vs_earlypred_common', 'vs_basefile')}, ensure_ascii=False, default=str))
    sys.exit(0 if rep['ok'] else 2)


if __name__ == '__main__':
    main()
