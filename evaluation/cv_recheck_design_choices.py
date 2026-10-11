"""cv_recheck_design_choices: re-judges design choices on CV by pre-registered rules (read-only, no training).

Decision rule (fixed before seeing results):
- Default rule: delta = SSE(on) - SSE(off) (all rows, including isA, the competition metric); 5 day folds (day_folds of residual_context.py, seed 0, 62 days of 2025-01/07);
  "keep / include" <=> delta <= 0 and the number of folds with delta_k < 0 is >= 4 (a fold without hit rows has delta_k = 0, which does not count as same-direction). The day-block paired SE is reported, not used for the decision.
- Averaged baseline: on = (ens_v5b + ens_v6)/2, off = ens_v5b; both arms get the same non-gated transform (clip(rint, 30) outside B rows, B rows unchanged); rows = the 344,419 rows of load_train.
  Alignment self-check: the RMSE of the on arm must equal the averaged-baseline RMSE without lower bound, 310.062496 (a difference > 1e-4 reports align_ok=false). Reported only: against ens_v6.
- LIRF airline artefact rule (rule 1 of appendix A, key RULE1) and LIRF extreme-row rule (rule 2, key RULE2) (exclusion side): the base is the chain proxy OOF (chain basis, residuals with CELLFB=off,
  the three reverted cells as in the full version), and both arms get the lower bound 119 at the end; on = rule applied, off = not applied.
  Rule 1 as in lirf_airline_rule.py (threshold .8, n >= 30, group rates estimated leave-one-month-out on the other 11 months, hit rows = rint(mvt_delay));
  rule 2, global-P version: LIRF & flt_null=1 & ms (mvt_delay) > 36,000; ms <= 50,000 -> ms, otherwise rint((1-P)*ms + P*87,300),
  P = share of rows with "y >= 86,400 and |y - ms| > 600" among the rows of the same kind in the other 11 months that have ms > 50,000.
Usage: any cwd; environment as for floor_oof_check.py (E_RESID + RESID_CELLFB=off)
  python cv_recheck_design_choices.py --r25 final/r25 --out final/logs/cellfb_off_cv_check.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

FLOOR = 119
ENSAVG_REF = 310.062496
RULE1_COLS = ['MVT_ID_mvt', 'ADEP_mvt', 'ADES_mvt', 'al', 'flt_null', 'mvt_delay', 'mvt_aobt3', 'ym', 'src', 'TAXITIME_SEC_mvt']


def nongated(b, B):
    return np.where(B, b, np.clip(np.rint(b), 30, None))


def judge(p_on, p_off, y, fd, dv):
    e_on, e_off = (p_on - y) ** 2, (p_off - y) ** 2
    d = e_on - e_off
    n = len(d)
    folds = [float(d[fd == k].sum()) for k in range(5)]
    days = sorted(set(dv))
    di = {x: i for i, x in enumerate(days)}
    ds = np.bincount(np.array([di[x] for x in dv]), weights=d, minlength=len(days))
    se_mse = float(np.std(ds, ddof=1) * np.sqrt(len(days)) / n)
    n_imp = sum(1 for x in folds if x < 0)
    tot = float(d.sum())
    return dict(n=int(n), n_changed=int((p_on != p_off).sum()), rmse_on=round(float(np.sqrt(e_on.mean())), 6),
                rmse_off=round(float(np.sqrt(e_off.mean())), 6), d_rmse=round(float(np.sqrt(e_on.mean()) - np.sqrt(e_off.mean())), 6),
                d_sse=tot, d_mse=round(tot / n, 4), se_mse_dayblock=round(se_mse, 4), fold_d_sse=folds,
                fold_n_changed=[int(((p_on != p_off) & (fd == k)).sum()) for k in range(5)],
                n_fold_improve=n_imp, keep=bool(tot <= 0 and n_imp >= 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r25', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--keep', default='EGLL-07,LIRF')
    a = ap.parse_args()
    out_path = str(Path(a.out).resolve())
    code = os.environ['RESID_CODE_DIR']
    os.chdir(a.r25)
    sys.path.insert(0, code)
    import residual_context as V
    T = V.T
    rep = dict(rule=__doc__.split('Decision rule (fixed before seeing results):')[1].split('Usage:')[0].strip(),
               code=str(T.__file__), feats=V.FE, cellfb=T.CELLFB, keep=a.keep)

    v = T.load_train('n2', V.FE)
    y, b, g, nA, av, dv, mv = V.base_arrays(v)
    n = v.height
    fd = V.day_folds(dv)
    ok = np.isfinite(y)
    assert ok.all(), int((~ok).sum())
    B = ((v['flt_null'] == 1) & (v['ADEP_mvt'] == 'LIRF') & (v['mvt_delay'] > 3600)).fill_null(False).to_numpy()
    ids = v['MVT_ID_mvt'].to_numpy()

    # ---- averaged baseline: avg(ens_v5b, ens_v6) vs ens_v5b (rows matched back to load_train by MVT_ID)
    feat = pl.read_parquet('n2/feat.parquet', columns=RULE1_COLS)
    parts = []
    for mo in (1, 7):
        f = feat.filter((pl.col('src') != 99) & (pl.col('ym') == 202500 + mo)).select('MVT_ID_mvt')
        b5 = np.load(Path('n2') / 'oof' / f'ens_v5b_{mo}.npy')
        b6 = np.load(Path('n2') / 'oof' / f'ens_v6_{mo}.npy')
        assert len(b5) == f.height == len(b6), (mo, len(b5), f.height, len(b6))
        parts.append(f.with_columns(pl.Series('b5', b5), pl.Series('b6', b6)))
    E = pl.concat(parts)
    assert E['MVT_ID_mvt'].n_unique() == E.height
    J = v.select('MVT_ID_mvt').join(E, on='MVT_ID_mvt', how='left', maintain_order='left')
    assert J['MVT_ID_mvt'].to_list() == v['MVT_ID_mvt'].to_list() and J['b5'].null_count() == 0
    b5, b6 = J['b5'].to_numpy(), J['b6'].to_numpy()
    avg = (b5 + b6) / 2
    base_eq = float(np.max(np.abs(avg - b)))
    p_avg, p_ensa, p_ensb = (np.rint(nongated(x, B)) for x in (avg, b5, b6))
    r_avg = float(np.sqrt(((p_avg - y) ** 2).mean()))
    rep['ensavg'] = dict(align=dict(max_abs_avg_minus_loadtrain_base=base_eq, rmse_on=round(r_avg, 6), ref_ensavg_only=ENSAVG_REF,
                                align_ok=bool(abs(r_avg - ENSAVG_REF) <= 1e-4 and base_eq < 1e-6)),
                     vs_ensa=judge(p_avg, p_ensa, y, fd, dv), report_vs_ensb=judge(p_avg, p_ensb, y, fd, dv))
    print('ensavg', json.dumps({k: rep['ensavg']['vs_ensa'][k] for k in ('rmse_on', 'rmse_off', 'd_rmse', 'n_fold_improve', 'keep')}),
          'align', rep['ensavg']['align'], flush=True)

    # ---- chain proxy OOF (as chain in floor_oof_check.py), both arms get the lower bound 119 at the end
    tk = V.load_oof('v22/cv_tkall_s8.oof.npy', n)
    anc = V.load_oof('resid_cv/cv_anchor_s8.oof.npy', n)
    key = V.cells(v)
    km = V.keep_mask(key, a.keep)
    sk = np.array(v['stand_key'].to_list(), dtype=object)
    G = g & nA & (key != 'LIRF-07')
    resid = y - (b + tk)
    corr = np.zeros(n)
    for k in range(5):
        tab = V.sc_table(sk, resid, G & (fd != k))
        te = g & ~km & (fd == k)
        corr[te] = [tab.get(x, 0.0) for x in sk[te]]
    day = np.array([int(d[8:10]) for d in dv])
    rv = ((av == 'EDDF') & (mv == '07')) | ((av == 'EDDF') & (mv == '01') & (day >= 29)) | ((av == 'LSZH') & (mv == '07'))
    p_afs = np.clip(b + tk + corr, 0, None)
    p_variant_simple = np.clip(b + anc, 0, None)
    use_simple = g & (km | rv)
    chain = np.maximum(np.rint(np.where(g, np.where(use_simple, p_variant_simple, p_afs), nongated(b, B))), FLOOR)
    rep['chain'] = dict(rows=int(n), gated=int(g.sum()), rmse=round(float(np.sqrt(((chain - y) ** 2).mean())), 6))

    tr_all = feat.filter(pl.col('src') != 99)
    vv = v.select('MVT_ID_mvt').join(tr_all.select([c for c in RULE1_COLS if c not in ('src', 'TAXITIME_SEC_mvt')]),
                                      on='MVT_ID_mvt', how='left', maintain_order='left')
    assert vv['MVT_ID_mvt'].to_list() == v['MVT_ID_mvt'].to_list()
    ym = vv['ym'].to_numpy()
    ms = vv['mvt_delay'].to_numpy().astype(float)

    # ---- rule 1 (key RULE1; as in lirf_airline_rule.py, hit rows = rint(mvt_delay))
    import importlib.util
    spec = importlib.util.spec_from_file_location('lirf_airline_rule', Path(code) / 'lirf_airline_rule.py')
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    p_rule1 = chain.copy()
    rule1_info = {}
    for mo in (1, 7):
        s1, s2 = R.rates(R.cand(tr_all.filter(pl.col('ym') != 202500 + mo)), 0.8)
        m = ym == 202500 + mo
        cm = R.cand(vv.filter(pl.Series(m)).with_columns(pl.lit(0).alias('TAXITIME_SEC_mvt')))
        hid = set(np.array(cm['MVT_ID_mvt'].to_list())[R.hits(cm, s1, s2)]) if cm.height else set()
        h = np.isin(ids, list(hid))
        p_rule1[h] = np.maximum(np.rint(ms[h]), FLOOR)
        rule1_info[str(mo)] = dict(cand=int(cm.height), hit=int(h.sum()), al=len(s1), al_ades=len(s2))
    rep['RULE1'] = dict(info=rule1_info, res=judge(p_rule1, chain, y, fd, dv))
    print('RULE1', json.dumps({k: rep['RULE1']['res'][k] for k in ('n_changed', 'd_rmse', 'fold_d_sse', 'n_fold_improve', 'keep')}), flush=True)

    # ---- rule 2 (key RULE2; global-P version)
    tr = tr_all.with_columns(pl.col('mvt_delay').cast(pl.Float64).alias('ms'), pl.col('TAXITIME_SEC_mvt').cast(pl.Float64).alias('y'))
    big = tr.filter((pl.col('ADEP_mvt') == 'LIRF') & (pl.col('flt_null') == 1) & (pl.col('ms') > 50000))
    p_rule2 = chain.copy()
    rule2_info = {}
    lirf_nm = (vv['ADEP_mvt'] == 'LIRF').fill_null(False).to_numpy() & (vv['flt_null'] == 1).fill_null(False).to_numpy()
    for mo in (1, 7):
        o = big.filter(pl.col('ym') != 202500 + mo)
        P = float(((o['y'] >= 86400) & ((o['y'] - o['ms']).abs() > 600)).mean()) if o.height else 0.0
        m = (ym == 202500 + mo) & lirf_nm & (ms > 36000)
        q = np.where(ms <= 50000, ms, (1 - P) * ms + P * 87300)
        p_rule2[m] = np.maximum(np.rint(q[m]), FLOOR)
        rule2_info[str(mo)] = dict(P=round(P, 4), n_big_other=int(o.height), hit=int(m.sum()), hit_le50k=int((m & (ms <= 50000)).sum()))
    rep['RULE2'] = dict(info=rule2_info, res=judge(p_rule2, chain, y, fd, dv))
    print('RULE2', json.dumps({k: rep['RULE2']['res'][k] for k in ('n_changed', 'd_rmse', 'fold_d_sse', 'n_fold_improve', 'keep')}), flush=True)

    rep['decision'] = dict(ensavg_keep=rep['ensavg']['vs_ensa']['keep'], rule1_include=rep['RULE1']['res']['keep'], rule2_include=rep['RULE2']['res']['keep'],
                           note='if ensavg_keep is false, the whole chain would have to be rebuilt on a different baseline (stop and report); rule1_include/rule2_include=true is reported only and not applied to the submitted variant')
    print('decision', json.dumps(rep['decision']), flush=True)
    tmp = out_path + '.tmp'
    json.dump(rep, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)


if __name__ == '__main__':
    main()
