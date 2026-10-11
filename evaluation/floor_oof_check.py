"""floor_oof_check: checks the prediction lower bound of the full version on 2025 OOF predictions.

Background: the full version ends with a lower bound of 119 s (set from the marginal distribution of the 2025 training DEP labels, not checked on validation data). This script is read-only,
trains nothing, and compares "no bound" with bounds in {30, 60, 74, 90, 100, 119} on 2025 OOF predictions: RMSE (the competition metric) and number of affected rows.

OOF (there is no 2025 OOF of the final merge chain; following the merge rules of the final chain, a "chain proxy OOF" is assembled from the existing component OOFs, and it is the main OOF):
- chain (main): the DEP rows of the 62 days of 2025-01/07 (r25 old n2, 344,419 rows, including isA). base = (ens_v5b + ens_v6)/2 leave-one-month-out OOF (the two members of the averaged baseline).
  Gated rows: variant_full_sc = clip(base + cv_tkall_s8 residual + stand correction, 0); the stand correction is built strictly per fold as in the full variant of evalx in residual_context.py
  (table from the other folds, G = residuals of gated & non-A & not LIRF-07 rows, m = 20) and added only on gated rows outside the keep cells; the keep cells (EGLL-07, LIRF)
  and the cells reverted by the drift gate (EDDF-07, EDDF-01 days 29-31, LSZH-07) use variant_simple = clip(base + cv_anchor_s8 residual, 0).
  Non-gated rows: outside B, clip(rint(base), 30) (the tabular term w*D of merge_base.py has no OOF in 2025 and is taken as 0); B rows = base. Finally rint.
- ensavg_only (reference): all rows take the non-gated path (no residual layer), same clip.
- srp12 (reference): oof12/srp_g5_avg (all 12 months of 2025, 2,085,047 DEP rows, in the training-row order of feat.parquet), same non-gated clip.
Applying a bound f: p' = max(p, f) (only rows below f are raised).

Decision rule (fixed before seeing results):
1. Look only at the overall RMSE of chain (all 344,419 rows, including isA); candidates = none, 30, 60, 74, 90, 100, 119.
2. The candidate with the smallest overall RMSE is "best"; exact ties go to the smallest bound ("none" counts as smallest).
3. If |RMSE(best) - RMSE(119)| < 0.001 s, keep 119; otherwise the final bound = best ("none" = drop the bound).
Report: per OOF and candidate, the overall and per-airport RMSE and the number of affected rows; among rows predicted < 119, the share whose truth is < 119; the decision.
Usage (any cwd):
  RESID_* (as E_RESID + RESID_CELLFB=off of asset25_evalx in run_stage_chain.sh) python floor_oof_check.py --r25 final/r25 --out final/logs/floor_oof_check.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

FLOORS = [None, 30, 60, 74, 90, 100, 119]
TOL = 0.001
CUR = 119


def nm(f):
    return 'none' if f is None else str(f)


def evaluate(p, y, apt):
    out = {}
    base_rmse = None
    for f in FLOORS:
        q = p if f is None else np.maximum(p, f)
        e2 = (q - y) ** 2
        aff = np.zeros(len(p), bool) if f is None else (p < f)
        r = float(np.sqrt(e2.mean()))
        if f is None:
            base_rmse = r
        o = dict(rmse=round(r, 6), delta_vs_none=round(r - base_rmse, 6), n_affected=int(aff.sum()), apt={})
        for a in sorted(set(apt.tolist())):
            m = apt == a
            o['apt'][a] = dict(rmse=round(float(np.sqrt(e2[m].mean())), 4), n_affected=int((aff & m).sum()), n=int(m.sum()))
        out[nm(f)] = o
    lo = p < CUR
    out['lt119_stats'] = dict(n_pred_lt119=int(lo.sum()), share_true_lt119=round(float((y[lo] < CUR).mean()), 4) if lo.any() else None,
                              y_median=float(np.median(y[lo])) if lo.any() else None, p_median=float(np.median(p[lo])) if lo.any() else None,
                              share_true_lt_pred=round(float((y[lo] < p[lo]).mean()), 4) if lo.any() else None,
                              n_true_lt119_all=int((y < CUR).sum()), n_all=int(len(y)))
    return out


def decide(res):
    r = {nm(f): res[nm(f)]['rmse'] for f in FLOORS}
    best_v = min(r.values())
    best = next(nm(f) for f in FLOORS if r[nm(f)] == best_v)
    gap = r[str(CUR)] - best_v
    final = str(CUR) if gap < TOL else best
    return dict(rmse=r, best=best, best_rmse=best_v, rmse_119=r[str(CUR)], gap_119_minus_best=round(gap, 6), tol=TOL,
                final_floor=final, keep_119=final == str(CUR))


def nongated(b, B):
    return np.where(B, b, np.clip(np.rint(b), 30, None))


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
    rep = dict(rule=__doc__.split('Decision rule (fixed before seeing results):')[1].split('Report:')[0].strip(), floors=[nm(f) for f in FLOORS],
               code=str(T.__file__), feats=V.FE, keep=a.keep)

    # ---- chain (main)
    v = T.load_train('n2', V.FE)
    y, b, g, nA, av, dv, mv = V.base_arrays(v)
    n = v.height
    tk = V.load_oof('v22/cv_tkall_s8.oof.npy', n)
    anc = V.load_oof('resid_cv/cv_anchor_s8.oof.npy', n)
    fd = V.day_folds(dv)
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
    B = ((v['flt_null'] == 1) & (v['ADEP_mvt'] == 'LIRF') & (v['mvt_delay'] > 3600)).fill_null(False).to_numpy()
    p_afs = np.clip(b + tk + corr, 0, None)
    p_variant_simple = np.clip(b + anc, 0, None)
    use_simple = g & (km | rv)
    p = np.rint(np.where(g, np.where(use_simple, p_variant_simple, p_afs), nongated(b, B)))
    ok = np.isfinite(y) & np.isfinite(p)
    rep['chain'] = dict(rows=int(n), rows_used=int(ok.sum()), gated=int(g.sum()), gated_variant_simple=int(use_simple.sum()), gated_revert=int((g & rv).sum()),
                        sc_rows=int((corr != 0).sum()), B=int(B.sum()), isA=int((~nA).sum()), min_pred=float(p[ok].min()),
                        res=evaluate(p[ok], y[ok], av[ok]))
    rep['chain']['res_nonA_rmse'] = {nm(f): round(float(np.sqrt(((np.maximum(p, f if f is not None else -1e18) - y)[ok & nA] ** 2).mean())), 6) for f in FLOORS}
    pb = np.rint(nongated(b, B))
    rep['ensavg_only'] = dict(rows_used=int(ok.sum()), res=evaluate(pb[ok], y[ok], av[ok]))
    print('chain', json.dumps({k: rep['chain']['res'][k]['rmse'] for k in rep['floors']}), flush=True)

    # ---- srp12 (reference, 12 months)
    d = pl.read_parquet('n2/feat.parquet', columns=['MVT_ID_mvt', 'src', 'ADEP_mvt', 'TAXITIME_SEC_mvt', 'flt_null', 'mvt_delay']).filter(pl.col('src') != 99)
    p12 = np.load('n2/oof12/srp_g5_avg.npy')
    assert len(p12) == d.height, (len(p12), d.height)
    y12 = d['TAXITIME_SEC_mvt'].cast(pl.Float64).to_numpy()
    B12 = ((d['flt_null'] == 1) & (d['ADEP_mvt'] == 'LIRF') & (d['mvt_delay'] > 3600)).fill_null(False).to_numpy()
    q12 = np.rint(nongated(p12.astype(np.float64), B12))
    ok12 = np.isfinite(y12) & np.isfinite(q12)
    rep['srp12'] = dict(rows=int(d.height), rows_used=int(ok12.sum()), ids_unique=bool(d['MVT_ID_mvt'].n_unique() == d.height),
                        min_raw=float(np.nanmin(p12)), res=evaluate(q12[ok12], y12[ok12], d['ADEP_mvt'].to_numpy()[ok12]))
    print('srp12', json.dumps({k: rep['srp12']['res'][k]['rmse'] for k in rep['floors']}), flush=True)

    rep['decision'] = decide(rep['chain']['res'])
    rep['decision_ref'] = dict(ensavg_only=decide(rep['ensavg_only']['res']), srp12=decide(rep['srp12']['res']))
    print('decision', json.dumps(rep['decision']), flush=True)
    tmp = out_path + '.tmp'
    json.dump(rep, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)


if __name__ == '__main__':
    main()
