"""tabular_weight_gate_production_rows: supplementary check of the tabular term (w = 0.25) on the production row set.
Read-only, no training, writes no existing artifact.

Background: the gate decision for the tabular term (shrink_w0.25 in tabular_weight_gate.py) is computed on all non-extreme (nonA) rows, including gated rows; production merge_predictions.py
adds w*D only on "non-gated and non-B" rows (including A rows) and applies clip(rint(.), 30). This script recomputes the gate on the production row set, with the same seed-0 D
(fcv_all_te2_pred.npz of tabular_cv.py), on the r25 rows of 2025-01/07. The decision gate is the original one, fixed before the computation:
  delta_pp >= 0.1, delta_pp > 2*SE (day blocks, block_se), both folds 01 and 07 > 0; denominator = SSE of the averaged baseline on all nonA rows (as in tabular_weight_gate.py).
Row set R = ~g & ~B (g = r25/v22/cv_tkall_s8.oof.npy != 0, i.e. res != 0 as in merge_predictions.py); off arm = clip(rint(baseline), 30), on arm = clip(rint(off arm + w*D), 30).
Self-check: the original computation must be reproduced as 0.391 +- 0.110 (folds .276 / .443); r25 has 344,419 rows, 344,411 nonA, and the npz covers all of them; averaged-baseline RMSE = 310.062496.
If the self-check fails, only the numbers are reported, no verdict.
Complexity: fully vectorised O(N) (N ~ 344k), one MVT_ID hash join; the dominant peak is load_train (about 1.14 GB measured for the same kind of load).
Usage (cwd = final/r25; environment as for floor_oof_check.py: E_RESID + RESID_CELLFB=off):
  python tabular_weight_gate_production_rows.py --r25 . --npz <fcv_all_te2_pred.npz> --out <json>
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

W = 0.25
REF_ORIG = {'d_pct': 0.391, 'se_pct': 0.110, '01': 0.276, '07': 0.443}
ENSAVG_REF = 310.062496
N_ROWS, N_NONA = 344419, 344411


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for ch in iter(lambda: fh.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def block_se(d, blocks):  # same as block_se in tabular_weight_gate.py
    s = np.bincount(blocks, weights=d)
    s = s[np.bincount(blocks) > 0]
    n = len(s)
    return float(np.sqrt(n / (n - 1) * ((s - s.mean()) ** 2).sum())) if n > 1 else float('nan')


def stat(v, mk, den, day):
    return {'d_pct': round(float(v[mk].sum() / den * 100), 4), 'se_pct': round(block_se(v[mk], day[mk]) / den * 100, 4)}


def judged(v, mk, den, den_fo, mo, day):
    """Original gate: d_pct >= 0.1, > 2 SE, both folds (01/07) > 0. mk = numerator row mask (production basis: all rows, v = 0 outside R)."""
    r = stat(v, mk, den, day)
    r['folds'] = {f'{fo:02d}': stat(v, mk & (mo == fo), den_fo[fo], day) for fo in (1, 7)}
    r['gate'] = bool(r['d_pct'] >= 0.1 and r['d_pct'] > 2 * r['se_pct'] and all(f['d_pct'] > 0 for f in r['folds'].values()))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r25', required=True)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out_path = Path(a.out).resolve()
    self_md5 = md5(os.path.abspath(__file__))
    npz_path = str(Path(a.npz).resolve())
    code = os.environ['RESID_CODE_DIR']
    os.chdir(a.r25)
    sys.path.insert(0, code)
    import residual_context as V
    T = V.T
    rep = {'script_md5': self_md5, 'code_dir': code, 'train_resid': str(T.__file__), 'feats': V.FE, 'w': W,
           'files': {'npz': {'path': npz_path, 'md5': md5(npz_path)},
                     'oof_gate': {'path': 'v22/cv_tkall_s8.oof.npy', 'md5': md5('v22/cv_tkall_s8.oof.npy')},
                     'feat': {'path': 'n2/feat.parquet'}}}

    # ---- r25 rows (as in floor_oof_check)
    v = T.load_train('n2', V.FE)
    n = v.height
    y = v['TAXITIME_SEC_mvt'].to_numpy().astype(np.float64)
    b = v['base'].to_numpy().astype(np.float64)
    nonA = ~v['isA'].to_numpy()
    gv = v['gate'].to_numpy()
    B = ((v['flt_null'] == 1) & (v['ADEP_mvt'] == 'LIRF') & (v['mvt_delay'] > 3600)).fill_null(False).to_numpy()
    mo = v['date'].str.slice(5, 2).cast(pl.Int32).to_numpy()
    day = (v['date'].rank('dense').cast(pl.Int64) - 1).to_numpy()
    nd = int(day.max()) + 1
    fold_of = np.empty(nd, np.int64)
    fold_of[np.random.default_rng(0).permutation(nd)] = np.arange(nd) % 5  # same as day_folds in residual_context.py (seed 0), vectorised
    fd = fold_of[day]
    oof = V.load_oof('v22/cv_tkall_s8.oof.npy', n)
    g = oof != 0

    # ---- D: npz idx -> feat row number -> MVT_ID -> r25 row
    z = np.load(npz_path)
    idx = z['idx']
    nb = z['base'].astype(np.float64)
    nn = z['new'].astype(np.float64)
    f = (pl.scan_parquet('n2/feat.parquet').select(['MVT_ID_mvt', 'src', 'ym', 'aobt_sched', 'synth'])
         .filter(pl.col('src') != 99).with_row_index('_i').filter(pl.col('ym').is_in([202501, 202507])).collect())
    assert f.height == len(idx) and (f['_i'].to_numpy() == idx).all(), 'npz row order does not match the January/July rows of feat'
    assert f['MVT_ID_mvt'].n_unique() == f.height
    f = f.select('MVT_ID_mvt', 'aobt_sched', 'synth').with_columns(pl.Series('D', nn - nb))
    j = v.select('MVT_ID_mvt', 'flt_null', 'mvt_delay').with_row_index('_r').join(f, on='MVT_ID_mvt', how='left').sort('_r')
    assert j.height == n
    n_unmatched = int(j['D'].null_count())
    D = j['D'].fill_null(0.0).to_numpy().astype(np.float64)
    A = ~nonA
    C = ((j['flt_null'] == 0) & (j['aobt_sched'] > 1200) & (j['mvt_delay'] > 1200)).fill_null(False).to_numpy() & ~A & ~B
    S = (j['synth'] == 1).fill_null(False).to_numpy() & ~A & ~B & ~C
    cls = np.where(B, 'B', np.where(A, 'A', np.where(C, 'C', np.where(S, 'S', 'D'))))
    del j, f

    rep['rows'] = {'n': int(n), 'nonA': int(nonA.sum()), 'A': int(A.sum()), 'B': int(B.sum()), 'gated_oof_ne0': int(g.sum()),
                   'gated_v_gate': int(gv.sum()), 'gate_mismatch_oof_vs_v': int((g != gv).sum()), 'npz_rows': int(len(idx)),
                   'unmatched_r25_rows': n_unmatched, 'D_B_max_abs': float(np.abs(D[B]).max()) if B.any() else 0.0,
                   'days': nd}
    allm = np.ones(n, bool)
    q8 = (b - y) ** 2
    e8 = b - y
    den = float(q8[nonA].sum())
    den_fo = {fo: float(q8[nonA & (mo == fo)].sum()) for fo in (1, 7)}
    rep['den'] = {'ensavg_nonA_sse': den, 'ensavg_nonA_mse': round(den / nonA.sum(), 1), 'folds': den_fo}

    # ---- self-check 1: original basis (shrink_w0.25 of tabular_weight_gate.py: all nonA rows, no rint/clip)
    d_orig = q8 - (e8 + W * D) ** 2
    orig = judged(d_orig, nonA, den, den_fo, mo, day)
    orig['split_pct'] = {k: round(float(d_orig[m].sum() / den * 100), 4) for k, m in
                         (('gated', nonA & g), ('nongated_nonB', nonA & ~g & ~B), ('nongated_B', nonA & ~g & B))}
    rep['orig_reproduce'] = orig

    # ---- production basis: R = ~g & ~B (includes A)
    R = ~g & ~B
    off = np.clip(np.rint(b), 30, None)
    on = off.copy()
    on[R] = np.clip(np.rint(off[R] + W * D[R]), 30, None)
    d_prod = np.where(R, (off - y) ** 2 - (on - y) ** 2, 0.0)
    prod = judged(d_prod, allm, den, den_fo, mo, day)
    prod['rows_R'] = int(R.sum())
    prod['rows_R_A'] = int((R & A).sum())
    prod['rows_changed'] = int((on != off).sum())
    prod['cls_pct'] = {k: round(float(d_prod[cls == k].sum() / den * 100), 4) for k in 'CDSA'}
    prod['day5'] = {'fold_d_sse_improve': [round(float(d_prod[fd == k].sum()), 1) for k in range(5)],
                    'n_fold_improve': int(sum(1 for k in range(5) if d_prod[fd == k].sum() > 0))}
    # dRMSE / dMSE at the level of the averaged baseline (outside R both arms equal the non-gated transform of the averaged baseline)
    pb = np.rint(np.where(B, b, np.clip(np.rint(b), 30, None)))
    pon = pb.copy()
    pon[R] = on[R]
    rm_off = float(np.sqrt(((pb - y) ** 2).mean()))
    rm_on = float(np.sqrt(((pon - y) ** 2).mean()))
    prod['rmse'] = {'off_ensavg_only': round(rm_off, 6), 'on': round(rm_on, 6), 'd_rmse': round(rm_on - rm_off, 6),
                    'd_mse': round(float(-d_prod.sum() / n), 4), 'se_mse_dayblock': round(block_se(d_prod, day) / n, 4)}
    rep['prod'] = prod

    # ---- report only: production row set without rint/clip; variant with lower bound 119
    d_u = np.where(R, q8 - (e8 + W * D) ** 2, 0.0)
    rep['prod_unrounded'] = judged(d_u, allm, den, den_fo, mo, day)
    off9, on9 = np.maximum(off, 119), np.maximum(on, 119)
    d9 = np.where(R, (off9 - y) ** 2 - (on9 - y) ** 2, 0.0)
    rep['prod_floor119'] = judged(d9, allm, den, den_fo, mo, day)

    # ---- self-check and verdict
    raw = {'d_pct': float(d_orig[nonA].sum() / den * 100), 'se_pct': block_se(d_orig[nonA], day[nonA]) / den * 100,
           '01': float(d_orig[nonA & (mo == 1)].sum() / den_fo[1] * 100), '07': float(d_orig[nonA & (mo == 7)].sum() / den_fo[7] * 100)}
    rep['orig_reproduce']['raw'] = raw
    sc = {'orig_reproduce': bool(all(round(raw[k], 3) == REF_ORIG[k] for k in REF_ORIG)),
          'rows': bool(n == N_ROWS and int(nonA.sum()) == N_NONA and n_unmatched == 0),
          'ensavg_only_rmse': bool(abs(rm_off - ENSAVG_REF) <= 1e-4)}
    rep['selfcheck'] = sc
    ok = all(sc.values())
    rep['verdict'] = {'selfcheck_ok': ok, 'gate_pass': prod['gate'] if ok else None,
                      'note': 'original gate: delta_pp>=0.1, >2SE (day blocks), both folds 01/07 > 0' if ok else 'self-check failed: numbers only, no verdict'}
    tmp = out_path.with_suffix('.tmp.json')
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, out_path)
    print(json.dumps({'rows': rep['rows'], 'selfcheck': sc, 'orig': {k: orig[k] for k in ('d_pct', 'se_pct', 'folds', 'split_pct')},
                      'prod': {k: prod[k] for k in ('d_pct', 'se_pct', 'folds', 'gate', 'rmse', 'day5')}, 'verdict': rep['verdict']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
