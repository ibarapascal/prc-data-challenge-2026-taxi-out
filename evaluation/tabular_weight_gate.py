"""Post-processing of the tabular formal CV for blended modes: adds day-paired SE, per-fold and per-cell results and a 2026 conversion for blend modes such as avgblend to the *_pred.npz of tabular_cv.py.

Background: tabular_cv.py reported SE only at member level and for the transplant "baseline + (new - base)"; avgblend had point values only. This script reads the same feature store (storedir) and pred.npz (no training)
and recomputes each blend mode with the noise-tax decomposition. Let e8/eb/en be the signed errors of the averaged baseline / base arm / new arm and D = new - base; then per row
  member M = eb^2 - en^2; transplant T = e8^2 - (e8+D)^2 = -D^2 - 2*e8*D; avgblend (avg(baseline,new) against avg(baseline,base)) A1 = (M+T+D^2)/4.
Modes reported (all with the baseline's nonA SSE as denominator, positive = better):
  avgblend_iso  avg(baseline,new) vs avg(baseline,base)  -- contribution of the features themselves (same definition as avgblend in tabular_cv.py, for cross-checking)
  avgblend_vs_ensavg  avg(baseline,new) vs baseline            -- gain of the actual candidate over the baseline (includes the ensemble effect of "one more refitted member")
  avgblendb_vs_ensavg avg(baseline,base) vs baseline           -- ensemble effect of one more base-arm member only (control)
  shrink_w   baseline + w*D vs baseline (w = .25/.5/1) -- shrunken transplant
Gate: nonA SSE >= +0.1 pp, > 2 SE (day-paired), positive in a majority of folds (with two folds: both positive).
Usage: python tabular_weight_gate.py <storedir> <pred.npz> <out.json>
Peak memory: reads 9 columns of the January and July rows only (about 400k rows), < 0.5 GB.
"""
import json
import os
import sys

import numpy as np
import polars as pl

os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
# ADDR26: label-free estimate of each airport-month cell's share (%) of the 2026 non-extreme SSE that the ADS-B layer cannot reach.
# Per-row MSE of the averaged-baseline 2025 month-held-out OOF by airport x month x row class, times the 2026 row counts per cell and class;
# the artefact class (B) is subtracted and the remainder scaled by (1 - 2026 gated share). Used only to express CV gains as a 2026 projection.
ADDR26 = {'LIRF-07': 28.0 - 12.0, 'LIRF-01': (8.7 - 3.6) * (1 - 0.79), 'LTFM-07': 8.8 - 0.2, 'LTFM-01': 7.6 - 0.4,
          'LFPG-01': (6.7 - 0.6) * (1 - 0.23), 'LFPG-07': (6.2 - 0.5) * (1 - 0.19)}



def block_se(d, blocks):
    s = np.bincount(blocks, weights=d)
    s = s[np.bincount(blocks) > 0]
    n = len(s)
    return float(np.sqrt(n / (n - 1) * ((s - s.mean()) ** 2).sum())) if n > 1 else float('nan')


def main():
    storedir, npz, out = sys.argv[1:4]
    cols = ['src', 'ym', 'TAXITIME_SEC_mvt', 'flt_null', 'ADEP_mvt', 'mvt_delay', 'aobt_sched', 'synth', 'MVT_TIME_UTC_mvt']
    d = (pl.scan_parquet(os.path.join(storedir, 'feat.parquet')).select(cols).filter(pl.col('src') != 99).with_row_index('_i')
         .filter(pl.col('ym').is_in([202501, 202507])).collect())
    z = np.load(npz)
    idx = z['idx']
    ii = d['_i'].to_numpy()
    assert len(ii) == len(idx) and (ii == idx).all(), 'pred.npz row order does not match the January/July rows of feat'
    mo = (d['ym'] - 202500).to_numpy()
    y = d['TAXITIME_SEC_mvt'].to_numpy().astype(np.float64)
    B = ((d['flt_null'] == 1) & (d['ADEP_mvt'] == 'LIRF') & (d['mvt_delay'] > 3600)).fill_null(False).to_numpy()
    A = (y > 10800) & ~B
    C = ((d['flt_null'] == 0) & (d['aobt_sched'] > 1200) & (d['mvt_delay'] > 1200)).fill_null(False).to_numpy() & ~A & ~B
    S = (d['synth'] == 1).fill_null(False).to_numpy() & ~A & ~B & ~C
    cls = np.where(B, 'B', np.where(A, 'A', np.where(C, 'C', np.where(S, 'S', 'D'))))
    ensavg = np.full(len(y), np.nan)
    for fo in (1, 7):
        mk = mo == fo
        ensavg[mk] = (np.load(os.path.join(storedir, 'oof', f'ens_v5b_{fo}.npy')) + np.load(os.path.join(storedir, 'oof', f'ens_v6_{fo}.npy'))) / 2
    base = z['base'].astype(np.float64); new = z['new'].astype(np.float64)
    assert np.isfinite(ensavg).all() and np.isfinite(base).all() and np.isfinite(new).all()
    day = d['MVT_TIME_UTC_mvt'].dt.date().cast(pl.Int32).to_numpy(); day = day - day.min()
    apt = d['ADEP_mvt'].to_numpy()
    nonA = cls != 'A'
    e8 = ensavg - y; eb = base - y; en = new - y; D = new - base
    q8 = e8 ** 2
    # per-row SSE improvement of each mode (positive = better)
    diffs = {'member': eb ** 2 - en ** 2,
             'transplant_w1': q8 - (e8 + D) ** 2,
             'shrink_w0.5': q8 - (e8 + 0.5 * D) ** 2,
             'shrink_w0.25': q8 - (e8 + 0.25 * D) ** 2,
             'avgblend_iso': ((e8 + eb) / 2) ** 2 - ((e8 + en) / 2) ** 2,
             'avgblend_vs_ensavg': q8 - ((e8 + en) / 2) ** 2,
             'avgblendb_vs_ensavg': q8 - ((e8 + eb) / 2) ** 2}

    def stat(v, mk, den):
        return {'d_pct': round(float(v[mk].sum() / den * 100), 3), 'se_pct': round(block_se(v[mk], day[mk]) / den * 100, 3)}

    tot8 = q8[nonA].sum(); totb = (eb ** 2)[nonA].sum()
    res = {'npz': os.path.basename(npz), 'n_nonA': int(nonA.sum()), 'ensavg_nonA_mse': round(float(tot8 / nonA.sum()), 1),
           'base_nonA_mse': round(float(totb / nonA.sum()), 1),
           'tax': {'meanD2_pct_of_ensavg': round(float((D ** 2)[nonA].sum() / tot8 * 100), 3),
                   'cross_minus2e8D_pct_of_ensavg': round(float((-2 * e8 * D)[nonA].sum() / tot8 * 100), 3),
                   'member_pct_of_ensavg': round(float(diffs['member'][nonA].sum() / tot8 * 100), 3),
                   'rmsD_s': round(float(np.sqrt((D ** 2)[nonA].mean())), 2),
                   'corr_e8_eb': round(float(np.corrcoef(e8[nonA], eb[nonA])[0, 1]), 4)},
           'modes': {}}
    for k, v in diffs.items():
        den = totb if k == 'member' else tot8
        r = stat(v, nonA, den)
        r['folds'] = {}
        for fo in (1, 7):
            mk = nonA & (mo == fo)
            dfo = (eb ** 2)[mk].sum() if k == 'member' else q8[mk].sum()
            r['folds'][f'{fo:02d}'] = stat(v, mk, dfo)
        r['cls'] = {c: round(float(v[nonA & (cls == c)].sum() / den * 100), 3) for c in 'CDS'}
        cells = {}
        for ap_ in APTS:
            for mm in (1, 7):
                mk = nonA & (apt == ap_) & (mo == mm) & (cls != 'B')
                dc = (eb ** 2)[mk].sum() if k == 'member' else q8[mk].sum()
                cells[f'{ap_}-{mm:02d}'] = {'rel': round(float(v[mk].sum() / dc * 100), 2), 'se': round(block_se(v[mk], day[mk]) / dc * 100, 2)}
        r['cells'] = cells
        p26 = sum(cells[c]['rel'] / 100 * w for c, w in ADDR26.items())
        r['p26_pct'] = round(p26, 3)
        r['gate'] = bool(r['d_pct'] >= 0.1 and r['d_pct'] > 2 * r['se_pct'] and all(f['d_pct'] > 0 for f in r['folds'].values()))
        res['modes'][k] = r
    with open(out, 'w') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for k, r in res['modes'].items():
        print(f"{k:14s} {r['d_pct']:+7.3f} +/- {r['se_pct']:.3f}  folds {[f['d_pct'] for f in r['folds'].values()]}  p26 {r['p26_pct']:+.3f}%  gate {r['gate']}")
    print('tax', res['tax'])


if __name__ == '__main__':
    main()
