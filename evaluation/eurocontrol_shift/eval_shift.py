"""Pre-specified evaluation of shifting each airport-month mean of the model towards EUROCONTROL's published monthly taxi-out mean.

What it does: on 2025 training-period OOF predictions, shift the predicted mean of every airport x month cell by alpha towards "published mean + airport bias".
Primary validation = forward 5 folds (test months 08-12); gate = pooled RMSE improvement >= 2 SE (cell-level bootstrap) and the same direction in 5/5 folds.
Also reported, for reference only: 12-fold leave-one-month-out, a strong baseline for months 1/7 (oof/ens_{1,7}), without class A, per airport and correlations.
Inputs are read-only and no artifact is modified; writes one JSON (temporary file, then renamed) and a stdout summary. All SSE values are computed in closed form from the cell-level
sufficient statistics (n, sum r, sum r^2); memory is O(rows) only while reading.

Usage:
  python eval_shift.py --feat <root>/work/n2/feat.parquet --storedir <root>/final/n2 \
    --ectx <root>/final/ext_ectx --out <root>/final/logs/ectx_eval.json
Dependencies: numpy, polars
"""
import argparse, csv, json, os

import numpy as np
import polars as pl

APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FWD = [8, 9, 10, 11, 12]
B, SEED = 2000, 20261009


def load_official(d):
    O = np.full((len(APTS), 13), np.nan)  # [apt, month], year 2025
    O26 = {}
    for y in (2025, 2026):
        with open(os.path.join(d, f'taxi_out_additional_time_{y}.csv'), newline='', encoding='utf-8-sig') as f:
            for x in csv.DictReader(f):
                if x['APT_ICAO'] not in APTS:
                    continue
                try:
                    n = float(x['TOTAL_REF_NB_FL']); v = (float(x['TOTAL_REF_TIME_MIN']) + float(x['TOTAL_ADD_TIME_MIN'])) * 60 / n
                except (ValueError, ZeroDivisionError):
                    continue
                m = int(x['MONTH_NUM'])
                if y == 2025:
                    O[APTS.index(x['APT_ICAO']), m] = v
                else:
                    O26[f"{x['APT_ICAO']}-{m:02d}"] = round(v, 1)
    return O, O26


def cell_stats(ai, mi, r, mask=None):
    """Cell-level sufficient statistics: n, sum r, sum r^2, shape [apt, 13]."""
    if mask is not None:
        ai, mi, r = ai[mask], mi[mask], r[mask]
    idx = ai * 13 + mi
    k = len(APTS) * 13
    n = np.bincount(idx, minlength=k).reshape(len(APTS), 13).astype(float)
    s1 = np.bincount(idx, weights=r, minlength=k).reshape(len(APTS), 13)
    s2 = np.bincount(idx, weights=r * r, minlength=k).reshape(len(APTS), 13)
    return n, s1, s2


def sse(st, a, m, dd):
    n, s1, s2 = st
    return s2[a, m] - 2 * dd * s1[a, m] + n[a, m] * dd * dd


class Shift:
    def __init__(self, O, Ybar, Pbar):
        self.O, self.Ybar, self.Pbar = O, Ybar, Pbar

    def c(self, S):
        S = list(S)
        return np.nanmean(self.Ybar[:, S] - self.O[:, S], axis=1)

    def d(self, alpha, S, m):
        """Shift per airport for test month m (vector of length 10); 0 where the published value is missing."""
        v = alpha * (self.O[:, m] + self.c(S) - self.Pbar[:, m])
        return np.where(np.isfinite(v), v, 0.0)

    def pick_alpha(self, st, S):
        best = None
        for al in ALPHAS:
            tot = 0.0
            for s in S:
                dd = self.d(al, [x for x in S if x != s], s)
                tot += sum(sse(st, a, s, dd[a]) for a in range(len(APTS)))
            if best is None or tot < best[1] - 1e-9:
                best = (al, tot)
        return best[0]


def fold_cells(sh, st, st_na, test, S, alpha):
    dd = sh.d(alpha, S, test)
    out = []
    for a in range(len(APTS)):
        out.append(dict(apt=APTS[a], m=test, n=st[0][a, test], b=sse(st, a, test, 0.0), s=sse(st, a, test, dd[a]),
                        n_na=st_na[0][a, test], b_na=sse(st_na, a, test, 0.0), s_na=sse(st_na, a, test, dd[a]),
                        shift=float(dd[a])))
    return out


def rmse_delta(cells, na=False):
    k = '_na' if na else ''
    n = sum(c['n' + k] for c in cells); b = sum(c['b' + k] for c in cells); s = sum(c['s' + k] for c in cells)
    return float(np.sqrt(b / n)), float(np.sqrt(s / n)), float(np.sqrt(b / n) - np.sqrt(s / n))


def boot_se(cells, na=False):
    k = '_na' if na else ''
    n = np.array([c['n' + k] for c in cells]); b = np.array([c['b' + k] for c in cells]); s = np.array([c['s' + k] for c in cells])
    rng = np.random.default_rng(SEED); out = np.empty(B)
    for i in range(B):
        j = rng.integers(0, len(cells), len(cells))
        out[i] = np.sqrt(b[j].sum() / n[j].sum()) - np.sqrt(s[j].sum() / n[j].sum())
    return float(out.std(ddof=1))


def run_scheme(sh, st, st_na, tests, S_of):
    folds, cells_all = [], []
    for t in tests:
        S = S_of(t); al = sh.pick_alpha(st, S); cells = fold_cells(sh, st, st_na, t, S, al)
        rb, rs, d = rmse_delta(cells); _, _, dna = rmse_delta(cells, True)
        folds.append(dict(test=t, alpha=al, rmse_base=round(rb, 3), rmse_shift=round(rs, 3), delta=round(d, 3),
                          delta_nonA=round(dna, 3), mean_abs_shift=round(float(np.mean([abs(c['shift']) for c in cells])), 2),
                          c=dict(zip(APTS, np.round(sh.c(S), 1).tolist()))))
        cells_all += cells
    rb, rs, d = rmse_delta(cells_all); _, _, dna = rmse_delta(cells_all, True)
    se = boot_se(cells_all); se_na = boot_se(cells_all, True)
    per_apt = {}
    for ap in APTS:
        cc = [c for c in cells_all if c['apt'] == ap]
        per_apt[ap] = round(rmse_delta(cc)[2], 3)
    return dict(folds=folds, pooled=dict(rmse_base=round(rb, 3), rmse_shift=round(rs, 3), delta=round(d, 4), se=round(se, 4),
                                         z=round(d / se, 2) if se > 0 else None, delta_nonA=round(dna, 4), se_nonA=round(se_na, 4),
                                         n_pos=sum(f['delta'] > 0 for f in folds), n_folds=len(folds)),
                per_apt_delta=per_apt, cells=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()
                                               if k in ('apt', 'm', 'n', 'shift')} for c in cells_all])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', required=True)
    ap.add_argument('--storedir', required=True)
    ap.add_argument('--ectx', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='srp_g5_avg')
    a = ap.parse_args()

    O, O26 = load_official(a.ectx)
    assert np.isfinite(O[:, 1:]).all(), 'official 2025 incomplete'
    d = pl.read_parquet(a.feat, columns=['src', 'ym', 'ADEP_mvt', 'TAXITIME_SEC_mvt', 'flt_null', 'mvt_delay']).filter(pl.col('src') != 99)
    y = d['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    ym = d['ym'].to_numpy().astype(int); assert ((ym // 100) == 2025).all()
    mi = ym % 100
    apt = d['ADEP_mvt'].to_numpy().astype(str); assert set(np.unique(apt)) <= set(APTS), set(np.unique(apt)) - set(APTS)
    ai = np.searchsorted(np.array(APTS), apt)
    isB = ((d['flt_null'] == 1) & (d['ADEP_mvt'] == 'LIRF') & (d['mvt_delay'] > 3600)).fill_null(False).to_numpy()
    nonA = ~((y > 10800) & ~isB)
    p = np.load(os.path.join(a.storedir, 'oof12', a.base + '.npy')); assert len(p) == len(y), (len(p), len(y))
    assert np.isfinite(p).all() and np.isfinite(y).all()
    r = y - p
    st = cell_stats(ai, mi, r); st_na = cell_stats(ai, mi, r, nonA)
    n = st[0]
    Ybar = np.bincount(ai * 13 + mi, weights=y, minlength=130).reshape(10, 13) / np.where(n > 0, n, np.nan)
    Pbar = np.bincount(ai * 13 + mi, weights=p, minlength=130).reshape(10, 13) / np.where(n > 0, n, np.nan)
    sh = Shift(O, Ybar, Pbar)
    rep = dict(rows=int(len(y)), base=a.base, alphas=ALPHAS, B=B, seed=SEED)

    rep['forward5'] = run_scheme(sh, st, st_na, FWD, lambda t: list(range(1, t)))
    f = rep['forward5']['pooled']
    rep['gate'] = dict(delta_ge_2se=bool(f['delta'] >= 2 * f['se']), all_folds_pos=bool(f['n_pos'] == 5))
    rep['gate']['pass'] = rep['gate']['delta_ge_2se'] and rep['gate']['all_folds_pos']
    rep['lomo12'] = run_scheme(sh, st, st_na, list(range(1, 13)), lambda t: [m for m in range(1, 13) if m != t])

    # Relation: after removing the per-airport mean over months, published mean vs label mean; baseline bias (Ybar - Pbar) vs proposed correction (O + c - Pbar) (c from the other 11 months)
    cor = {}
    for i, apn in enumerate(APTS):
        yy, oo = Ybar[i, 1:], O[i, 1:]
        bias = Ybar[i, 1:] - Pbar[i, 1:]
        corr_fix = np.array([O[i, m] + sh.c([x for x in range(1, 13) if x != m])[i] - Pbar[i, m] for m in range(1, 13)])
        cor[apn] = dict(corr_level=round(float(np.corrcoef(yy - yy.mean(), oo - oo.mean())[0, 1]), 3),
                        corr_bias_fix=round(float(np.corrcoef(bias, corr_fix)[0, 1]), 3),
                        c_full=round(float(np.mean(yy - oo)), 1), sd_Ybar_minus_O=round(float(np.std(yy - oo, ddof=1)), 1),
                        sd_base_bias=round(float(np.std(bias, ddof=1)), 1))
    rep['relation'] = cor

    # Strong-baseline check (two-fold ensemble for months 1/7), c from the other 11 months; every alpha is reported
    strong = {}
    for mo in (1, 7):
        fp = os.path.join(a.storedir, 'oof', f'ens_{mo}.npy')
        if not os.path.exists(fp):
            strong[mo] = 'missing'; continue
        msk = mi == mo; pe = np.load(fp); assert len(pe) == msk.sum(), (len(pe), msk.sum())
        re = y[msk] - pe; ste = cell_stats(ai[msk], mi[msk], re); ste_na = cell_stats(ai[msk], mi[msk], re, nonA[msk])
        Pe = np.full((10, 13), np.nan); Pe[:, mo] = np.bincount(ai[msk], weights=pe, minlength=10) / ste[0][:, mo]
        she = Shift(O, Ybar, Pe); S = [x for x in range(1, 13) if x != mo]
        res = {}
        for al in ALPHAS:
            cells = fold_cells(she, ste, ste_na, mo, S, al)
            rb, rs, dl = rmse_delta(cells); _, _, dna = rmse_delta(cells, True)
            res[str(al)] = dict(rmse_base=round(rb, 3), delta=round(dl, 3), delta_nonA=round(dna, 3),
                                se=round(boot_se(cells), 3) if al > 0 else 0.0)
        strong[mo] = res
    rep['strong_ens'] = strong
    rep['official_2026_target_months'] = {k: v for k, v in O26.items() if k[-2:] in ('01', '02', '06', '07')}

    tmp = a.out + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(tmp, 'w') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)
    summ = {k: rep[k] for k in ('rows', 'gate', 'relation', 'strong_ens')}
    for sc in ('forward5', 'lomo12'):
        summ[sc] = dict(pooled=rep[sc]['pooled'], per_apt_delta=rep[sc]['per_apt_delta'],
                        folds=[{k: v for k, v in f.items() if k != 'c'} for f in rep[sc]['folds']])
    summ['forward5_c_last'] = rep['forward5']['folds'][-1]['c']
    print(json.dumps(summ, ensure_ascii=False))
    print('ECTX_EVAL_DONE')


if __name__ == '__main__':
    main()
