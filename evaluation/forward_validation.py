"""forward_validation.py - forward validation of transferring the ADS-B residual layer to February / June (decision rule fixed before the computation).

On a held-out 2025 month (06 or 02), simulate the two paths of the final chain and compare their RMSE against the truth:
- ADS-B path (what the full version does for that month; identifiers containing "adsb_path" or "KEEP_ADSB" below refer to this path): gated rows
  (ab_obs_first not null) = variant_full (RESID_TK=all) + stand correction (sc_table_all_full, outside the keep cells); the keep cells EGLL-06/EGLL-07/LIRF and the cells
  actually reverted by the drift gate in the full version (EDDF-02, LSZH-06) take variant_simple (no track context); non-gated rows = rint(base) + w*D (rows outside class B,
  clip 30); finally a global lower bound of 119. The clip/round of every layer is the same as in train_resid.write_sub, the applysc/merge steps of
  residual_context.py, merge_predictions.py and floor_clip.py.
- Fallback path (no ADS-B in that month; identifiers containing "safe"): all rows = rint(base) + w*D (rows outside class B, clip 30), no lower bound (as in the fallback version file).
The residual layer is trained as in the production residual step (functions of train_resid.py called unchanged; the environment is given by the shell: art, hyb:01,
threshold .15, STAND=fix, CELLFB=pred, WINTER=11+12 for seven airports, WINTER_TEST=01,02, 8 seeds, 600 rounds). The only differences from production:
- Test rows = the DEP rows of the held-out 2025 month (days with features in --test-feats), not the 2026 test file; their base = oof12/srp_g5_delta linearly calibrated per airport
  to the scale of the averaged baseline (same base convention as the winter pool in load_pool; the calibration is fitted on 1/7 of the training rows), then rint,
  in place of the retrained baseline file.
- The held-out month's labels enter no training input built by this script: the art table excludes the held-out month; residual training rows exclude the --buffer days
  (training days adjacent to the held-out month).
- The tabular term D comes from tabular_term_holdout.py (both arms retrained with that month held out, 8 seeds); with --tab none, D = 0.
- The drift gate is not rerun: the reverted cells are copied from the full version's actual decision (--revert).
Output: --out json (decision, overall / non-extreme / per-airport / gated-row RMSE, day-block bootstrap SE, 5 day folds, secondary variants) and --rows parquet (row level, for checking).
Usage (in the final/ work dir, environment as E_RESID + E_FIN + W7 of run_stage_chain.sh):
  python forward_validation.py --month 06 --buffer 2025-07-01 --test-feats feats_tmafy --tk-test fv/tk_fv06.parquet --tab fv/tabterm_ho06.npz --out fv/forward_val_06.json
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_resid as T  # noqa: E402

assert Path(T.__file__).resolve().parent == HERE, (T.__file__, HERE)
t0 = time.time()


def log(*x):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # macOS: bytes; records the process peak at each stage to check the memory estimate
    print(f'[{time.time() - t0:7.0f}s maxrss {rss:.2f}GB]', *x, flush=True)


def day_folds(dates, nfold=5, seed=0):  # same fold definition as day_folds in residual_context.py
    days = sorted(set(dates))
    perm = np.random.default_rng(seed).permutation(len(days))
    fold_of = {days[i]: k % nfold for k, i in enumerate(perm)}
    return np.array([fold_of[d] for d in dates])


def load_test(v, ym, feats_dir, tk_file):
    """Held-out month DEP rows (all, including isA) + ADS-B features from feats_dir + tk; base = oof12 linearly calibrated per airport (as in load_pool)."""
    d = T.al_icao(pl.read_parquet(Path('n2') / 'feat.parquet', columns=T.BASE_COLS))
    b12 = np.load(Path('n2') / 'oof12' / f'{T.POOL_BASE}.npy')
    dtr = d.filter(pl.col('src') != 99)
    assert len(b12) == dtr.height, (len(b12), dtr.height)
    dtr = dtr.with_columns(pl.Series('b12', b12))
    j = v.select('MVT_ID_mvt', 'ADEP_mvt', 'base').join(dtr.select('MVT_ID_mvt', 'b12'), on='MVT_ID_mvt', how='inner')
    cal = {}
    for x in T.APTS:
        s = j.filter(pl.col('ADEP_mvt') == x)
        if T.POOL_CAL == 'lin' and s.height > 100:
            be, al_ = np.polyfit(s['b12'].to_numpy(), s['base'].to_numpy(), 1)
            cal[x] = (float(al_), float(be))
        else:
            cal[x] = (0.0, 1.0)
    mm = f'{ym - 202500:02d}'
    fs = [f for f in sorted(Path(feats_dir).glob('2025-*.parquet')) if f.stem[5:7] == mm]
    assert fs, f'{feats_dir} has no feature file for 2025-{mm}'
    ab = pl.concat([pl.read_parquet(f) for f in fs], how='diagonal_relaxed').unique('MVT_ID_mvt')
    days = sorted(ab['date'].unique().to_list())
    p = T.add_art(dtr.filter(pl.col('ym') == ym), d.filter(pl.col('src') != 99), exclude_ym=ym).with_columns(
        pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d').alias('date'))
    n_month = p.height
    p = p.filter(pl.col('date').is_in(days)).join(ab.drop('date'), on='MVT_ID_mvt', how='left')
    a_, b = p['ADEP_mvt'].to_numpy(), p['b12'].to_numpy().astype(float).copy()
    for x, (al_, be) in cal.items():
        b[a_ == x] = al_ + be * b[a_ == x]
    p = T.add_tk(T.prep(p.with_columns(pl.Series('base', np.rint(b)))), tk_file)
    p = p.select([pl.col(c).cast(t) for c, t in v.schema.items()])
    info = dict(month_rows=n_month, test_rows=p.height, days=len(days), first_day=days[0], last_day=days[-1],
                gate=round(float(p['gate'].mean()), 4), cal={k: [round(x, 4) for x in c] for k, c in cal.items()},
                tk_nonnull_gated=int((p['gate'] & p['tk_stop_code'].is_not_null()).sum()))
    return p, info


def rmse(p, y, m):
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.any() else None


def compare(pa, pb, y, dates, m, nboot=2000, seed=0):
    """pa = ADS-B path (adsb_path), pb = fallback path. dRMSE = RMSE(pa) - RMSE(pb) (positive = ADS-B path worse); day-block bootstrap SE; SE of the day-paired dMSE; 5 day folds."""
    dd = np.array(dates)[m]
    ea, eb = (pa[m] - y[m]) ** 2, (pb[m] - y[m]) ** 2
    days, inv = np.unique(dd, return_inverse=True)
    Sa, Sb, n = np.bincount(inv, ea), np.bincount(inv, eb), np.bincount(inv).astype(float)
    ra, rb = np.sqrt(Sa.sum() / n.sum()), np.sqrt(Sb.sum() / n.sum())
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=nboot).astype(float)
    bd = np.sqrt(W @ Sa / (W @ n)) - np.sqrt(W @ Sb / (W @ n))
    dm = Sa - Sb
    fd = day_folds(list(dd))
    fold = []
    for k in range(5):
        mk = fd == k
        fold.append(round(float(np.sqrt(ea[mk].mean()) - np.sqrt(eb[mk].mean())), 4))
    return dict(n=int(m.sum()), days=int(len(days)), rmse_adsb_path=round(float(ra), 4), rmse_safe=round(float(rb), 4),
                d_rmse=round(float(ra - rb), 4), se_boot=round(float(bd.std(ddof=1)), 4),
                boot_q025_q975=[round(float(np.quantile(bd, .025)), 4), round(float(np.quantile(bd, .975)), 4)],
                d_mse=round(float(dm.sum() / n.sum()), 2), se_mse_dayblock=round(float(dm.std(ddof=1) * np.sqrt(len(days)) / n.sum()), 2),
                fold_d_rmse=fold, folds_adsb_path_worse=int(sum(x > 0 for x in fold)), folds_adsb_path_better=int(sum(x < 0 for x in fold)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', required=True, choices=['02', '06'])
    ap.add_argument('--buffer', default='', help='comma-separated dates removed from the residual training rows (adjacent to the held-out month)')
    ap.add_argument('--test-feats', required=True)
    ap.add_argument('--tk-test', required=True)
    ap.add_argument('--tab', default='none')
    ap.add_argument('--sc', default='v22/sc_table_all_full.parquet')
    ap.add_argument('--keep', default='EGLL-06,EGLL-07,LIRF')
    ap.add_argument('--revert', default='EDDF-02,LSZH-06')
    ap.add_argument('--floor', type=float, default=119)
    ap.add_argument('--w', type=float, default=0.25)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--rounds', type=int, default=600)
    ap.add_argument('--feats', default=os.environ.get('RESID_FEATS', 'feats_tma'))
    ap.add_argument('--nowinter', action='store_true', help='(02 only) also run the secondary variant with WINTER=off')
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', default='')
    a = ap.parse_args()
    ym = 202500 + int(a.month)
    buf = [x for x in a.buffer.split(',') if x]
    log('env', {k: v for k, v in os.environ.items() if k.startswith('RESID_')})
    assert T.MONTH == 'hyb:01' and T.CELLFB == 'pred' and T.STAND == 'fix' and 'art' in T.EXTRA, 'environment is not the production configuration'
    _add_art = T.add_art
    T.add_art = lambda v, full, exclude_ym=None: _add_art(v, full.filter(pl.col('ym') != ym), exclude_ym)  # the art table does not see the held-out month's labels
    T.TK_LOAD = True
    T.PRED_FB = T.CELLFB == 'pred'   # test-side grid-cell fallback on the prediction side only (training is unchanged)

    v = T.load_train('n2', a.feats)
    n0 = v.height
    v = v.filter(~pl.col('date').is_in(buf))
    log('train rows', n0, '-> after buffer', v.height, 'buffer', buf)
    te, tinfo = load_test(v, ym, a.test_feats, a.tk_test)
    log('test', tinfo)
    if T.WINTER != 'off' and a.month in T.WINTER_TEST:
        T.POOL, _ = T.load_pool('n2', v, T.WINTER.split('+'))
    trv = v.filter(~pl.col('isA'))
    kw = dict(rounds=a.rounds, seeds=tuple(range(a.seeds)))
    fits = {}
    for nm, tk in (('variant_simple', []), ('variant_full', T.tk_cols('all'))):
        T.set_tk(tk)
        fits[nm] = T.fit_mixed(trv, te, **kw)
        log('fit', nm, 'done', json.dumps(fits[nm][1], ensure_ascii=False, default=str)[:600])
    if a.nowinter and a.month == '02':
        w0 = T.WINTER
        T.WINTER = 'off'
        for nm, tk in (('variant_simple', []), ('variant_full', T.tk_cols('all'))):
            T.set_tk(tk)
            fits[nm + '_nowinter'] = T.fit_mixed(trv, te, **kw)
            log('fit', nm, 'nowinter done')
        T.WINTER = w0
    T.set_tk([])

    y = te['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    b = te['base'].to_numpy().astype(float)          # already rint, so integer-valued like the production baseline file
    g = te['gate'].to_numpy()
    nA = ~te['isA'].to_numpy()
    apt = te['ADEP_mvt'].to_numpy()
    dates = te['date'].to_list()
    key = np.array([f'{x}-{a.month}' for x in apt])
    kl = [k for k in a.keep.split(',') if k]
    keep = np.array([(k in kl) or (k[:4] in kl) for k in key])
    rv = np.isin(key, [k for k in a.revert.split(',') if k])
    tab = dict(pl.read_parquet(a.sc).iter_rows())
    corr = np.array([tab.get(k_, 0.0) for k_ in te['stand_key'].to_list()]) * (g & ~keep)
    ids = te['MVT_ID_mvt'].to_numpy()
    if a.tab == 'none':
        D, B = np.zeros(len(y)), ((te['flt_null'] == 1) & (te['ADEP_mvt'] == 'LIRF') & (te['mvt_delay'] > 3600)).fill_null(False).to_numpy()
    else:
        z = np.load(a.tab)
        pos = {x: i for i, x in enumerate(z['MVT_ID'])}
        ti = np.array([pos[x] for x in ids])
        D, B = z['D'][ti].astype(float), z['B'][ti].astype(bool)
    assert np.isfinite(D).all() and (D[B] == 0).all()

    def chain(r21, r22, use_sc=True, use_rv=True, use_floor=True, Dv=D):
        g21, g22 = r21 * g, r22 * g
        p21 = np.round(np.clip(b + g21, 0, None))
        p22s = np.round(np.clip(b + g22 + (corr if use_sc else 0), 0, None))
        cand = np.where(keep, p21, p22s)
        gate_m = g22 != 0                              # as in merge_predictions.py: gate = (variant_full res != 0)
        ng = np.where(B, b, np.clip(np.rint(b + a.w * Dv), 30, None))
        p = np.where(gate_m, cand, ng)
        if use_rv:
            p = np.where(gate_m & rv, p21, p)
        return np.maximum(p, a.floor) if use_floor else p

    safe = np.where(B, b, np.clip(np.rint(b + a.w * D), 30, None))
    r21, r22 = fits['variant_simple'][0], fits['variant_full'][0]
    adsb_path = chain(r21, r22)
    allm = np.ones(len(y), bool)
    gm = (r22 * g) != 0
    out = dict(month=a.month, ym=ym, buffer=buf, test=tinfo, train_rows=v.height, train_days=int(v['date'].n_unique()),
               tab=a.tab, w=a.w, floor=a.floor, keep=a.keep, revert=a.revert, seeds=a.seeds, rounds=a.rounds,
               env={k: v_ for k, v_ in os.environ.items() if k.startswith('RESID_')},
               fit_info={k: f[1] for k, f in fits.items()},
               changed_rows=int((adsb_path != safe).sum()), changed_share=round(float((adsb_path != safe).mean()), 4), gated_rows=int(gm.sum()),
               primary=compare(adsb_path, safe, y, dates, allm))
    pr = out['primary']
    switch = bool(pr['d_rmse'] > 0 and pr['d_rmse'] >= 2 * pr['se_boot'] and pr['folds_adsb_path_worse'] == 5)
    out['decision'] = dict(rule='switch to fallback iff dRMSE>0 and dRMSE>=2*SE_boot and dRMSE>0 in all 5 day folds; otherwise keep the ADS-B path', switch_to_safe=switch,
                           verdict='SAFE' if switch else 'KEEP_ADSB')
    out['nonA'] = compare(adsb_path, safe, y, dates, nA)
    out['gated_only'] = compare(adsb_path, safe, y, dates, gm)
    out['by_apt'] = {x: dict(**compare(adsb_path, safe, y, dates, apt == x, nboot=500), gated=int((gm & (apt == x)).sum()),
                             gate_rate=round(float(g[apt == x].mean()), 4))
                     for x in T.APTS if (apt == x).any()}
    out['secondary'] = {
        'no_revert': compare(chain(r21, r22, use_rv=False), safe, y, dates, allm, nboot=500),
        'no_sc': compare(chain(r21, r22, use_sc=False), safe, y, dates, allm, nboot=500),
        'no_floor': compare(chain(r21, r22, use_floor=False), safe, y, dates, allm, nboot=500),
        'D0_both': compare(chain(r21, r22, Dv=np.zeros(len(y))), np.where(B, b, np.clip(b, 30, None)), y, dates, allm, nboot=500),
        'only_simple_gated': compare(chain(r21, r21, use_sc=False), safe, y, dates, allm, nboot=500),
        'base_vs_safe': dict(rmse_base=round(rmse(b, y, allm), 4), rmse_safe=round(rmse(safe, y, allm), 4)),
    }
    if 'variant_full_nowinter' in fits:
        nw = chain(fits['variant_simple_nowinter'][0], fits['variant_full_nowinter'][0])
        out['secondary']['nowinter_vs_safe'] = compare(nw, safe, y, dates, allm, nboot=500)
        out['secondary']['adsb_path_vs_nowinter'] = compare(adsb_path, nw, y, dates, allm, nboot=500)
        w7 = np.isin(apt, T.WINTER_APTS)
        out['secondary']['adsb_path_vs_nowinter_w7'] = compare(adsb_path, nw, y, dates, w7, nboot=500)
    out['sec'] = round(time.time() - t0)
    out['maxrss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 3)
    tmp = a.out + '.tmp'
    json.dump(out, open(tmp, 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str)
    if a.rows:
        pl.DataFrame({'MVT_ID_mvt': ids, 'date': dates, 'ADEP_mvt': apt, 'y': y, 'base': b, 'gate': g, 'isA': ~nA, 'B': B,
                      'r21': r21, 'r22': r22, 'corr': corr, 'D': D, 'p_v2': adsb_path, 'p_safe': safe}).write_parquet(a.rows + '.tmp')
        os.replace(a.rows + '.tmp', a.rows)
    os.replace(tmp, a.out)
    log('decision', out['decision'], 'primary', pr)


if __name__ == '__main__':
    main()
