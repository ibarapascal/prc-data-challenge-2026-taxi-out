"""tabular_term_holdout.py - "hold out one 2025 month" version of the tabular transplant term D (used for the forward validation of the February / June ADS-B transfer).

Adapted from tabular_term.py (the production version of the tabular arms); only the split into training rows and prediction rows differs, everything else is unchanged item by item:
- training rows = 2025 DEP rows of feat.parquet with src != 99 and ym != --hold-ym; prediction rows = rows with ym == --hold-ym (the 2026 test rows are neither read nor used).
- the residual TE (te_fold) uses training-row statistics only (keep_te excludes the held-out month); held-out rows get the "all training months" value, which is the convention for the
  2026 rows in production; categorical codes use the unique values of the training rows only; the oof12/srp_g5_avg residual is as in production.
- features, family columns, TE keys, LightGBM parameters, rounds (mean best_iteration of the two folds in the fcv json x 1.15), seeds, the taxi/delta two-path average and D = 0 on B rows are as in tabular_term.py.
Output: <out>.npz (MVT_ID, base, new, D, B; row order = order of the held-out month's rows in feat; with several seeds D_seeds is also stored) + <out>.json.
Usage: python tabular_term_holdout.py <storedir> <pf.parquet> <fcv.json> <out prefix> --hold-ym 202506 --fam a,b,c,d,e --tekeys te_alst,te_strw,te_tyst,te_alde
      --threads 4 --lr 0.1 --seeds 0,1,2,3,4,5,6,7
Memory: columns are read one at a time and filled into X (polars holds a single column at a time). Estimated peak = X 2.09M x 177 x 4 B 1.48 GB + X[mt] copy 1.35 GB + binning 0.35 GB
+ runtime libraries ~0.6 GB, about 3.8 GB.
"""
import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import tabular_cv as F

os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
FLIGHT_TE = {'te_fl', 'te_flst', 'te_cs'}


def log(*x):
    print(time.strftime('%H:%M:%S'), *x, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('storedir'); ap.add_argument('pf'); ap.add_argument('fcv'); ap.add_argument('out')
    ap.add_argument('--hold-ym', type=int, required=True)
    ap.add_argument('--fam', default='a,b,c,d,e'); ap.add_argument('--tekeys', default='te_alst,te_strw,te_tyst,te_alde')
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--lr', type=float, default=0.1); ap.add_argument('--itmul', type=float, default=1.15)
    ap.add_argument('--seeds', default='0')
    a = ap.parse_args()
    t0 = time.time()
    fams = [x for x in a.fam.split(',') if x and x != 'none']
    tek = [k for k in a.tekeys.split(',') if k]
    assert not (set(tek) & FLIGHT_TE)
    for k in [k for k in F.TE_KEYS if k not in tek]:
        F.TE_KEYS.pop(k)
    assert list(F.TE_KEYS) == tek, (list(F.TE_KEYS), tek)
    fcv = json.load(open(a.fcv, encoding='utf-8'))
    assert fcv['fam'] == fams and fcv['te'] is True, ('fcv json does not match this configuration', fcv['fam'], fcv['te'])
    inf = fcv['info']
    it = {arm: [int(np.mean([inf[f'iters_{fo}_{arm}'][j] for fo in (1, 7)]) * a.itmul) for j in (0, 1)] for arm in ('base', 'new')}
    log('iters (taxi, delta)', it)

    # Memory-lean build: no full polars wide table; columns are read from parquet one at a time and filled into X,
    # so polars holds one column at a time (<= 2.8M rows); pf is gathered by MVT_ID row index (asserting that all training IDs are in pf).
    fp = os.path.join(a.storedir, 'feat.parquet')
    sel = pl.read_parquet(fp, columns=['src'])['src'] != 99

    def fcol(c):
        return pl.read_parquet(fp, columns=[c])[c].filter(sel)

    ids_all = fcol('MVT_ID_mvt')
    n = len(ids_all)
    assert ids_all.n_unique() == n
    p12 = np.load(os.path.join(a.storedir, 'oof12', 'srp_g5_avg.npy'))
    assert len(p12) == n, (len(p12), n)
    ym = fcol('ym').to_numpy()
    hold = ym == a.hold_ym
    istr = ~hold
    assert hold.sum() > 0, f'no rows for ym={a.hold_ym}'
    log(f'train {int(istr.sum()):,} hold {int(hold.sum()):,}')
    pfs = pl.read_parquet_schema(a.pf)
    new = [c for c in pfs if any(c.startswith(f + '_') for f in fams)]
    assert new == [c for c in fcv['new_cols'] if not c.startswith(('te_', 'n_te_'))], 'new-arm columns do not match fcv'
    pid = pl.read_parquet(a.pf, columns=['MVT_ID_mvt'])['MVT_ID_mvt']
    assert pid.n_unique() == len(pid), 'MVT_ID of pf is not unique'
    gi = pl.DataFrame({'MVT_ID_mvt': ids_all}).with_row_index('_r').join(
        pid.to_frame().with_row_index('_p'), on='MVT_ID_mvt', how='left').sort('_r')['_p']
    assert gi.null_count() == 0, f'{gi.null_count()} training IDs are not in pf'
    del pid

    def pcol(c):
        return pl.read_parquet(a.pf, columns=[c])[c].gather(gi)

    miss = pcol(new[0]).is_null().to_numpy()
    log(f'pf indexed; first new col null rate train {miss[istr].mean():.4f} hold {miss[hold].mean():.4f}')
    y = fcol('TAXITIME_SEC_mvt').to_numpy().astype(np.float64)
    adep = fcol('ADEP_mvt')
    B = ((fcol('flt_null') == 1) & (adep == 'LIRF') & (fcol('mvt_delay') > 3600)).fill_null(False).to_numpy()
    A = istr & (np.nan_to_num(y, nan=0.0) > 10800) & ~B
    keep_te = istr & ~A & ~B          # the held-out month does not enter the TE statistics
    r12 = y - p12
    tecols = list(F.TE_KEYS) + [f'n_{k}' for k in F.TE_KEYS]
    cols_new = F.SRP + new + tecols
    assert cols_new[len(F.SRP):] == fcv['new_cols']
    srp = set(F.SRP)
    X = np.full((n, len(cols_new)), np.nan, dtype=np.float32)
    for j, c in enumerate(cols_new):
        if c in tecols:
            continue
        s = fcol(c) if c in srp else pcol(c)
        if c in F.CAT:
            s = s.cast(pl.String)
            u = s.filter(pl.Series(istr)).drop_nulls().unique().sort().to_list()   # codes from the training rows only
            X[:, j] = s.replace_strict(dict(zip(u, range(len(u)))), default=None, return_dtype=pl.Float32).to_numpy()
        else:
            X[:, j] = s.cast(pl.Float32).to_numpy()
        del s
    kcols = list(dict.fromkeys(sum(F.TE_KEYS.values(), [])))
    tef = pl.read_parquet(fp, columns=kcols).filter(sel).with_columns(pl.Series('mo', ym - 202500))
    for k, keys in F.TE_KEYS.items():
        t_, n_ = F.te_fold(tef, r12, keep_te, keys)
        X[:, cols_new.index(k)] = t_; X[:, cols_new.index('n_' + k)] = n_
    del tef
    dl = fcol('delta').to_numpy().astype(np.float64)
    mao = fcol('mvt_aobt3').to_numpy().astype(np.float64)
    aob = fcol('AOBT_3_flt').is_not_null().to_numpy()
    mvr = hold & aob
    assert (np.isfinite(mao[mvr])).all(), 'AOBT_3 is non-null but mvt_aobt3 is null'
    ids = ids_all.to_numpy()[hold]
    apt = adep.to_numpy()[hold]
    del ids_all, adep, gi; gc.collect()
    log(f'X built {X.shape} {X.nbytes / 1e9:.2f}GB')
    nb = len(F.SRP)
    cat_b = [j for j, c in enumerate(F.SRP) if c in F.CAT]
    cat_n = [j for j, c in enumerate(cols_new) if c in F.CAT]
    P = dict(objective='regression', metric='rmse', learning_rate=a.lr, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=a.threads, seed=0, max_bin=63)
    seeds = [int(x) for x in a.seeds.split(',') if x != '']
    Xh = X[hold]
    per = {'base': [], 'new': []}
    for sd in seeds:
        for arm, ncol, cat in (('base', nb, cat_b), ('new', len(cols_new), cat_n)):
            out = {}
            for j, tgt in enumerate(('taxi', 'delta')):
                mt = istr if tgt == 'taxi' else istr & np.isfinite(dl)
                yt = y if tgt == 'taxi' else dl
                m = lgb.train(dict(P, seed=sd), lgb.Dataset(X[mt, :ncol], yt[mt], categorical_feature=cat, free_raw_data=True), it[arm][j])
                out[tgt] = m.predict(Xh[:, :ncol])
                del m; gc.collect()
                log(f'seed {sd} {arm} {tgt}: iters {it[arm][j]} {time.time() - t0:.0f}s')
            pt = out['taxi']; pd_ = pt.copy(); mv = mvr[hold]; pd_[mv] = mao[hold][mv] + out['delta'][mv]
            per[arm].append((pt + pd_) / 2)
    pred = {arm: v[0] if len(v) == 1 else np.mean(v, axis=0) for arm, v in per.items()}
    Bh = B[hold]
    D = pred['new'] - pred['base']; D[Bh] = 0.0
    extra = {}
    if len(seeds) > 1:
        Dk = np.stack([n_ - b_ for n_, b_ in zip(per['new'], per['base'])]); Dk[:, Bh] = 0.0
        extra = dict(D_seeds=Dk.astype(np.float32), seeds=np.array(seeds))
    tmp = a.out + '.tmp'
    np.savez_compressed(tmp + '.npz', MVT_ID=ids, base=pred['base'], new=pred['new'], D=D, B=Bh, **extra)
    yh = y[hold]
    ok = ~Bh & np.isfinite(yh)
    res = {'hold_ym': a.hold_ym, 'fam': fams, 'tekeys': tek, 'iters': it, 'itmul': a.itmul, 'lr': a.lr, 'seeds': seeds,
           'n_train': int(istr.sum()), 'n_hold': int(hold.sum()), 'n_B': int(Bh.sum()), 'delta_rows': int(mvr.sum()),
           'pf_null_hold': round(float(miss[hold].mean()), 4),
           'D_all': {'mean': round(float(D[~Bh].mean()), 3), 'rms': round(float(np.sqrt((D[~Bh] ** 2).mean())), 2)},
           'arm_rmse_hold_nonB': {arm: round(float(np.sqrt(((pred[arm][ok] - yh[ok]) ** 2).mean())), 3) for arm in pred},
           'cells': {x: {'n': int(((apt == x) & ~Bh).sum()), 'mean_D': round(float(D[(apt == x) & ~Bh].mean()), 2),
                         'rms_D': round(float(np.sqrt((D[(apt == x) & ~Bh] ** 2).mean())), 2)} for x in F.APTS if ((apt == x) & ~Bh).any()},
           'finite': bool(np.isfinite(pred['base']).all() and np.isfinite(pred['new']).all()), 'sec': round(time.time() - t0)}
    with open(tmp + '.json', 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    os.replace(tmp + '.npz', a.out + '.npz')
    os.replace(tmp + '.json', a.out + '.json')
    log('done', res['D_all'], res['arm_rmse_hold_nonB'], 'sec', res['sec'])


if __name__ == '__main__':
    main()
