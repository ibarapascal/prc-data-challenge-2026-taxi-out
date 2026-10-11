"""tabular_term: 2026 arm predictions of the shrunken tabular transplant. Base arm / new arm are fitted on the full year on the basis of tabular_cv.py and predict only the 2026 ranking rows.

Basis (aligned item by item with tabular_cv.py; definitions are imported from tabular_cv.py in the same directory, not copied):
- Features: base arm = SRP; new arm = SRP + the --fam family columns of pf.parquet + the non-flight-level residual TE of --tekeys (same definition as te_fold: training rows
  leave their own month out, ranking rows use all 12 training months; residual = y - oof12/srp_g5_avg; A/B rows do not enter the TE statistics).
  Flight-level TE (te_fl / te_flst / te_cs) are not allowed (asserted).
- Category encoding: only from the sorted unique values of the training rows (as in tabular_cv.py); categories unseen in the ranking rows = NaN.
- LightGBM parameters as in tabular_cv.py (255 leaves, lr --lr, ff .7, bagging .8, lambda2 10, min_data 100, max_bin 63, seed 0), same seed for both arms;
  the full-year number of rounds is fixed = mean best_iteration of the two folds in the fcv json x 1.15 (the baseline's full-year fitting convention), no early stopping.
- The taxi and delta paths are averaged; the delta path is used only on ranking rows with a non-null AOBT_3 (same basis as the baseline's full-year fit), otherwise the taxi path.
- B rows (LIRF, flt_null=1, mvt_delay>3600) are replaced by the averaged baseline in both arms -> D = 0 (B=True in the npz; not changed when merging).
- --seeds: default 0 = original single-seed setting (output as before); with several seeds each arm is averaged over the seeds,
  D = difference of the two arm means, the npz additionally stores the per-seed D_seeds (float32), the json additionally reports the label-free sigma_D^2 and the
  expected gain w^2 sigma_D^2 (1 - 1/K) (with --ref pointing to the npz of the single-seed run it also reports the seed-0 alignment and the row-wise difference to it).
Output: <out>.npz (MVT_ID, base, new, D, B; ranking row order = order of the src==99 rows in feat) and <out>.json (iterations, row counts, D statistics per cell).
Usage: python tabular_term.py <storedir: baseline feature-store dir, n2/> <pf.parquet> <fcv.json> <out prefix> --fam a,b,c,d,e --tekeys te_alst,te_strw,te_tyst,te_alde [--threads 4] [--lr 0.1] [--seeds 0,1,...,7 --ref tabterm_all_te2.npz]
Memory (estimate): X of 2.43 M rows x about 180 columns float32 = 1.75 GB plus the LightGBM training copy and binning, peak about 6 GB (5.2 GB measured for the training-row version in tabular_cv.py).
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
    ap.add_argument('--fam', default='a,b,c,d,e'); ap.add_argument('--tekeys', default='te_alst,te_strw,te_tyst,te_alde')
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--lr', type=float, default=0.1); ap.add_argument('--itmul', type=float, default=1.15)
    ap.add_argument('--seeds', default='0'); ap.add_argument('--ref', default=''); ap.add_argument('--w', type=float, default=0.25)
    a = ap.parse_args()
    t0 = time.time()
    fams = [x for x in a.fam.split(',') if x and x != 'none']
    tek = [k for k in a.tekeys.split(',') if k]
    assert not (set(tek) & FLIGHT_TE), 'flight-level TE are not allowed in the 2026 candidate'
    for k in [k for k in F.TE_KEYS if k not in tek]:
        F.TE_KEYS.pop(k)
    assert list(F.TE_KEYS) == tek, (list(F.TE_KEYS), tek)
    fcv = json.load(open(a.fcv, encoding='utf-8'))
    assert fcv['fam'] == fams and fcv['te'] is True, ('fcv json does not match the setting', fcv['fam'], fcv['te'])
    inf = fcv['info']
    it = {arm: [int(np.mean([inf[f'iters_{fo}_{arm}'][j] for fo in (1, 7)]) * a.itmul) for j in (0, 1)] for arm in ('base', 'new')}
    log('iters (taxi, delta)', it)

    aux = ['MVT_ID_mvt', 'src', 'ym', 'TAXITIME_SEC_mvt', 'delta', 'aobt_sched', 'MVT_TIME_UTC_mvt', 'FLIGHT_mvt', 'AOBT_3_flt']
    cols = list(dict.fromkeys(F.SRP + aux))
    fp = os.path.join(a.storedir, 'feat.parquet')
    d = pl.read_parquet(fp, columns=cols).with_row_index('_i')
    srcv = d['src'].to_numpy()
    ntr = int((srcv != 99).sum()); nrk = d.height - ntr
    assert (srcv[ntr:] == 99).all(), 'ranking rows of feat.parquet are not at the end (the baseline sorts by src; they must be last)'
    log(f'train {ntr:,} ranking {nrk:,}')
    assert d['MVT_ID_mvt'].n_unique() == d.height, 'MVT_ID of train + ranking is not unique'
    pf = pl.read_parquet(a.pf)
    new = [c for c in pf.columns if any(c.startswith(f + '_') for f in fams)]
    assert new == [c for c in fcv['new_cols'] if not c.startswith(('te_', 'n_te_'))], 'new-arm columns differ from fcv'
    keep = ['MVT_ID_mvt'] + new + ['k_cs', 'e_loc_hour']
    pfk = pf.select(list(dict.fromkeys(keep)))
    del pf
    assert pfk['MVT_ID_mvt'].n_unique() == pfk.height, 'MVT_ID of pf is not unique'
    h0 = d.height
    d = d.join(pfk, on='MVT_ID_mvt', how='left').sort('_i')
    assert d.height == h0, f'row count changed after joining pf: {h0} -> {d.height}'
    del pfk; gc.collect()
    miss = d[new[0]].is_null().to_numpy()
    log(f'pf joined; first new col null rate train {miss[:ntr].mean():.4f} ranking {miss[ntr:].mean():.4f}')
    d = d.with_columns((pl.col('ym') - 202500).alias('mo'))
    y = d['TAXITIME_SEC_mvt'].to_numpy().astype(np.float64)
    istr = np.arange(d.height) < ntr
    p12 = np.load(os.path.join(a.storedir, 'oof12', 'srp_g5_avg.npy'))
    assert len(p12) == ntr
    B = ((d['flt_null'] == 1) & (d['ADEP_mvt'] == 'LIRF') & (d['mvt_delay'] > 3600)).fill_null(False).to_numpy()
    A = istr & (np.nan_to_num(y, nan=0.0) > 10800) & ~B
    keep_te = istr & ~A & ~B
    r12 = np.zeros(d.height); r12[:ntr] = y[:ntr] - p12
    tecols = list(F.TE_KEYS) + [f'n_{k}' for k in F.TE_KEYS]
    cols_new = F.SRP + new + tecols
    assert cols_new[len(F.SRP):] == fcv['new_cols'], 'new-arm column order differs from fcv'
    X = np.full((d.height, len(cols_new)), np.nan, dtype=np.float32)
    for j, c in enumerate(cols_new):
        if c in tecols:
            continue
        s = d[c]
        if c in F.CAT:
            s = s.cast(pl.String)
            u = s[:ntr].drop_nulls().unique().sort().to_list()   # encode from the training rows only (as in tabular_cv.py)
            X[:, j] = s.replace_strict(dict(zip(u, range(len(u)))), default=None, return_dtype=pl.Float32).to_numpy()
        else:
            X[:, j] = s.cast(pl.Float32).to_numpy()
    for k, keys in F.TE_KEYS.items():
        t_, n_ = F.te_fold(d, r12, keep_te, keys)
        X[:, cols_new.index(k)] = t_; X[:, cols_new.index('n_' + k)] = n_
    dl = d['delta'].to_numpy().astype(np.float64)
    mao = d['mvt_aobt3'].to_numpy().astype(np.float64)
    aob = d['AOBT_3_flt'].is_not_null().to_numpy()
    mvr = ~istr & aob
    assert (np.isfinite(mao[mvr])).all(), 'AOBT_3 is not null but mvt_aobt3 is null'
    ids = d['MVT_ID_mvt'].to_numpy()[ntr:]
    apt = d['ADEP_mvt'].to_numpy()[ntr:]; mo26 = d['mo'].to_numpy()[ntr:]
    del d; gc.collect()
    log(f'X built {X.shape} {X.nbytes / 1e9:.2f}GB; te {tecols}')
    nb = len(F.SRP)
    cat_b = [j for j, c in enumerate(F.SRP) if c in F.CAT]
    cat_n = [j for j, c in enumerate(cols_new) if c in F.CAT]
    P = dict(objective='regression', metric='rmse', learning_rate=a.lr, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=a.threads, seed=0, max_bin=63)
    seeds = [int(x) for x in a.seeds.split(',') if x != '']
    per = {'base': [], 'new': []}
    for sd in seeds:
        for arm, ncol, cat in (('base', nb, cat_b), ('new', len(cols_new), cat_n)):
            out = {}
            for j, tgt in enumerate(('taxi', 'delta')):
                mt = istr if tgt == 'taxi' else istr & np.isfinite(dl)
                yt = y if tgt == 'taxi' else dl
                m = lgb.train(dict(P, seed=sd), lgb.Dataset(X[mt, :ncol], yt[mt], categorical_feature=cat, free_raw_data=True), it[arm][j])
                out[tgt] = m.predict(X[ntr:, :ncol])
                del m; gc.collect()
                log(f'seed {sd} {arm} {tgt}: iters {it[arm][j]} {time.time() - t0:.0f}s')
            pt = out['taxi']; pd_ = pt.copy(); mv = mvr[ntr:]; pd_[mv] = mao[ntr:][mv] + out['delta'][mv]
            per[arm].append((pt + pd_) / 2)
    pred = {arm: v[0] if len(v) == 1 else np.mean(v, axis=0) for arm, v in per.items()}
    Br = B[ntr:]
    D = pred['new'] - pred['base']; D[Br] = 0.0
    extra, sk = {}, {}
    if len(seeds) > 1:
        Dk = np.stack([n_ - b_ for n_, b_ in zip(per['new'], per['base'])]); Dk[:, Br] = 0.0
        vD = Dk.var(axis=0, ddof=1); K = len(seeds)
        extra = dict(D_seeds=Dk.astype(np.float32), seeds=np.array(seeds))
        sk = {'seeds': seeds, 'sigmaD2_mean_s2_nonB': round(float(vD[~Br].mean()), 1),
              'exp_gain_w_mse_nonB_rows': round(float(a.w ** 2 * vD[~Br].mean() * (1 - 1 / K)), 2),
              'meanD2_per_seed_nonB': [round(float((x[~Br] ** 2).mean()), 1) for x in Dk]}
        if a.ref:
            z = np.load(a.ref)
            assert (z['MVT_ID'] == ids).all(), 'ref row order differs'
            s0 = seeds.index(0) if 0 in seeds else None
            sk['ref'] = {'seed0_D_max_abs': round(float(np.abs(Dk[s0] - z['D']).max()), 4) if s0 is not None else None,
                         'wdiff_vs_ref_mean_d2_nonB': round(float(((a.w * (D - z['D']))[~Br] ** 2).mean()), 2)}
    np.savez_compressed(a.out + '.npz', MVT_ID=ids, base=pred['base'], new=pred['new'], D=D, B=Br, **extra)
    cells = {}
    for ap_ in F.APTS:
        for mm in (101, 107):
            mk = (apt == ap_) & (mo26 == mm) & ~Br
            if mk.any():
                cells[f'{ap_}-{mm - 100:02d}'] = {'n': int(mk.sum()), 'mean_D': round(float(D[mk].mean()), 2), 'rms_D': round(float(np.sqrt((D[mk] ** 2).mean())), 2)}
    res = {'fam': fams, 'tekeys': tek, 'iters': it, 'itmul': a.itmul, 'lr': a.lr, 'n_train': ntr, 'n_ranking': nrk, 'n_B': int(Br.sum()),
           'delta_rows': int(mvr.sum()), 'pf_null_ranking': round(float(miss[ntr:].mean()), 4),
           'D_all': {'mean': round(float(D[~Br].mean()), 3), 'rms': round(float(np.sqrt((D[~Br] ** 2).mean())), 2)}, 'cells': cells,
           'finite': bool(np.isfinite(pred['base']).all() and np.isfinite(pred['new']).all()), 'sec': round(time.time() - t0), **({'seedavg': sk} if sk else {})}
    with open(a.out + '.json', 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    log('done', res['D_all'], 'sec', res['sec'])


if __name__ == '__main__':
    main()
