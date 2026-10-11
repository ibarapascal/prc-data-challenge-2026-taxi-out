"""catboost_seeds_cv: 2025 two-fold OOF comparison of 3 vs 8 seeds for the CatBoost member (cat_delta) of ensemble B in the averaged baseline (pre-registered).

Setup (same as the earlier baseline OOF): the feature store of the early phase (night2/feat.parquet; read by columns and filter, never loaded whole), the same two leave-one-month-out
folds as baseline_models.fold() (2025-01, 2025-07; training = the other 11 months), LOWMEM=1.
The member recipe matches, item by item, the cat branch of baseline_models.stage_fullens on the 2026 side: featset = exp_meta('cat').fs (sr), fixed rounds it/itd = exp_meta('cat')
(mean of the two-fold early-stopping rounds x 1.15 = 2846/3436), lr .08, depth 8, border 128, RMSE, random_seed=s; categorical columns use int32 codes from a mapping shared by training and
evaluation, where taxi builds the mapping from all training rows and delta from the training rows that have a delta (as in the cframe of fullens). cat_delta = taxi for rows without delta,
mvt_aobt3 + delta for rows with delta.
The LIRF sub-model override on B rows (LIRF rule rows) does not depend on the seed: the historical cat_delta OOF values are used on B rows.
Evaluation: baseline_K = 1/2*ens_v5b + 1/2*(ens_v6 + (2/7)*(cat_K - cat_delta_hist)), cat_K = mean of seeds 0..K-1. Primary criterion = RMSE over all rows of both folds pooled,
delta = RMSE(baseline_{k8}) - RMSE(baseline_{k3}) <= 0 passes the gate (rule fixed beforehand).
Outputs: <work>/parts/cat_{taxi,delta}_s<s>_<mo>.npy (per seed, written to .tmp first and renamed; restartable), <work>/va_<mo>.npz, <work>/catseeds_cv.json.
Usage:
  python catboost_seeds_cv.py --code <dir containing baseline_models.py> --work <work dir with feat.parquet and a results.tsv symlink> --hist <dir of historical OOF files>
                     [--folds 1,7] [--seeds 8] [--threads 6] [--k3 3 --k8 8] [--smoke-iters N] (smoke test: rounds set to N, rounds not asserted)
                     [--eval-only] (evaluate the parts already on disk) [--train-only] (train only, for running one fold on a separate machine)
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np


def log(*x):
    print(time.strftime('%H:%M:%S'), *x, flush=True)


def save(fp, arr):
    tmp = fp + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, fp)


def res_rows(res, exp):
    out = []
    for line in open(res):
        p = line.rstrip('\n').split('\t')
        if p[0] == exp:
            out.append(dict(fold=int(p[1]), rmse=float(p[2]), extra=json.loads(p[5])))
    return out


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def block_se(d, blocks):
    """d = per-row e_new^2 - e_old^2; SE of mean(d) by block (day)."""
    n = len(d); m = d.mean()
    _, inv = np.unique(blocks, return_inverse=True)
    s = np.bincount(inv, weights=d); c = np.bincount(inv)
    return float(np.sqrt(((s - c * m) ** 2).sum()) / n)


def lowmem(df):
    """As LOWMEM=1 in baseline_models.data(): Float64 -> Float32, in-range Int64 -> Int32 (CatBoost uses float32 internally, so values are unchanged)."""
    import polars as pl
    sch = df.schema
    i64 = [c for c, t in sch.items() if t == pl.Int64]
    ok = []
    if i64:
        mm = df.select([pl.col(c).min().alias(c + '_lo') for c in i64] + [pl.col(c).max().alias(c + '_hi') for c in i64]).row(0, named=True)
        ok = [c for c in i64 if mm[c + '_lo'] is not None and mm[c + '_lo'] > -2 ** 31 and mm[c + '_hi'] < 2 ** 31 - 1]
    return df.with_columns([pl.col(c).cast(pl.Float32) for c, t in sch.items() if t == pl.Float64] + [pl.col(c).cast(pl.Int32) for c in ok])


def read(feat, mo, cols, val, delta_only=False):
    """Read the early-phase feature store by columns and with a filter; row order = file order (same filter as baseline_models.fold())."""
    import polars as pl
    f = (pl.col('src') != 99) & ((pl.col('ym') == 202500 + mo) if val else (pl.col('ym') != 202500 + mo))
    if delta_only: f = f & pl.col('delta').is_not_null()
    return lowmem(pl.scan_parquet(feat).filter(f).select(cols).collect())


def coded_pandas(T, V, cc, cs):
    """Categorical encoding as in the cframe of baseline_models.stage_fullens (training and evaluation share one mapping in order of first appearance,
    nulls become 'NA', int32 codes), but done in polars without pandas object strings; returns pandas (numeric and int32 code columns only)."""
    import polars as pl
    tc, vc = [], []
    for c in cs:
        a = T[c].cast(pl.String).fill_null('NA'); b = V[c].cast(pl.String).fill_null('NA')
        cats = pl.concat([a, b]).unique(maintain_order=True)
        idx = pl.Series(range(len(cats)), dtype=pl.Int32)
        tc.append(a.replace_strict(cats, idx, return_dtype=pl.Int32).alias(c)); vc.append(b.replace_strict(cats, idx, return_dtype=pl.Int32).alias(c))
    T = T.with_columns(tc).select(cc); V = V.with_columns(vc).select(cc)
    return T.to_pandas(), V.to_pandas()


def train(a, bmod, it, itd, cc, cs, mo):
    from catboost import CatBoostRegressor, Pool
    feat = os.path.join(a.work, 'feat.parquet'); parts = os.path.join(a.work, 'parts')
    seeds = [int(x) for x in a.train_seeds.split(',')] if a.train_seeds else list(range(a.seeds))
    mk = lambda n, s: CatBoostRegressor(iterations=n, learning_rate=0.08, depth=8, loss_function='RMSE', thread_count=a.threads,
                                        verbose=0, border_count=128, random_seed=s)
    va = read(feat, mo, list(dict.fromkeys(cc + ['TAXITIME_SEC_mvt', 'delta', 'mvt_aobt3', 'flt_null', 'ADEP_mvt', 'mvt_delay', 'MVT_TIME_UTC_mvt'])), True)
    mv = va['delta'].is_not_null().to_numpy()
    y = va['TAXITIME_SEC_mvt'].to_numpy().astype(float); ma = va['mvt_aobt3'].to_numpy().astype(float); B = bmod.BMASK(va)
    day = va['MVT_TIME_UTC_mvt'].dt.strftime('%Y-%m-%d').to_numpy().astype(str)
    np.savez(os.path.join(a.work, f'va_{mo}.npz'), y=y, mv=mv, ma=ma, B=B, day=day)
    va = va.select(cc)
    log(f'fold {mo}: val {len(y):,} (delta {int(mv.sum()):,}, B {int(B.sum())}) it {it}/{itd} seeds {seeds}')
    todo = lambda part: [s for s in seeds if not os.path.exists(os.path.join(parts, f'cat_{part}_s{s}_{mo}.npy'))]
    for part, n, lab in (('taxi', it, 'TAXITIME_SEC_mvt'), ('delta', itd, 'delta')):
        if not todo(part): continue
        T = read(feat, mo, cc + [lab], False, delta_only=(part == 'delta'))   # each target is read separately; the polars and pandas copies are freed before training
        ytr = T[lab].to_numpy().astype(float)
        C, R = coded_pandas(T.drop(lab), va, cc, cs); del T; gc.collect()
        if part == 'delta': R = R[mv]
        log(f'  fold {mo} {part}: train {len(ytr):,}x{C.shape[1]}')
        P = Pool(C, ytr, cat_features=cs); del C; gc.collect()            # Pool is reused across seeds
        for s in todo(part):
            t0 = time.time(); m = mk(n, s).fit(P); p = m.predict(R); del m; gc.collect()
            save(os.path.join(parts, f'cat_{part}_s{s}_{mo}.npy'), p)
            log(f'  fold {mo} {part} seed {s}: {time.time() - t0:.0f}s' + (f' rmse {rmse(p, y):.2f}' if part == 'taxi' else ''))
        del P, R; gc.collect()


def evaluate(a, res):
    parts = os.path.join(a.work, 'parts'); folds = [int(x) for x in a.folds.split(',')]
    cat_rows = {r['fold']: r['rmse'] for r in res_rows(res, 'cat')}
    ensb_w = {'srp_g5_delta': 0.429, 'cat_delta': 0.286, 'anc_iobt': 0.143, 'apt_delta': 0.143}
    ens_v6 = {r['fold']: r['rmse'] for r in res_rows(res, 'ens') if r['extra'].get('weights') == ensb_w}
    W = 2.0 / 7.0
    out = dict(config=dict(folds=folds, seeds=a.seeds, k3=a.k3, k8=a.k8, w_cat_in_ensb=W, smoke_iters=a.smoke_iters), checks={}, folds={})
    pool = {k: [] for k in ('y', 'nonA', 'day', 'hist', 'cat_hist')}
    catK = {K: [] for K in range(1, a.seeds + 1)}; ensavgK = {K: [] for K in range(1, a.seeds + 1)}
    ok = True
    for mo in folds:
        z = np.load(os.path.join(a.work, f'va_{mo}.npz')); y, mv, ma, B, day = z['y'], z['mv'], z['ma'], z['B'], z['day']
        H = lambda n: np.load(os.path.join(a.hist, f'{n}_{mo}.npy'))
        e5, e6, ch, cth = H('ens_v5b'), H('ens_v6'), H('cat_delta'), H('cat_taxi')
        lens = dict(y=len(y), ens_v5b=len(e5), ens_v6=len(e6), cat_delta=len(ch), cat_taxi=len(cth))
        c_cat = rmse((cth + ch) / 2, y); c_e6 = rmse(e6, y)
        chk = dict(lens=lens, len_ok=len(set(lens.values())) == 1,
                   cat_hist_rmse=round(c_cat, 3), cat_res_tsv=cat_rows.get(mo), cat_ok=abs(c_cat - cat_rows.get(mo, -1e9)) <= 0.01,
                   ens_v6_rmse=round(c_e6, 3), ens_v6_res_tsv=ens_v6.get(mo), ens_v6_ok=abs(c_e6 - ens_v6.get(mo, -1e9)) <= 0.01,
                   ens_v5b_rmse=round(rmse(e5, y), 3))
        out['checks'][str(mo)] = chk
        ok &= chk['len_ok'] and chk['cat_ok'] and chk['ens_v6_ok']
        PT = np.stack([np.load(os.path.join(parts, f'cat_taxi_s{s}_{mo}.npy')) for s in range(a.seeds)])
        PD = np.stack([np.load(os.path.join(parts, f'cat_delta_s{s}_{mo}.npy')) for s in range(a.seeds)])
        assert PT.shape[1] == len(y) and PD.shape[1] == int(mv.sum()), (PT.shape, PD.shape, len(y), int(mv.sum()))
        nonA = ~((~B) & (y > 10800)); hist = (e5 + e6) / 2
        fo = dict(n=int(len(y)), B=int(B.sum()), nonA=int(nonA.sum()), ensavg_hist_rmse=round(rmse(hist, y), 4), cat_hist_rmse=round(rmse(ch, y), 4), K={})
        for K in range(1, a.seeds + 1):
            p = PT[:K].mean(0); p[mv] = ma[mv] + PD[:K].mean(0); p[B] = ch[B]
            v = 0.5 * e5 + 0.5 * (e6 + W * (p - ch))
            catK[K].append(p); ensavgK[K].append(v)
            fo['K'][str(K)] = dict(cat_rmse=round(rmse(p, y), 4), ensavg_rmse=round(rmse(v, y), 4), ensavg_nonA_rmse=round(rmse(v[nonA], y[nonA]), 4))
        p0 = PT[0].copy(); p0[mv] = ma[mv] + PD[0]; p0[B] = ch[B]
        fo['seed0_vs_hist'] = dict(rms=round(rmse(p0, ch), 3), mean=round(float((p0 - ch).mean()), 3), corr=round(float(np.corrcoef(p0, ch)[0, 1]), 6))
        fo['seed_sd_taxi'] = round(float(np.sqrt(np.mean(np.var(PT, axis=0)))), 3)
        fo['seed_sd_delta'] = round(float(np.sqrt(np.mean(np.var(PD, axis=0)))), 3)
        out['folds'][str(mo)] = fo
        for k, v in (('y', y), ('nonA', nonA), ('day', day), ('hist', hist), ('cat_hist', ch)): pool[k].append(v)
    y = np.concatenate(pool['y']); nonA = np.concatenate(pool['nonA']); day = np.concatenate(pool['day']); hist = np.concatenate(pool['hist'])
    allK = {}
    for K in range(1, a.seeds + 1):
        v = np.concatenate(ensavgK[K]); c = np.concatenate(catK[K])
        allK[str(K)] = dict(ensavg_rmse=rmse(v, y), ensavg_nonA_rmse=rmse(v[nonA], y[nonA]), cat_rmse=rmse(c, y))
    out['pooled'] = dict(n=int(len(y)), ensavg_hist_rmse=rmse(hist, y), K=allK)
    k3, k8 = a.k3, a.k8
    ensavg_k3 = np.concatenate(ensavgK[k3]); ensavg_k8 = np.concatenate(ensavgK[k8])
    d = (ensavg_k8 - y) ** 2 - (ensavg_k3 - y) ** 2; dn = d[nonA]
    r3, r8 = rmse(ensavg_k3, y), rmse(ensavg_k8, y)
    se = block_se(d, day)
    gate = dict(rule='delta = RMSE(baseline with K8) - RMSE(baseline with K3), all rows of both folds pooled; delta <= 0 passes the gate (fixed beforehand)',
                rmse_k3=r3, rmse_k8=r8, delta_rmse=r8 - r3, delta_mse=float(d.mean()), delta_mse_day_se=se,
                delta_mse_z=float(d.mean() / se) if se > 0 else None,
                nonA=dict(rmse_k3=rmse(ensavg_k3[nonA], y[nonA]), rmse_k8=rmse(ensavg_k8[nonA], y[nonA]), delta_mse=float(dn.mean()), day_se=block_se(dn, day[nonA])),
                by_fold={str(mo): dict(rmse_k3=out['folds'][str(mo)]['K'][str(k3)]['ensavg_rmse'], rmse_k8=out['folds'][str(mo)]['K'][str(k8)]['ensavg_rmse']) for mo in folds},
                checks_ok=bool(ok))
    gate['PASS'] = bool(ok and gate['delta_rmse'] <= 0)
    out['gate'] = gate
    fp = os.path.join(a.work, 'catseeds_cv.json')
    json.dump(out, open(fp + '.tmp', 'w'), ensure_ascii=False, indent=1); os.replace(fp + '.tmp', fp)
    log('checks', json.dumps(out['checks'], ensure_ascii=False))
    log('gate', json.dumps(gate, ensure_ascii=False))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', required=True); ap.add_argument('--work', required=True); ap.add_argument('--hist', required=True)
    ap.add_argument('--folds', default='1,7'); ap.add_argument('--seeds', type=int, default=8); ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--k3', type=int, default=3); ap.add_argument('--k8', type=int, default=8)
    ap.add_argument('--smoke-iters', type=int, default=0); ap.add_argument('--eval-only', action='store_true'); ap.add_argument('--train-only', action='store_true'); ap.add_argument('--train-seeds', default='')
    a = ap.parse_args()
    os.environ['LOWMEM'] = '1'
    sys.path.insert(0, a.code)
    import baseline_models as bmod
    bmod.configure(a.work, a.work, threads=a.threads, tag='catseedscv')
    mm = bmod.exp_meta('cat'); cc = bmod.FEATSETS[mm['fs']]; cs = [c for c in cc if c in bmod.CAT]
    it, itd = mm['it'], mm['itd']
    log('cat meta', mm, 'cols', len(cc), 'cat cols', cs)
    if a.smoke_iters: it = itd = a.smoke_iters
    else: assert (mm['fs'], it, itd) == ('sr', 2846, 3436), mm
    os.makedirs(os.path.join(a.work, 'parts'), exist_ok=True)
    if not a.eval_only:
        for mo in [int(x) for x in a.folds.split(',')]:
            train(a, bmod, it, itd, cc, cs, mo)
    if a.train_only: log('CAT8_OOF_TRAIN_ONLY_DONE', a.folds); return
    ok = evaluate(a, bmod.RES)
    if not ok: raise SystemExit('alignment self-check failed (see checks in catseeds_cv.json)')
    log('CAT8_OOF_DONE')


if __name__ == '__main__':
    main()
