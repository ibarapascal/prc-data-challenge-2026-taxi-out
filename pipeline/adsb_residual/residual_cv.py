"""Evaluation of the winter-pool and generic-model questions for the residual layer: three gap experiments under the
reference residual configuration (feats_tm + STAND=fix + CELLFB=pred + art + hyb:01 + threshold .15; "reference" and
the labels refcfg below refer to it; in winter_preds.npz its array keeps the key v20). The code directory is the directory of this file (the train_resid in the same
directory is imported); the configuration is taken entirely from environment variables.

Subcommands (run from the working directory, outputs go to resid_cv/):
  winter --out resid_cv/winter.json
      Winter hold-out CV plus proxy 1 for the generic-model risk. Test set = all labelled DEP rows of all airports in
      2025-11/12 (isA excluded; baseline = oof12 linearly calibrated per airport to the averaged-baseline scale),
      date-grouped 5 folds (seed 0) plus a forward fold (test December, pool only from November). Each scheme replaces
      only the LEMD / LIRF rows, all other rows follow the reference assignment:
        v20   = (array key in winter_preds.npz) the reference assignment for 2026-01: the 6 airports whose January gating rate is >= the threshold use the
                January-only full model (J), the others use the 62-day all-day model (A, "off" assignment)
        W1112 = full model trained on January training rows + pool 11/12 (days outside the test fold); W061112 adds pool 06;
                Wp1112 = pool 11/12 only
        B     = all-day "off" assignment trained on the 62 days + pools 06/08/11/12 (outside the test fold) (a "whole year in training" control)
        G     = leave-one-airport-out generic model (trained on the gated rows of the 62 days of the other airports) -> proxy 1
                for the generic-model risk: LEMD has only a sparse, extrapolated July pattern in training and a dense
                pattern in test, like EGLL-07 in 2026; compares A (full model trained on the airport's own sparse samples) with G
        W1112_8apt = the January rows of 8 airports (the 6 airports + LEMD/LIRF) all replaced by W1112 (control: does the winter pool also help the 6 airports?)
      The test-side grid-cell encoding matches the reference prediction path (PRED_FB=True when CELLFB=pred; training unaffected).
      Reports: day-paired +-SE against the reference assignment (denominator = baseline SSE of all nonA rows of the test set,
      in percentage points), the 5 folds one by one, RMSE and f per airport / month, and a 2026 conversion (n26 x per-row MSE difference, estimate).
  winter_deconf --out resid_cv/winter_deconf.json
      De-confounding of the winter CV: R = ensavg (the 62-day training rows keep the averaged baseline, original set-up) | o12
      (training rows also use the oof12-calibrated baseline, consistent with the test rows) x W06 (January + pool 06, not
      adjacent to the test months) | W1112 (January + pool 11/12 outside the test fold); the 6 January-full-model airports serve as negative controls.
  winter_fwd --out resid_cv/winter_fwd.json
      Forward check: pool = 2025-11 (+06), test = all airports in 2025-12 (simulates a one-month neighbour, 2025-12 -> 2026-01);
      W11 / W0611 / Wp11 replace the December rows of LEMD, LEMD+LIRF, the 6 airports and the 8 airports respectively, compared
      with the reference assignment; the 2026 conversion uses the January gated-row counts of each airport.
      --test-base srp_g5_avg: use another oof12 member as the test-row baseline (the pool stays delta), to check whether the
      gain comes from "pool and test share the same baseline".
  genloao --out resid_cv/genloao.json
      Proxy 2 for the generic-model risk: regular 62-day CV (same folds), for each of the 6 airports with 2025 coverage a
      leave-one-airport-out generic model (G) replaces that airport's gated rows, compared with the CV OOF of the reference
      configuration (--oof, default cvtg_tm.oof.npy); for EGLL also Gd (trained only on "dense-pattern" rows of the other
      airports: etype 1 and >= 20 points). Reported stratified by airport x etype x number of points (EGLL 2026-07 is almost all
      etype 1 with a median of 93 points; EDDF 2026 is sparser).
  standcv --mode single|cluster --seed S --out resid_cv/standcv_<mode>.json
      Stand-masking CV: same 62-day folds; in each fold stands are drawn at random among the gated rows of the test fold
      (single = one stand; cluster = a seed stand together with all stands within 120 m, simulating a whole new pier that was
      never seen) until the --rate (default .10) of each airport's gated rows is reached; the training rows of these stands are
      removed entirely; the same boosters (T.dual) compare STANDFB=off with pred (needs RESID_STANDFB=pred, RESID_CELLFB=off to
      isolate the stand fallback effect). Also reports the number of 2026 rows with unseen stands (against the encoding pool of the
      reference prediction path) and the distance to the nearest stand in the pool.
  covdrop --out resid_cv/covdrop.json
      Risk ablation: the coverage-dependent columns (drop3 = ab_taxi_now10/ab_npts/ab_max_gap; dropcov adds
      ab_etype/ab_last_ground/ab_dwell_seen) are set to null, (1) on the winter proxy (all rows of 2025-11/12, reference
      assignment; LEMD is sparse in training -> dense in test) and (2) on the regular 62-day CV against --oof.
  cv --out resid_cv/cv_<name>.json [--oof cvtg_tm.oof.npy] [--seeds N (LightGBM seed averaging, damps retraining noise)] [--test-feats DIR (swap only the test-side features, paired on the same boosters)]
      62-day CV (same folds) under the current environment configuration, day-paired against --oof; if the features have ab_anchor, the trimmed rows are reported separately
  pair --a X.oof.npy --oof REF.oof.npy --out resid_cv/pair_<name>.json   day-paired comparison of two 62-day CV OOF files (no training)
  diff --a X.parquet --b Y.parquet --out X.json   row-by-row difference (reuses dec_eval.cmd_diff; the dec_eval module is not part of this repository, so only this subcommand is affected)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_resid as T  # noqa: E402

FE = os.environ.get('RESID_FEATS', 'feats_tm')
N26 = 344841


def day_folds(dates, nfold=5, seed=0):
    days = sorted(set(dates))
    perm = np.random.default_rng(seed).permutation(len(days))
    fold_of = {days[i]: k % nfold for k, i in enumerate(perm)}
    return np.array([fold_of[d] for d in dates])


def paired(dates, d, mask, denom):
    """d = per-row (e_new - e_ref); summed and paired by day. Returns the improvement in percentage points (positive = better) and its SE."""
    s = pl.DataFrame({'d': dates, 'x': np.where(mask, d, 0.0)}).group_by('d').agg(pl.col('x').sum())['x'].to_numpy()
    return dict(pp=round(float(-100 * s.sum() / denom), 4), se=round(float(100 * s.std(ddof=1) * np.sqrt(len(s)) / denom), 4))


def per_fold(d, mask, fd, denom, nfold=5):
    per = np.array([-d[mask & (fd == j)].sum() / denom for j in range(nfold)])
    return dict(per_fold_pp=[round(100 * x, 4) for x in per], folds_positive=int((per > 0).sum()))



def rank_gate_counts(feats):
    rk = T.load_rank('n2', feats)
    m = rk.filter(pl.col('gate')).with_columns(pl.col('date').str.slice(5, 2).alias('mo'))
    return {f'{a_}-{mo}': n for a_, mo, n in m.group_by('ADEP_mvt', 'mo').len().iter_rows()}, rk


# ---------------------------------------------------------------- winter (winter hold-out CV + generic-model proxy 1)
def cmd_winter(a):
    T.PRED_FB = T.CELLFB == 'pred'
    v = T.load_train('n2', FE)
    trv = v.filter(~pl.col('isA'))
    pool, cal = T.load_pool('n2', v, ['06', '08', '11', '12'])
    pm = pool['date'].str.slice(5, 2).to_numpy()
    te = pool.filter(pl.Series(np.isin(pm, ['11', '12'])))
    pool06, pool08 = pool.filter(pl.Series(pm == '06')), pool.filter(pl.Series(pm == '08'))
    y, b, g = te['TAXITIME_SEC_mvt'].to_numpy().astype(float), te['base'].to_numpy(), te['gate'].to_numpy()
    apt, tmo, dates = te['ADEP_mvt'].to_numpy(), te['date'].str.slice(5, 2).to_numpy(), te['date'].to_list()
    fd = day_folds(dates)
    n = te.height
    mo_tr = trv['date'].str.slice(5, 2).to_numpy()
    jan = trv.filter(pl.Series(mo_tr == '01'))
    fa_jan = [x for x in T.APTS if float(jan.filter(pl.col('ADEP_mvt') == x)['gate'].mean() or 0) >= T.FULL_THR]
    WA = ['LEMD', 'LIRF']
    print('test rows', n, 'days', len(set(dates)), 'fa_jan', fa_jan, flush=True)
    P = {}
    rA, infoA = T.fit_mixed(trv, te, mode='off')
    rJ, _ = T.fit_predict(jan, te)
    P['A'] = rA
    P['v20'] = np.where(np.isin(apt, fa_jan), rJ, rA)
    rG = P['v20'].copy()
    for x in WA:
        m = apt == x
        rG[m] = T.fit_predict_generic(trv.filter(pl.col('gate') & (pl.col('ADEP_mvt') != x)), te.filter(pl.Series(m)))
    P['G'] = rG
    names = ['W1112', 'W061112', 'Wp1112', 'B']
    for nm in names:
        P[nm] = np.full(n, np.nan)
        P[nm + '_fwd'] = np.full(n, np.nan)
    for k in list(range(5)) + ['fwd']:
        if k == 'fwd':
            tm, trp = tmo == '12', te.filter(pl.Series(tmo == '11'))
        else:
            tm, trp = fd == k, te.filter(pl.Series(fd != k))
        tes = te.filter(pl.Series(tm))
        sfx = '_fwd' if k == 'fwd' else ''
        P['W1112' + sfx][tm], _ = T.fit_predict(pl.concat([jan, trp], how='vertical_relaxed'), tes)
        P['W061112' + sfx][tm], _ = T.fit_predict(pl.concat([jan, trp, pool06], how='vertical_relaxed'), tes)
        P['Wp1112' + sfx][tm], _ = T.fit_predict(trp, tes)
        P['B' + sfx][tm], _ = T.fit_mixed(pl.concat([trv, trp, pool06, pool08], how='vertical_relaxed'), tes, mode='off')
        print('fold', k, 'done', int(tm.sum()), flush=True)
    wa = np.isin(apt, WA)
    C = {}  # candidates = reference assignment, replacing only the LEMD/LIRF rows
    for nm in ['A', 'G'] + names:
        C[nm] = np.where(wa, P[nm], P['v20'])
    C['W1112_8apt'] = np.where(np.isin(apt, WA + fa_jan), P['W1112'], P['v20'])
    C['blendAG'] = np.where(wa, 0.5 * (P['A'] + P['G']), P['v20'])
    Cf = {nm: np.where(wa, P[nm + '_fwd'], P['v20']) for nm in names}
    e0 = (b - y) ** 2
    E = {nm: (b + r * g - y) ** 2 for nm, r in C.items()}
    Ef = {nm: (b + r * g - y) ** 2 for nm, r in Cf.items()}
    err2_refcfg = (b + P['v20'] * g - y) ** 2
    denom = float(e0.sum())
    subsets = {'all': np.ones(n, bool), 'LEMD+LIRF': wa, 'others': ~wa, 'm11': tmo == '11', 'm12': tmo == '12'}
    for x in T.APTS:
        subsets[x] = apt == x
    n26, _ = rank_gate_counts(FE)
    out = dict(test_rows=n, test_days=len(set(dates)), base_nonA_sse=denom, fa_jan=fa_jan, pool_cal=cal,
               pool_shape={f'{x}-{mm}': dict(gate=round(float(pool.filter((pl.col('ADEP_mvt') == x) & (pl.col('date').str.slice(5, 2) == mm))['gate'].mean() or 0), 3),
                                              etype2=round(float(pool.filter((pl.col('ADEP_mvt') == x) & (pl.col('date').str.slice(5, 2) == mm) & pl.col('gate'))['ab_etype'].eq(2).mean() or 0), 3),
                                              npts_med=pool.filter((pl.col('ADEP_mvt') == x) & (pl.col('date').str.slice(5, 2) == mm) & pl.col('gate'))['ab_npts'].median())
                           for x in ['LEMD', 'LIRF', 'EGLL', 'EDDF'] for mm in ['06', '08', '11', '12']},
               n26_gated=n26, info_A=infoA, var={})
    rm = lambda e, m: round(float(np.sqrt(e[m].mean())), 1) if m.any() else None  # noqa: E731
    for nm in E:
        d = E[nm] - err2_refcfg
        o = dict(vs_refcfg={s: paired(dates, d, m, denom) for s, m in subsets.items() if m.any()},
                 folds=per_fold(d, np.ones(n, bool), fd, denom))
        o['rmse_gated'] = {x: dict(base=rm(e0, (apt == x) & g), refcfg=rm(err2_refcfg, (apt == x) & g), new=rm(E[nm], (apt == x) & g),
                                   n=int(((apt == x) & g).sum())) for x in T.APTS if ((apt == x) & g).any()}
        # 2026 conversion: January gated-row count per airport x per-row MSE difference (only airports that were replaced)
        rep = WA + (fa_jan if nm == 'W1112_8apt' else [])
        dm = sum(n26.get(f'{x}-01', 0) * (err2_refcfg[(apt == x) & g].mean() - E[nm][(apt == x) & g].mean()) for x in rep
                 if ((apt == x) & g).any()) / N26
        o['est26'] = dict(dmse_per_row=round(float(dm), 3))
        out['var'][nm] = o
        print(nm, o['vs_refcfg']['all'], o['vs_refcfg'].get('LEMD'), o['vs_refcfg'].get('LIRF'), o['folds'], o['est26'], flush=True)
    dec = tmo == '12'
    denom_dec = float(e0[dec].sum())
    out['fwd'] = {nm: dict(vs_refcfg=paired(dates, Ef[nm] - err2_refcfg, dec, denom_dec),
                           LEMD=paired(dates, Ef[nm] - err2_refcfg, dec & (apt == 'LEMD'), denom_dec),
                           LIRF=paired(dates, Ef[nm] - err2_refcfg, dec & (apt == 'LIRF'), denom_dec),
                           rmse_gated={x: dict(refcfg=rm(err2_refcfg, dec & (apt == x) & g), new=rm(Ef[nm], dec & (apt == x) & g)) for x in WA})
                  for nm in Ef}
    print('fwd', json.dumps(out['fwd']), flush=True)
    # Generic-model proxy 1: on LEMD rows, A (full model trained on the airport's own sparse samples) vs G (leave-one-airport-out generic) vs W
    lm = (apt == 'LEMD') & g
    out['p03_proxy1_LEMD'] = {nm: dict(rmse=rm(E[nm], lm), f=round(float(E[nm][lm].sum() / e0[lm].sum()), 4)) for nm in ['A', 'G', 'blendAG', 'W1112']}
    out['p03_proxy1_LEMD']['base_rmse'] = rm(e0, lm)
    np.savez_compressed('resid_cv/winter_preds.npz', **{k: v_ for k, v_ in P.items()}, y=y, b=b, g=g)
    json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- genloao (generic-model proxy 2)
def cmd_genloao(a):
    v = T.load_train('n2', FE)
    oof = np.load(a.oof)
    assert len(oof) == v.height, (len(oof), v.height)
    y, b, g = v['TAXITIME_SEC_mvt'].to_numpy().astype(float), v['base'].to_numpy(), v['gate'].to_numpy()
    nA, apt, dates = ~v['isA'].to_numpy(), v['ADEP_mvt'].to_numpy(), v['date'].to_list()
    et, npts = v['ab_etype'].to_numpy(), v['ab_npts'].to_numpy()
    mo = v['date'].str.slice(5, 2).to_numpy()
    fd = day_folds(dates)
    AP = ['EGLL', 'EDDF', 'EDDM', 'EHAM', 'LEBL', 'LSZH']
    G, Gd = oof.copy(), oof.copy()
    for k in range(5):
        trk = v.filter(pl.Series((fd != k) & nA & g))
        for x in AP:
            m = (fd == k) & (apt == x) & g
            G[m] = T.fit_predict_generic(trk.filter(pl.col('ADEP_mvt') != x), v.filter(pl.Series(m)))
        m = (fd == k) & (apt == 'EGLL') & g
        dense = trk.filter((pl.col('ADEP_mvt') != 'EGLL') & (pl.col('ab_etype') == 1) & (pl.col('ab_npts') >= 20))
        Gd[m] = T.fit_predict_generic(dense, v.filter(pl.Series(m)))
        print('fold', k, 'done', flush=True)
    e0, ev = (b - y) ** 2, (b + oof - y) ** 2
    E = {'G': (b + G - y) ** 2, 'Gd': (b + Gd - y) ** 2, 'blend': (b + 0.5 * (G + oof) - y) ** 2}
    denom = float(e0[nA].sum())
    rm = lambda e, m: round(float(np.sqrt(e[m].mean())), 1) if m.any() else None  # noqa: E731
    out = dict(oof=a.oof, base_nonA_sse=denom, strata={})
    for x in AP:
        for sname, sm in (('all', np.ones(v.height, bool)), ('et1', et == 1), ('et2', et == 2), ('npts<10', npts < 10),
                          ('npts10-30', (npts >= 10) & (npts < 30)), ('npts>=30', npts >= 30), ('et1_npts>=30', (et == 1) & (npts >= 30)),
                          ('jan', mo == '01'), ('jul', mo == '07')):
            m = (apt == x) & g & nA & sm
            if m.sum() < 50:
                continue
            o = dict(n=int(m.sum()), rmse_base=rm(e0, m), rmse_refcfg=rm(ev, m))
            for nm, e in E.items():
                if nm == 'Gd' and x != 'EGLL':
                    continue
                o[nm] = dict(rmse=rm(e, m), vs_refcfg=paired(dates, e - ev, m, denom), **per_fold(e - ev, m, fd, denom))
            out['strata'][f'{x}|{sname}'] = o
            print(x, sname, json.dumps(o), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- standcv (stand-masking CV)
def mask_stands(v, fd, k, rate, mode, seed, has_ll):
    rng = np.random.default_rng(1000 * seed + k)
    gg = v['gate'].to_numpy()
    t = v.filter(pl.Series((fd == k) & has_ll & gg))
    M = set()
    for x in ['EDDF', 'EDDM', 'EHAM', 'EGLL', 'LEBL', 'LSZH', 'LEMD']:
        tx = t.filter(pl.col('ADEP_mvt') == x)
        tgt = rate * int(((fd == k) & gg & (v['ADEP_mvt'].to_numpy() == x)).sum())
        if tx.height == 0 or tgt <= 0:
            continue
        cnt = tx.group_by('stand_key').len().sort('stand_key')
        keys, ns = cnt['stand_key'].to_list(), cnt['len'].to_numpy()
        ll = np.array([T.stand_ll(kk) for kk in keys], dtype=float)
        cur = 0
        for i in rng.permutation(len(keys)):
            if cur >= tgt:
                break
            if keys[i] in M:
                continue
            grp = [i] if mode == 'single' else list(np.where(T._hav(ll[i, 0], ll[i, 1], ll[:, 0], ll[:, 1]) <= 120)[0])
            for j in grp:
                if keys[j] not in M:
                    M.add(keys[j])
                    cur += int(ns[j])
    return M


def cmd_standcv(a):
    assert T.STANDFB == 'pred' and T.CELLFB == 'off', 'requires RESID_STANDFB=pred and RESID_CELLFB=off'
    v = T.load_train('n2', FE)
    y, b, g = v['TAXITIME_SEC_mvt'].to_numpy().astype(float), v['base'].to_numpy(), v['gate'].to_numpy()
    nA, apt, dates, isA = ~v['isA'].to_numpy(), v['ADEP_mvt'].to_numpy(), v['date'].to_list(), v['isA'].to_numpy()
    fd = day_folds(dates)
    sk = v['stand_key'].to_list()
    llc = {kk: T.stand_ll(kk) for kk in set(sk)}
    has_ll = np.array([llc[kk] is not None for kk in sk])
    r0, r1, mask = np.zeros(v.height), np.zeros(v.height), np.zeros(v.height, bool)
    info = []
    for k in range(5):
        M = mask_stands(v, fd, k, a.rate, a.mode, a.seed, has_ll)
        inM = v['stand_key'].is_in(list(M)).fill_null(False).to_numpy()
        tr_m = (fd != k) & ~isA & ~inM
        m = fd == k
        mask[m & inM & g] = True
        T.STAND_LOG.clear()
        (o0, _, _), (o1, _, _) = T.dual(T.fit_all, v.filter(pl.Series(tr_m)), v.filter(pl.Series(m)))
        r0[m], r1[m] = o0, o1
        ch = m & (r0 != r1)
        info.append(dict(fold=k, n_masked_keys=len(M), n_masked_test=int((m & inM & g).sum()), n_changed=int(ch.sum()),
                         n_changed_outside_mask=int((ch & ~inM).sum()), stand_log=list(T.STAND_LOG)))
        print('fold', {x: y_ for x, y_ in info[-1].items() if x != 'stand_log'}, flush=True)
    e0, ea, eb = (b - y) ** 2, (b + r0 * g - y) ** 2, (b + r1 * g - y) ** 2
    denom = float(e0[nA].sum())
    d = eb - ea
    rm = lambda e, m: round(float(np.sqrt(e[m].mean())), 1) if m.any() else None  # noqa: E731
    out = dict(mode=a.mode, seed=a.seed, rate=a.rate, r_m=T.STANDFB_R, base_nonA_sse=denom, folds=info,
               all=dict(**paired(dates, d, nA, denom), **per_fold(d, nA, fd, denom)),
               masked=dict(**paired(dates, d, nA & mask, denom), n=int((nA & mask).sum()),
                           rmse_off=rm(ea, nA & mask), rmse_fb=rm(eb, nA & mask)),
               apt={x: dict(n=int((nA & mask & (apt == x)).sum()), **paired(dates, d, nA & mask & (apt == x), denom),
                            rmse_off=rm(ea, nA & mask & (apt == x)), rmse_fb=rm(eb, nA & mask & (apt == x)),
                            dmse_per_masked_row=round(float((ea - eb)[nA & mask & (apt == x)].mean()), 1) if (nA & mask & (apt == x)).any() else None)
                    for x in T.APTS})
    # 2026: gated rows with unseen stands on the reference prediction path (the 6 January airports look up the January pool, the others the whole pool) + distance to the nearest stand in the pool
    _, rk = rank_gate_counts(FE)
    trv = v.filter(~pl.col('isA'))
    ok = ~np.isnan(T.te_targets(trv)['gap_push'])
    ktr, mtr = trv['stand_key'].to_numpy(), trv['date'].str.slice(5, 2).to_numpy()
    pa, pj = set(ktr[ok]) - {None}, set(ktr[ok & (mtr == '01')]) - {None}
    fa_jan = [x for x in T.APTS if float(trv.filter(pl.col('date').str.slice(5, 2) == '01').filter(pl.col('ADEP_mvt') == x)['gate'].mean() or 0) >= T.FULL_THR]
    rg, ra, rkk, rmo = rk['gate'].to_numpy(), rk['ADEP_mvt'].to_numpy(), rk['stand_key'].to_list(), rk['date'].str.slice(5, 2).to_numpy()
    unseen = np.array([kk is not None and kk not in (pj if (mm == '01' and x in fa_jan) else pa) for kk, x, mm in zip(rkk, ra, rmo)]) & rg
    llp = {x: np.array([T.stand_ll(kk) for kk in pa if kk.startswith(x + '|') and T.stand_ll(kk)], dtype=float) for x in T.APTS}
    u26 = {}
    for x in T.APTS:
        for mm in ('01', '07'):
            m = unseen & (ra == x) & (rmo == mm)
            if not m.any():
                continue
            ks = sorted(set(kk for kk, t in zip(rkk, m) if t))
            dm = []
            for kk in ks:
                ll = T.stand_ll(kk)
                if ll and len(llp[x]):
                    dm.append(float(T._hav(ll[0], ll[1], llp[x][:, 0], llp[x][:, 1]).min()))
            u26[f'{x}-{mm}'] = dict(rows=int(m.sum()), keys=len(ks), keys_with_ll=len(dm),
                                    dmin_q=[round(q, 1) for q in np.quantile(dm, [.1, .5, .9])] if dm else None,
                                    keys_within_r=int(sum(q <= T.STANDFB_R for q in dm)), top_keys=ks[:15])
    out['unseen_2026'] = u26
    dmse = sum(u26[k_]['rows'] * (out['apt'][k_[:4]]['dmse_per_masked_row'] or 0) for k_ in u26) / N26
    out['est26'] = dict(dmse_per_row=round(dmse, 3),
                        note='Assumption: the per-row MSE reduction on 2026 unseen-stand rows equals that on masked-stand rows (and all of them can fall back)')
    print(json.dumps({k_: out[k_] for k_ in ('all', 'masked', 'apt', 'unseen_2026', 'est26')}, indent=1, ensure_ascii=False), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- covdrop (risk ablation)
COV_SETS = {'drop3': ['ab_taxi_now10', 'ab_npts', 'ab_max_gap'],
            'dropcov': ['ab_taxi_now10', 'ab_npts', 'ab_max_gap', 'ab_etype', 'ab_last_ground', 'ab_dwell_seen']}


def _null(df, cols):
    return df.with_columns([pl.lit(None, df.schema[c]).alias(c) for c in cols])


def cmd_covdrop(a):
    """Two ablations with the coverage-dependent columns set to null: (1) the winter proxy (LEMD sparse in training -> dense in test, LIRF the other way round; reference assignment); (2) the regular 62-day CV (against --oof)."""
    T.PRED_FB = T.CELLFB == 'pred'
    v = T.load_train('n2', FE)
    pool, _ = T.load_pool('n2', v, ['11', '12'])
    y, b, g = pool['TAXITIME_SEC_mvt'].to_numpy().astype(float), pool['base'].to_numpy(), pool['gate'].to_numpy()
    apt, dates = pool['ADEP_mvt'].to_numpy(), pool['date'].to_list()
    fdw = day_folds(dates)
    trv = v.filter(~pl.col('isA'))
    mo_tr = trv['date'].str.slice(5, 2).to_numpy()
    fa_jan = [x for x in T.APTS if float(trv.filter(pl.Series(mo_tr == '01')).filter(pl.col('ADEP_mvt') == x)['gate'].mean() or 0) >= T.FULL_THR]
    e0 = (b - y) ** 2
    denom_w = float(e0.sum())
    Rw = {}
    for nm, cols in [('none', [])] + list(COV_SETS.items()):
        tr_, te_ = (_null(trv, cols), _null(pool, cols)) if cols else (trv, pool)
        rA, _ = T.fit_mixed(tr_, te_, mode='off')
        rJ, _ = T.fit_predict(tr_.filter(pl.Series(mo_tr == '01')), te_)
        Rw[nm] = np.where(np.isin(apt, fa_jan), rJ, rA)
        print('winter proxy', nm, 'done', flush=True)
    rm = lambda e, m: round(float(np.sqrt(e[m].mean())), 1) if m.any() else None  # noqa: E731
    Ew = {k: (b + r * g - y) ** 2 for k, r in Rw.items()}
    out = dict(winter_proxy={}, cv={})
    for nm in COV_SETS:
        d = Ew[nm] - Ew['none']
        out['winter_proxy'][nm] = dict(all=paired(dates, d, np.ones(len(y), bool), denom_w), **per_fold(d, np.ones(len(y), bool), fdw, denom_w),
                                       apt={x: dict(**paired(dates, d, apt == x, denom_w), rmse_none=rm(Ew['none'], (apt == x) & g),
                                                    rmse=rm(Ew[nm], (apt == x) & g)) for x in T.APTS if (apt == x).any()})
        print('winter', nm, json.dumps(out['winter_proxy'][nm]), flush=True)
    # Regular 62-day CV (same as the reference cvtg_tm: no grid-cell fallback on the test side)
    T.PRED_FB = False
    yv, bv, gv = v['TAXITIME_SEC_mvt'].to_numpy().astype(float), v['base'].to_numpy(), v['gate'].to_numpy()
    nA, av, dv, isA = ~v['isA'].to_numpy(), v['ADEP_mvt'].to_numpy(), v['date'].to_list(), v['isA'].to_numpy()
    mv = v['date'].str.slice(5, 2).to_numpy()
    fd = day_folds(dv)
    ref = np.load(a.oof)
    eref = (bv + ref - yv) ** 2
    denom = float(((bv - yv) ** 2)[nA].sum())
    for nm, cols in COV_SETS.items():
        vn = _null(v, cols)
        res = np.zeros(v.height)
        for k in range(5):
            r, _, _ = T.fit_all(vn.filter(pl.Series((fd != k) & ~isA)), vn.filter(pl.Series(fd == k)))
            res[fd == k] = np.where(gv[fd == k], r, 0.0)
        np.save(f'resid_cv/cv_{nm}.oof.npy', res)
        d = (bv + res - yv) ** 2 - eref
        out['cv'][nm] = dict(all=paired(dv, d, nA, denom), **per_fold(d, nA, fd, denom),
                             jan=paired(dv, d, nA & (mv == '01'), denom), jul=paired(dv, d, nA & (mv == '07'), denom),
                             apt={x: paired(dv, d, nA & (av == x), denom) for x in T.APTS})
        print('cv', nm, json.dumps(out['cv'][nm]), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- cv (generic: 62-day CV under the current environment configuration, paired against --oof)
def cmd_cv(a):
    v = T.load_train('n2', FE)
    y, b, g = v['TAXITIME_SEC_mvt'].to_numpy().astype(float), v['base'].to_numpy(), v['gate'].to_numpy()
    nA, av, dv, isA = ~v['isA'].to_numpy(), v['ADEP_mvt'].to_numpy(), v['date'].to_list(), v['isA'].to_numpy()
    mv = v['date'].str.slice(5, 2).to_numpy()
    fd = day_folds(dv)
    res = np.zeros(v.height)
    VT, RT = [None], [np.zeros(v.height)]
    if a.test_feats:  # swap only the test-side features (training identical to the reference, same boosters), so differences fall only on rows where the two feature sets differ
        sd, T.STAND_DIR = T.STAND_DIR, a.test_feats
        VT[0] = T.load_train('n2', a.test_feats)
        T.STAND_DIR = sd
        assert VT[0]['MVT_ID_mvt'].equals(v['MVT_ID_mvt'])
        T._BC.update(mode='store', models=[], i=0)
    for k in range(5):
        r, _, _ = T.fit_all(v.filter(pl.Series((fd != k) & ~isA)), v.filter(pl.Series(fd == k)), seeds=tuple(range(a.seeds)))
        res[fd == k] = np.where(g[fd == k], r, 0.0)
        if a.test_feats:
            T._BC.update(mode='reuse', i=0)
            r2, _, _ = T.fit_all(v.filter(pl.Series((fd != k) & ~isA)), VT[0].filter(pl.Series(fd == k)), seeds=tuple(range(a.seeds)))
            assert T._BC['i'] == len(T._BC['models'])
            RT[0][fd == k] = np.where(VT[0]['gate'].to_numpy()[fd == k], r2, 0.0)
            T._BC.update(mode='store', models=[], i=0)
        print('fold', k, flush=True)
    np.save(a.out.replace('.json', '.oof.npy'), res)
    if a.test_feats:  # pairing target = the training-side feature version of the same boosters (res); the new scheme = test-side features swapped (RT)
        ref, res, v = res, RT[0], VT[0]
        T._BC.update(mode=None, models=[], i=0)
        np.save(a.out.replace('.json', '.testonly.oof.npy'), res)
    else:
        ref = np.load(a.oof)
    assert len(ref) == v.height
    d = (b + res - y) ** 2 - (b + ref - y) ** 2
    denom = float(((b - y) ** 2)[nA].sum())
    out = dict(feats=FE, test_feats=a.test_feats or None, ref=a.oof if not a.test_feats else 'same boosters, train-side feats', seeds=a.seeds, n=v.height, gate_cover=round(float(g.mean()), 4),
               all=dict(**paired(dv, d, nA, denom), **per_fold(d, nA, fd, denom)),
               jan=paired(dv, d, nA & (mv == '01'), denom), jul=paired(dv, d, nA & (mv == '07'), denom),
               apt={x: paired(dv, d, nA & (av == x), denom) for x in T.APTS})
    if 'ab_anchor' in v.columns:
        an = v['ab_anchor'].fill_null(0).to_numpy() == 1
        out['anchored'] = dict(rows=int(an.sum()), share_gated=round(float(an[g].mean()), 4),
                               by_apt={x: int((an & (av == x)).sum()) for x in T.APTS},
                               on_anchored=paired(dv, d, nA & an, denom))
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)



# ---------------------------------------------------------------- winter_deconf (de-confounding: consistent baseline + non-adjacent month control)
def _rebase(df, b):
    return df.with_columns(pl.Series('base', b)).with_columns(
        (pl.col('ab_obs_move') - pl.col('base')).alias('d_move'), (pl.col('ab_obs_first') - pl.col('base')).alias('d_first'),
        (pl.col('ab_obs_push') - pl.col('base')).alias('d_push'), (pl.col('mvt_delay') - pl.col('base')).alias('d_sched'))


def _b12cal(v, cal):
    d = pl.read_parquet('n2/feat.parquet', columns=['MVT_ID_mvt', 'src']).filter(pl.col('src') != 99)
    d = d.with_columns(pl.Series('b12', np.load(f'n2/oof12/{T.POOL_BASE}.npy')))
    b = v.select('MVT_ID_mvt').join(d.select('MVT_ID_mvt', 'b12'), on='MVT_ID_mvt', how='left')['b12'].to_numpy().astype(float)
    ap_ = v['ADEP_mvt'].to_numpy()
    for x, (al_, be) in cal.items():
        b[ap_ == x] = al_ + be * b[ap_ == x]
    return b


def cmd_winter_deconf(a):
    """Two confounders of the winter CV: (1) the test-row baseline is oof12 while the reference training rows use the averaged baseline (the pool rows of W are also oof12, so a consistent baseline is itself an advantage);
    (2) pool months 11/12 are adjacent to the test days. Controls: R = averaged baseline (original set-up) | o12 (the 62-day training rows also use the oof12-calibrated baseline, consistent with the test);
    W06 = January + pool 06 (not adjacent to 11/12; LEMD-06 is also a dense pattern); W1112 = January + pool 11/12 on days outside the test fold.
    The 6 January-full-model airports serve as negative controls (they already have same-pattern samples in January; in a clean January CV adding the winter pool gives only +0.05 pp)."""
    T.PRED_FB = T.CELLFB == 'pred'
    v = T.load_train('n2', FE)
    pool, cal = T.load_pool('n2', v, ['06', '11', '12'])
    pm = pool['date'].str.slice(5, 2).to_numpy()
    te = pool.filter(pl.Series(np.isin(pm, ['11', '12'])))
    pool06 = pool.filter(pl.Series(pm == '06'))
    y, b, g = te['TAXITIME_SEC_mvt'].to_numpy().astype(float), te['base'].to_numpy(), te['gate'].to_numpy()
    apt, dates = te['ADEP_mvt'].to_numpy(), te['date'].to_list()
    fd = day_folds(dates)
    n = te.height
    vr = {'ensavg': v, 'o12': _rebase(v, _b12cal(v, cal))}
    P = {}
    for R, vv in vr.items():
        trv = vv.filter(~pl.col('isA'))
        mo_tr = trv['date'].str.slice(5, 2).to_numpy()
        jan = trv.filter(pl.Series(mo_tr == '01'))
        fa_jan = [x for x in T.APTS if float(jan.filter(pl.col('ADEP_mvt') == x)['gate'].mean() or 0) >= T.FULL_THR]
        rA, _ = T.fit_mixed(trv, te, mode='off')
        rJ, _ = T.fit_predict(jan, te)
        P[f'{R}_ref'] = np.where(np.isin(apt, fa_jan), rJ, rA)
        P[f'{R}_W06'], _ = T.fit_predict(pl.concat([jan, pool06], how='vertical_relaxed'), te)
        w = np.full(n, np.nan)
        for k in range(5):
            tm = fd == k
            w[tm], _ = T.fit_predict(pl.concat([jan, te.filter(pl.Series(fd != k))], how='vertical_relaxed'), te.filter(pl.Series(tm)))
        P[f'{R}_W1112'] = w
        print(R, 'done', flush=True)
    e0 = (b - y) ** 2
    denom = float(e0.sum())
    E = {k: (b + r * g - y) ** 2 for k, r in P.items()}
    E['base'] = e0
    rm = lambda e, m: round(float(np.sqrt(e[m].mean())), 1) if m.any() else None  # noqa: E731
    six = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LSZH']
    groups = {'LEMD': apt == 'LEMD', 'LIRF': apt == 'LIRF', 'six': np.isin(apt, six)}
    groups.update({x: apt == x for x in six})
    out = dict(test_rows=n, base_nonA_sse=denom, cmp={}, rmse={})
    for R in vr:
        for nm in ['W06', 'W1112', 'base']:
            key = f'{R}_{nm}' if nm != 'base' else 'base'
            d = E[key] - E[f'{R}_ref']
            out['cmp'][f'{R}:{nm}-ref'] = {gname: dict(**paired(dates, d, m, denom), **per_fold(d, m, fd, denom)) for gname, m in groups.items()}
            print(R, nm, {gname: (o['pp'], o['se'], o['folds_positive']) for gname, o in out['cmp'][f'{R}:{nm}-ref'].items()}, flush=True)
    for key, e in E.items():
        out['rmse'][key] = {gname: rm(e, m & g) for gname, m in groups.items()}
    print(json.dumps(out['rmse']), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


# ---------------------------------------------------------------- winter_fwd (forward check: pool 11 -> test 12, simulating a one-month neighbour 2025-12 -> 2026-01)
def cmd_winter_fwd(a):
    """winter_deconf shows that a pool from adjacent days of the same month also gives +1.5 pp for the 6 negative-control airports (neighbour/seasonal effect, unrelated to the pattern), while the non-adjacent pool 06 gives only +0.03 pp for LEMD.
    The real situation of 2026-01 is that "the previous month, 2025-12, has samples", so a forward split quantifies the "one-month neighbour": test = all airports in 2025-12,
    training pool = 2025-11 (+06), with no December rows. Each scheme replaces only the December rows of the listed airports, all others follow the reference assignment (6 airports J, the rest A)."""
    T.PRED_FB = T.CELLFB == 'pred'
    v = T.load_train('n2', FE)
    pool, cal = T.load_pool('n2', v, ['06', '11', '12'])
    pm = pool['date'].str.slice(5, 2).to_numpy()
    te = pool.filter(pl.Series(pm == '12'))
    p11, p06 = pool.filter(pl.Series(pm == '11')), pool.filter(pl.Series(pm == '06'))
    if a.test_base:  # Robustness: use another oof12 member as the test-row baseline (the pool stays srp_g5_delta), simulating 2026 where the pool baseline differs from the test baseline (averaged baseline)
        pb, T.POOL_BASE = T.POOL_BASE, a.test_base
        te, _ = T.load_pool('n2', v, ['12'])
        T.POOL_BASE = pb
    y, b, g = te['TAXITIME_SEC_mvt'].to_numpy().astype(float), te['base'].to_numpy(), te['gate'].to_numpy()
    apt, dates = te['ADEP_mvt'].to_numpy(), te['date'].to_list()
    fd = day_folds(dates)
    trv = v.filter(~pl.col('isA'))
    mo_tr = trv['date'].str.slice(5, 2).to_numpy()
    jan = trv.filter(pl.Series(mo_tr == '01'))
    fa_jan = [x for x in T.APTS if float(jan.filter(pl.col('ADEP_mvt') == x)['gate'].mean() or 0) >= T.FULL_THR]
    rA, _ = T.fit_mixed(trv, te, mode='off')
    rJ, _ = T.fit_predict(jan, te)
    ref = np.where(np.isin(apt, fa_jan), rJ, rA)
    P = {'W11': T.fit_predict(pl.concat([jan, p11], how='vertical_relaxed'), te)[0],
         'W0611': T.fit_predict(pl.concat([jan, p06, p11], how='vertical_relaxed'), te)[0],
         'Wp11': T.fit_predict(p11, te)[0]}
    print('fits done', flush=True)
    e0 = (b - y) ** 2
    denom = float(e0.sum())
    eref = (b + ref * g - y) ** 2
    n26, _ = rank_gate_counts(FE)
    sets = {'LEMD': ['LEMD'], 'LEMD+LIRF': ['LEMD', 'LIRF'], 'six': fa_jan, 'eight': fa_jan + ['LEMD', 'LIRF']}
    rm = lambda e, m: round(float(np.sqrt(e[m].mean())), 1) if m.any() else None  # noqa: E731
    out = dict(test='2025-12', test_base=a.test_base or T.POOL_BASE, test_rows=te.height, days=len(set(dates)), base_nonA_sse=denom, fa_jan=fa_jan, var={})
    for nm, r in P.items():
        for sn, aps in sets.items():
            c = np.where(np.isin(apt, aps), r, ref)
            e = (b + c * g - y) ** 2
            d = e - eref
            dm = sum(n26.get(f'{x}-01', 0) * (eref[(apt == x) & g].mean() - e[(apt == x) & g].mean()) for x in aps if ((apt == x) & g).any()) / N26
            o = dict(all=paired(dates, d, np.ones(te.height, bool), denom), **per_fold(d, np.ones(te.height, bool), fd, denom),
                     apt={x: dict(**paired(dates, d, apt == x, denom), rmse_ref=rm(eref, (apt == x) & g), rmse=rm(e, (apt == x) & g))
                          for x in aps},
                     est26=dict(dmse_per_row=round(float(dm), 3)))
            out['var'][f'{nm}|{sn}'] = o
            print(nm, sn, o['all'], o['per_fold_pp'], o['folds_positive'], o['est26'], flush=True)
            print('    ', {x: (z['pp'], z['se'], z['rmse_ref'], z['rmse']) for x, z in o['apt'].items()}, flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


# ---------------------------------------------------------------- pair (day-paired comparison of two 62-day CV OOF files, no training)
def cmd_pair(a):
    v = T.load_train('n2', FE)
    y, b = v['TAXITIME_SEC_mvt'].to_numpy().astype(float), v['base'].to_numpy()
    nA, av, dv = ~v['isA'].to_numpy(), v['ADEP_mvt'].to_numpy(), v['date'].to_list()
    mv = v['date'].str.slice(5, 2).to_numpy()
    fd = day_folds(dv)
    x, ref = np.load(a.a), np.load(a.oof)
    assert len(x) == len(ref) == v.height
    d = (b + x - y) ** 2 - (b + ref - y) ** 2
    denom = float(((b - y) ** 2)[nA].sum())
    out = dict(a=a.a, ref=a.oof, all=dict(**paired(dv, d, nA, denom), **per_fold(d, nA, fd, denom)),
               jan=paired(dv, d, nA & (mv == '01'), denom), jul=paired(dv, d, nA & (mv == '07'), denom),
               apt={x_: paired(dv, d, nA & (av == x_), denom) for x_ in T.APTS})
    print(json.dumps(out), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['pair', 'winter', 'winter_deconf', 'winter_fwd', 'genloao', 'standcv', 'covdrop', 'cv', 'diff'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--oof', default='cvtg_tm.oof.npy')
    ap.add_argument('--mode', default='single', choices=['single', 'cluster'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rate', type=float, default=0.10)
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--test-base', default='')
    ap.add_argument('--test-feats', default='')
    ap.add_argument('--a')
    ap.add_argument('--b')
    a = ap.parse_args()
    Path('resid_cv').mkdir(exist_ok=True)
    if a.cmd == 'diff':
        import dec_eval
        return dec_eval.cmd_diff(a)
    {'winter': cmd_winter, 'genloao': cmd_genloao, 'standcv': cmd_standcv, 'covdrop': cmd_covdrop, 'cv': cmd_cv, 'winter_deconf': cmd_winter_deconf, 'winter_fwd': cmd_winter_fwd, 'pair': cmd_pair}[a.cmd](a)


if __name__ == '__main__':
    main()
