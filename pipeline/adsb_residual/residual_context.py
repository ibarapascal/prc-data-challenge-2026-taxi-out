"""Evaluation and candidate tools for the track-context residual variant: on top of the simpler residual variant's
set-up (feats_tma anchored features + 8 seeds + winter pool 11/12) it adds RESID_TK (the track-context features tk_*
before the segment start). The code directory is the directory of this file (the train_resid in the same directory is
imported; T.__file__ is checked against RESID_CODE_DIR at start-up to avoid silently running an older copy).

Subcommands (outputs go to v22/):
  cv --tk gap|all|off --seeds 8 --out v22/cv_tk<name>_s8.json [--oof resid_cv/cv_anchor_s8.oof.npy]
      Date-grouped 5-fold CV on the 62 training days (seed 0, see day_folds). Environment = the CV set-up of the simpler
      variant: RESID_FEATS / RESID_STAND_DIR = feats_tma, RESID_CELLFB=off, RESID_WINTER unset (the winter pool only
      affects model selection for 2026-01 and cannot be measured by the 62-day CV; winter_fwd tests it instead).
      The tk columns enter BASEF/GENERIC via T.set_tk; the OOF (gated residual) is saved; paired by day against --oof
      (8-seed OOF of the simpler variant).
      --tk off = reproduction check (must equal --oof row by row, max_abs_diff = 0).
  evalx --ref resid_cv/cv_anchor_s8.oof.npy --new gap=v22/cv_tkgap_s8.oof.npy[,all=...] --keep EGLL-07,LIRF --out v22/cv_eval.json
      No training: compares each variant with ref on all nonA rows / excluding LIRF-07 / January / July / per airport /
      per cell; conservative version (cells listed in --keep take ref = the simpler variant's original prediction);
      per-stand residual correction (second switch, no training, strictly fold-wise, m=20, applied only to gated nonA rows
      outside LIRF-07 and outside the keep cells) on top of the full and the conservative version;
      2026 conversion (estimate: 2025 CV per-cell MSE reduction per gated row x number of 2026 gated rows per cell);
      also stores the 2026 stand-correction table v22/sc_table_<name>.parquet.
  clsx --ref resid_cv/cv_anchor_s8.oof.npy --new gap=v22/cv_tkgap_s8.oof.npy --keep EGLL-07,LIRF --out v22/cls_gap.json
      No training: splits the evalx comparisons by row class (B / C / D / S, 2025 rows of gap/track_rows.parquet; LIRF-07 nonA
      excluded) plus the drift of per-cell gated-row means.
      All json files are written with explicit utf-8.
  winter_fwd --tk gap,all --seeds 8 --out v22/winter_fwd_tk.json [--sc-oof v22/cv_tkgap_s8.oof.npy --sc-tk gap]
      Forward split (guards against drift between periods): train = 2025-01 (the January rows of the 62 training days)
      + pool 2025-11, test = all airports in 2025-12 (isA rows excluded).
      W11 (full model on January + pool 11, as used for the seven airports in 2026-01) and A (62-day model with tk off,
      as used for 2026-07) are each compared with tk off / on; reported for the 7 airports / all / excluding LIRF.
      If --sc-oof is given and the file exists, also tests, on December forward, a stand-correction table built from
      the 62-day TK OOF residuals.
  merge --base resid_pred_early/upstanding-firefly_variant_simple.parquet --new resid_pred/upstanding-firefly_variant_gapfam.parquet --keep EGLL-07,LIRF
        --out resid_pred/upstanding-firefly_variant_gapfam_merged.parquet
      Conservative merge: rows of the airport-months listed in --keep (or of all months of an airport if only the airport
      is given) take --base (the simpler variant's original prediction), all other rows take --new.
  applysc --sub X.parquet --res X.res.npy --table v22/sc_table_<name>.parquet --keep EGLL-07,LIRF --out Y.parquet
      (Only used if the stand correction passes its gate.) Adds the stand correction to a candidate:
      p = clip(base + res + corr), corr is added only on gated rows outside the keep cells; base = the test-side baseline file (RESID_BASE_SUB).
  diff --a X.parquet --b Y.parquet --out Z.json      row-by-row difference (share of changed rows, mean, RMS, by airport x month)
  md5 --files a,b --expect a=<md5>,b=<md5> --out Z.json    check file md5 values against the expected ones (with the switches off the simpler variant is reproduced bit for bit)
Gate: versus the reference >= +0.1 pp, > 2 SE, positive in the majority of folds (>= 3/5); the pp denominator is the
baseline SSE over all nonA rows of the evaluation set, and +- is the day-paired SE.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_resid as T  # noqa: E402

assert Path(T.__file__).resolve().parent == HERE, ('train_resid is not the copy next to this script', T.__file__, HERE)
_CD = os.environ.get('RESID_CODE_DIR')
if _CD:
    assert str(Path(_CD).resolve()).lower() == str(HERE).lower(), ('RESID_CODE_DIR differs from the script directory', _CD, HERE)

FE = os.environ.get('RESID_FEATS', 'feats_tma')
N26 = 344841
W7 = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LSZH', 'LEMD']


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


def gate_pass(o):
    return bool(o['pp'] >= 0.1 and o['pp'] > 2 * o['se'] and o.get('folds_positive', 5) >= 3)



def cells(v):
    return np.array([f'{x}-{d[5:7]}' for x, d in zip(v['ADEP_mvt'].to_list(), v['date'].to_list())])


def keep_mask(key, keep):
    ks = [k for k in keep.split(',') if k]
    return np.array([(k_ in ks) or (k_[:4] in ks) for k_ in key])


def base_arrays(v):
    y = v['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    return (y, v['base'].to_numpy(), v['gate'].to_numpy(), ~v['isA'].to_numpy(), v['ADEP_mvt'].to_numpy(),
            v['date'].to_list(), v['date'].str.slice(5, 2).to_numpy())


def summarize(v, new, ref, fd, key=None):
    """new / ref: 62-day OOF (gated residuals). Summaries paired by day against ref, for each slice."""
    y, b, g, nA, av, dv, mv = base_arrays(v)
    key = cells(v) if key is None else key
    x7 = key != 'LIRF-07'
    e_ref, e_new = (b + ref - y) ** 2, (b + new - y) ** 2
    d = e_new - e_ref
    denom = float(((b - y) ** 2)[nA].sum())
    out = dict(max_abs_diff=float(np.max(np.abs(new - ref))), changed_rows=int((new != ref).sum()),
               all=dict(**paired(dv, d, nA, denom), **per_fold(d, nA, fd, denom)),
               excl_lirf07=dict(**paired(dv, d, nA & x7, denom), **per_fold(d, nA & x7, fd, denom)),
               jan=paired(dv, d, nA & (mv == '01'), denom), jul_excl_lirf07=paired(dv, d, nA & (mv == '07') & x7, denom),
               lirf07=paired(dv, d, nA & ~x7, denom),
               apt={x: paired(dv, d, nA & (av == x), denom) for x in T.APTS if (nA & (av == x)).any()})
    cl = {}
    for k_ in sorted(set(key[g])):
        m = nA & g & (key == k_)
        if m.sum() >= 300:
            cl[k_] = dict(n_gated=int(m.sum()), red_gated=round(float(-d[m].sum() / e_ref[m].sum()), 4),
                          dmse_per_gated_row=round(float(-d[m].mean()), 2), mse_ref_gated=round(float(e_ref[m].mean()), 1))
    out['cell'] = cl
    m = nA & g & x7
    out['gated_red_rel_excl_lirf07'] = round(float(-d[m].sum() / e_ref[m].sum()), 4)
    out['gate_all'] = gate_pass(out['all'])
    out['gate_excl_lirf07'] = gate_pass(out['excl_lirf07'])
    return out


def load_oof(path, n):
    x = np.load(path)
    assert len(x) == n, (path, len(x), n)
    return x


# ---------------------------------------------------------------- cv
def cmd_cv(a):
    assert T.WINTER == 'off', 'CV must not set RESID_WINTER (the winter pool is only used for the 2026-01 predictions)'
    if T.CELLFB != 'off':
        print('Warning: RESID_CELLFB is not off; the reference cv_anchor_s8 was made with it off (cmd_cv does not enable PRED_FB, so it is effectively off)', flush=True)
    T.TK_LOAD = True
    v = T.load_train('n2', FE)
    T.set_tk(T.tk_cols(a.tk))
    print('cv tk', a.tk, 'cols', T.TKC, 'BASEF tail', T.BASEF[-8:], 'GENERIC tail', T.GENERIC[-8:], 'feats', FE, flush=True)
    g, isA, dv = v['gate'].to_numpy(), v['isA'].to_numpy(), v['date'].to_list()
    fd = day_folds(dv)
    res = np.zeros(v.height)
    for k in range(5):
        r, _, _ = T.fit_all(v.filter(pl.Series((fd != k) & ~isA)), v.filter(pl.Series(fd == k)), seeds=tuple(range(a.seeds)))
        res[fd == k] = np.where(g[fd == k], r, 0.0)
        print('fold', k, 'done', flush=True)
    T.set_tk([])
    np.save(a.out.replace('.json', '.oof.npy'), res)
    ref = load_oof(a.oof, v.height)
    out = dict(tk=a.tk, tk_cols=T.tk_cols(a.tk), seeds=a.seeds, feats=FE, ref=a.oof, code=str(T.__file__), **summarize(v, res, ref, fd))
    print('cv', a.tk, json.dumps({k: out[k] for k in ('max_abs_diff', 'all', 'excl_lirf07', 'gated_red_rel_excl_lirf07', 'gate_all',
                                                     'gate_excl_lirf07')}), flush=True)
    json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- per-stand residual correction (second switch)
def sc_table(keys, resid, mask, m_s=20.0):
    df = pl.DataFrame({'k': keys[mask], 'r': resid[mask]}).filter(pl.col('k').is_not_null()).group_by('k').agg(
        pl.col('r').sum().alias('s'), pl.len().alias('n'))
    return {k_: s_ / (n_ + m_s) for k_, s_, n_ in df.iter_rows()}  # rows without a stand are not corrected (table lookup gives 0)


def sc_cv(keys, resid, mask, fd, m_s=20.0):
    """Strictly fold-wise: the correction for a test fold comes only from the OOF residuals of the other folds (the OOF itself comes from models that have seen the test fold, so this is slightly optimistic)."""
    corr = np.zeros(len(resid))
    for k in range(5):
        mp = sc_table(keys, resid, mask & (fd != k), m_s)
        te = mask & (fd == k)
        corr[te] = [mp.get(k_, 0.0) for k_ in keys[te]]
    return corr


def rank_gated_by_cell(feats):
    rk = T.load_rank('n2', feats)
    m = rk.filter(pl.col('gate')).with_columns(pl.col('date').str.slice(5, 2).alias('mo'))
    return {f'{a_}-{mo}': int(n) for a_, mo, n in m.group_by('ADEP_mvt', 'mo').len().iter_rows()}


def est26(cl, n26g, keep_cells, apt_fb=True):
    """2026 conversion (estimate): sum over cells of (2026 gated rows x 2025 CV MSE reduction per gated row in that cell) / N26; cells without 2025 samples fall back to the airport mean (apt_fb)."""
    tot, used = 0.0, {}
    for k_, n in n26g.items():
        if k_ in keep_cells or k_[:4] in keep_cells:
            continue
        if k_ in cl:
            x, src = cl[k_]['dmse_per_gated_row'], 'cell'
        elif apt_fb:
            same = [cl[c]['dmse_per_gated_row'] * cl[c]['n_gated'] for c in cl if c[:4] == k_[:4]]
            nn = sum(cl[c]['n_gated'] for c in cl if c[:4] == k_[:4])
            if not nn:
                continue
            x, src = sum(same) / nn, 'apt'
        else:
            continue
        tot += n * x
        used[k_] = dict(n26_gated=n, dmse_per_row=round(float(x), 2), src=src)
    dm = tot / N26
    return dict(dmse_per_row_2026=round(float(dm), 2), cells=used)


def cmd_evalx(a):
    v = T.load_train('n2', FE)
    y, b, g, nA, av, dv, mv = base_arrays(v)
    fd = day_folds(dv)
    key = cells(v)
    km = keep_mask(key, a.keep)
    ref = load_oof(a.ref, v.height)
    n26g = rank_gated_by_cell(FE)
    sk = v['stand_key'].to_numpy()
    G = g & nA & (key != 'LIRF-07')
    out = dict(ref=a.ref, keep=a.keep, feats=FE, n26_gated=n26g, var={})
    for nm, path in [x.split('=', 1) for x in a.new.split(',') if x]:
        if not Path(path).exists():
            print('skip missing', path, flush=True)
            continue
        new = load_oof(path, v.height)
        cons = np.where(km, ref, new)
        o = {'full': summarize(v, new, ref, fd, key), 'cons': summarize(v, cons, ref, fd, key)}
        for vn, arr, msk in (('full', new, G), ('cons', cons, G & ~km)):
            resid = y - (b + arr)
            corr = sc_cv(sk, resid, msk, fd)
            arr_sc = arr + np.where(msk, corr, 0.0)
            o[f'{vn}_sc_vs_{vn}'] = summarize(v, arr_sc, arr, fd, key)   # the stand correction itself (gate of the second switch)
            o[f'{vn}_sc_vs_ref'] = summarize(v, arr_sc, ref, fd, key)
            o[f'{vn}_sc_vs_{vn}']['n_keys'] = int(len(set(sk[msk].tolist())))
            tab = sc_table(sk, resid, msk)
            pl.DataFrame({'stand_key': list(tab.keys()), 'corr': list(tab.values())}).write_parquet(f'v22/sc_table_{nm}_{vn}.parquet')
        kc = [k for k in a.keep.split(',') if k]
        for vn, kcs in (('full', []), ('cons', kc)):
            e = est26(o[vn]['cell'], n26g, kcs)
            e['note'] = 'Estimate: 2025 CV per-cell reduction x number of 2026 gated rows (no k factor); the seven January airports actually use the winter-pool model, so the winter_fwd forward result is the reference'
            o[f'est26_{vn}'] = e
        out['var'][nm] = o
        print(nm, 'full', o['full']['all'], o['full']['excl_lirf07'], 'cons', o['cons']['all'], o['cons']['excl_lirf07'],
              'sc(full)', o['full_sc_vs_full']['excl_lirf07'], 'sc(cons)', o['cons_sc_vs_cons']['excl_lirf07'],
              'est26', o['est26_full']['dmse_per_row_2026'], o['est26_cons']['dmse_per_row_2026'], flush=True)
    json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


def cmd_clsx(a):
    """Splits the evalx comparisons by row class (klass: B / C / D / S, taken from the 2025 rows of gap/track_rows.parquet) and by per-cell mean drift (no training)."""
    v = T.load_train('n2', FE)
    y, b, g, nA, av, dv, mv = base_arrays(v)
    fd = day_folds(dv)
    key = cells(v)
    km = keep_mask(key, a.keep)
    ref = load_oof(a.ref, v.height)
    sk = v['stand_key'].to_numpy()
    gh = pl.read_parquet('gap/track_rows.parquet', columns=['yr', 'MVT_ID_mvt', 'cls']).filter(pl.col('yr') == 2025).unique('MVT_ID_mvt')
    cj = v.select('MVT_ID_mvt').with_row_index('_r').join(gh.select('MVT_ID_mvt', 'cls'), on='MVT_ID_mvt', how='left').sort('_r')
    assert cj.height == v.height
    cls = cj['cls'].fill_null('NA').to_numpy()
    x7 = key != 'LIRF-07'
    G = g & nA & x7
    denom = float(((b - y) ** 2)[nA].sum())

    def by(new, base):
        d = (b + new - y) ** 2 - (b + base - y) ** 2
        o = {}
        for c in sorted(set(cls[nA & x7].tolist())):
            m = nA & x7 & (cls == c)
            o[c] = dict(n=int(m.sum()), n_gated=int((m & g).sum()), **paired(dv, d, m, denom))
        return o

    out = dict(ref=a.ref, keep=a.keep, feats=FE, cls_rows_matched=float((cls != 'NA').mean()), var={})
    for nm, path in [x.split('=', 1) for x in a.new.split(',') if x]:
        new = load_oof(path, v.height)
        cons = np.where(km, ref, new)
        arr = {'full': new, 'cons': cons}
        for vn, msk in (('full', G), ('cons', G & ~km)):
            corr = sc_cv(sk, y - (b + arr[vn]), msk, fd)
            arr[f'{vn}_sc'] = arr[vn] + np.where(msk, corr, 0.0)
        o = {'full': by(arr['full'], ref), 'cons': by(arr['cons'], ref),
             'full_sc_vs_full': by(arr['full_sc'], arr['full']), 'cons_sc_vs_cons': by(arr['cons_sc'], arr['cons'])}
        gm = g & nA
        o['cell_mean_shift_gated'] = {k_: dict(n=int((gm & (key == k_)).sum()),
                                               full=round(float((new - ref)[gm & (key == k_)].mean()), 2),
                                               sc_full=round(float((arr['full_sc'] - new)[gm & (key == k_)].mean()), 2))
                                      for k_ in sorted(set(key[gm].tolist())) if (gm & (key == k_)).sum() >= 300}
        out['var'][nm] = o
        print(nm, {c: (x['pp'], x['se']) for c, x in o['full'].items()}, flush=True)
    json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- winter_fwd (forward split)
def cmd_winter_fwd(a):
    T.PRED_FB = T.CELLFB == 'pred'
    T.TK_LOAD = True
    v = T.load_train('n2', FE)
    pool, cal = T.load_pool('n2', v, ['11', '12'])
    pm = pool['date'].str.slice(5, 2).to_numpy()
    te, p11 = pool.filter(pl.Series(pm == '12')), pool.filter(pl.Series(pm == '11'))
    y, b, g = te['TAXITIME_SEC_mvt'].to_numpy().astype(float), te['base'].to_numpy(), te['gate'].to_numpy()
    apt, dates = te['ADEP_mvt'].to_numpy(), te['date'].to_list()
    fd = day_folds(dates)
    trv = v.filter(~pl.col('isA'))
    jan = trv.filter(pl.Series(trv['date'].str.slice(5, 2).to_numpy() == '01'))
    seeds = tuple(range(a.seeds))
    tks = ['off'] + [t for t in a.tk.split(',') if t and t != 'off']
    P = {}
    for tk in tks:
        T.set_tk(T.tk_cols(tk))
        P[f'W11_{tk}'] = T.fit_predict(pl.concat([jan, p11], how='vertical_relaxed'), te, seeds=seeds)[0]
        P[f'A_{tk}'] = T.fit_mixed(trv, te, mode='off', seeds=seeds)[0]
        print('fits', tk, 'done', flush=True)
    T.set_tk([])
    e0 = (b - y) ** 2
    denom = float(e0.sum())
    n26g = rank_gated_by_cell(FE)
    w7 = np.isin(apt, W7)
    sets = {'w7': w7, 'all': np.ones(len(y), bool), 'excl_LIRF': apt != 'LIRF'}
    rm = lambda e, m: round(float(np.sqrt(e[m].mean())), 1) if m.any() else None  # noqa: E731
    out = dict(test='2025-12', train='2025-01+pool 2025-11 (W11) / 62 days off (A)', seeds=a.seeds, feats=FE, pool_feats=T.POOL_FEATS,
               test_rows=te.height, days=len(set(dates)), base_nonA_sse=denom, cal=cal, var={})
    for tk in tks[1:]:
        for M in ('W11', 'A'):
            e_off, e_on = (b + P[f'{M}_off'] * g - y) ** 2, (b + P[f'{M}_{tk}'] * g - y) ** 2
            d = e_on - e_off
            o = {s: dict(**paired(dates, d, m, denom), **per_fold(d, m, fd, denom)) for s, m in sets.items()}
            o['apt'] = {x: dict(**paired(dates, d, apt == x, denom), rmse_off=rm(e_off, (apt == x) & g), rmse_on=rm(e_on, (apt == x) & g),
                                dmse_per_gated_row=round(float(-d[(apt == x) & g].mean()), 2) if ((apt == x) & g).any() else None)
                        for x in T.APTS if (apt == x).any()}
            o['gate_w7'] = gate_pass(o['w7'])
            o['gate_all'] = gate_pass(o['all'])
            if M == 'W11':  # 2026-01 conversion for the seven airports (winter-pool model): forward reduction per gated row x number of 2026-01 gated rows (estimate)
                dm = sum(n26g.get(f'{x}-01', 0) * (o['apt'][x]['dmse_per_gated_row'] or 0) for x in W7 if x in o['apt']) / N26
                o['est26_jan_w7'] = dict(dmse_per_row_2026=round(float(dm), 2))
            out['var'][f'{M}|{tk}'] = o
            print(M, tk, {s: (o[s]['pp'], o[s]['se'], o[s]['folds_positive']) for s in sets}, o.get('est26_jan_w7'), flush=True)
    if a.sc_oof and Path(a.sc_oof).exists() and f'W11_{a.sc_tk}' in P:
        oof = load_oof(a.sc_oof, v.height)
        yv, bv, gv, nAv, _, _, _ = base_arrays(v)
        Gv = gv & nAv & (cells(v) != 'LIRF-07')
        tab = sc_table(v['stand_key'].to_numpy(), yv - (bv + oof), Gv)
        corr = np.array([tab.get(k_, 0.0) for k_ in te['stand_key'].to_list()]) * (g & w7)
        e_on = (b + P[f'W11_{a.sc_tk}'] * g - y) ** 2
        e_sc = (b + P[f'W11_{a.sc_tk}'] * g + corr - y) ** 2
        d = e_sc - e_on
        out['sc_fwd'] = dict(tk=a.sc_tk, oof=a.sc_oof, n_rows_corrected=int((corr != 0).sum()),
                             w7=dict(**paired(dates, d, w7, denom), **per_fold(d, w7, fd, denom)),
                             apt={x: paired(dates, d, apt == x, denom) for x in W7})
        out['sc_fwd']['gate_w7'] = gate_pass(out['sc_fwd']['w7'])
        print('sc_fwd', out['sc_fwd']['w7'], flush=True)
    else:
        out['sc_fwd'] = f'skipped (sc_oof={a.sc_oof!r} missing or tk not run)'
    json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- candidates: conservative merge, stand correction, difference, md5
def rank_keys(ids):
    full = pl.read_parquet('n2/feat.parquet', columns=['MVT_ID_mvt', 'src', 'ADEP_mvt', 'MVT_TIME_UTC_mvt']).filter(pl.col('src') == 99)
    k = ids.to_frame().with_row_index('_r').join(full, on='MVT_ID_mvt', how='left').sort('_r')
    assert k.height == len(ids) and k['ADEP_mvt'].null_count() == 0
    return np.array([f'{x}-{t:%m}' for x, t in zip(k['ADEP_mvt'].to_list(), k['MVT_TIME_UTC_mvt'].to_list())])


def cmd_merge(a):
    base, new = pl.read_parquet(a.base), pl.read_parquet(a.new)
    assert base.columns == new.columns == ['MVT_ID_mvt', 'TAXITIME_SEC_mvt'], (base.columns, new.columns)
    assert base['MVT_ID_mvt'].equals(new['MVT_ID_mvt']), 'the two files have different row order'
    key = rank_keys(new['MVT_ID_mvt'])
    km = keep_mask(key, a.keep)
    col = np.where(km, base['TAXITIME_SEC_mvt'].to_numpy(), new['TAXITIME_SEC_mvt'].to_numpy())
    out = new.with_columns(pl.Series('TAXITIME_SEC_mvt', col).cast(new.schema['TAXITIME_SEC_mvt']))
    out.write_parquet(a.out)
    info = dict(base=a.base, new=a.new, keep=a.keep, rows=out.height, kept_rows=int(km.sum()),
                kept_by_cell={k_: int((km & (key == k_)).sum()) for k_ in sorted(set(key[km]))},
                md5=hashlib.md5(open(a.out, 'rb').read()).hexdigest())
    json.dump(info, open(a.out.replace('.parquet', '.merge.json'), 'w', encoding='utf-8'), indent=1)
    print('merge', info, flush=True)


def cmd_applysc(a):
    sub = pl.read_parquet(a.sub)
    res = np.load(a.res)
    rk = T.load_rank('n2', FE)
    assert rk['MVT_ID_mvt'].equals(sub['MVT_ID_mvt']) and len(res) == sub.height
    tab = dict(pl.read_parquet(a.table).iter_rows())
    key = rank_keys(sub['MVT_ID_mvt'])
    msk = rk['gate'].to_numpy() & ~keep_mask(key, a.keep)
    corr = np.array([tab.get(k_, 0.0) for k_ in rk['stand_key'].to_list()]) * msk
    p = np.clip(rk['base'].to_numpy() + res + corr, 0, None)
    chk = np.clip(rk['base'].to_numpy() + res, 0, None)
    assert (np.round(chk).astype(np.int32) == sub['TAXITIME_SEC_mvt'].to_numpy()).all(), 'res.npy does not correspond to the candidate'
    out = pl.DataFrame({'MVT_ID_mvt': sub['MVT_ID_mvt'], 'TAXITIME_SEC_mvt': np.round(p).astype(np.int32)})
    out.write_parquet(a.out)
    info = dict(sub=a.sub, table=a.table, rows_corrected=int((corr != 0).sum()), rms_corr=float(np.sqrt(np.mean(corr ** 2))),
                md5=hashlib.md5(open(a.out, 'rb').read()).hexdigest())
    json.dump(info, open(a.out.replace('.parquet', '.sc.json'), 'w', encoding='utf-8'), indent=1)
    print('applysc', info, flush=True)


def cmd_diff(a):
    x = pl.read_parquet(a.a).rename({'TAXITIME_SEC_mvt': 'a'})
    z = pl.read_parquet(a.b).rename({'TAXITIME_SEC_mvt': 'b'})
    full = pl.read_parquet('n2/feat.parquet', columns=['MVT_ID_mvt', 'src', 'ADEP_mvt', 'MVT_TIME_UTC_mvt']).filter(pl.col('src') == 99)
    m = x.join(z, on='MVT_ID_mvt').join(full, on='MVT_ID_mvt').with_columns(
        (pl.col('b').cast(pl.Float64) - pl.col('a')).alias('d'), pl.col('MVT_TIME_UTC_mvt').dt.strftime('%m').alias('mo'))
    assert m.height == x.height == z.height, (m.height, x.height, z.height)
    by = m.group_by('ADEP_mvt', 'mo').agg(pl.len().alias('n'), (pl.col('d') != 0).mean().alias('changed'),
                                          pl.col('d').mean().alias('mean'), (pl.col('d') ** 2).mean().sqrt().alias('rms')).sort('ADEP_mvt', 'mo')
    out = dict(a=a.a, b=a.b, rows=m.height, changed=float((m['d'] != 0).mean()), mean=float(m['d'].mean()),
               rms=float(np.sqrt((m['d'] ** 2).mean())), mean_d2=float((m['d'] ** 2).mean()),
               by_apt_mo=[{k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in r.items()} for r in by.iter_rows(named=True)])
    print(json.dumps({k: out[k] for k in ('rows', 'changed', 'mean', 'rms', 'mean_d2')}), flush=True)
    json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=1)


def cmd_md5(a):
    exp = dict(x.rsplit('=', 1) for x in a.expect.split(',') if x) if a.expect else {}
    out = {}
    for f in [x for x in a.files.split(',') if x]:
        h = hashlib.md5(open(f, 'rb').read()).hexdigest() if Path(f).exists() else None
        out[f] = dict(md5=h, expect=exp.get(f), ok=(h == exp[f]) if f in exp else None)
    out['all_ok'] = all(o['ok'] for o in out.values() if isinstance(o, dict) and o['ok'] is not None)
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['cv', 'evalx', 'clsx', 'winter_fwd', 'merge', 'applysc', 'diff', 'md5'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--tk', default='gap')
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--oof', default='resid_cv/cv_anchor_s8.oof.npy')
    ap.add_argument('--ref', default='resid_cv/cv_anchor_s8.oof.npy')
    ap.add_argument('--new', default='')
    ap.add_argument('--keep', default='EGLL-07,LIRF')
    ap.add_argument('--sc-oof', default='')
    ap.add_argument('--sc-tk', default='gap')
    ap.add_argument('--base')
    ap.add_argument('--sub')
    ap.add_argument('--res')
    ap.add_argument('--table')
    ap.add_argument('--files', default='')
    ap.add_argument('--expect', default='')
    ap.add_argument('--a')
    ap.add_argument('--b')
    a = ap.parse_args()
    Path('v22').mkdir(exist_ok=True)
    print('code', T.__file__, 'RESID_CODE_DIR', _CD, 'FE', FE, 'TK', T.TK, flush=True)
    {'cv': cmd_cv, 'evalx': cmd_evalx, 'clsx': cmd_clsx, 'winter_fwd': cmd_winter_fwd, 'merge': cmd_merge, 'applysc': cmd_applysc, 'diff': cmd_diff,
     'md5': cmd_md5}[a.cmd](a)


if __name__ == '__main__':
    main()
