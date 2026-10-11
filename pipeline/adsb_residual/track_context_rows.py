"""Accuracy-ceiling diagnostics on gated rows (rows with an ADS-B ground segment): how far is the reference residual
configuration (columns/keys named refcfg below) from what is attainable on these rows.

Contents: (1) per airport x month MSE of the averaged baseline (ensavg) and the reference configuration (refcfg), and f;
(2) "direct measurement" estimates pred = MVT - t_X + offset (t_X = segment start / first point with gs >= 1/3/5/8 /
point 30 m from the segment start / push-back after standing still at the segment start / push-back detected by reversing
or turn reversal / extended segment / leaving the registered stand), compared row by row with the reference;
(3) source decomposition of the head errors (top 1/5/20%) of the reference on gated rows;
(4) class-A rows (y > 3 h and not LIRF-B): long-window tracks, can a "movement from standstill" after arrival, a push-back
or a return to the stand be seen hours before take-off; (5) 2026 weight conversion; the fusion subcommand (training)
is a strictly fold-wise small-model fusion upper bound.
Set-up: the reference configuration = feats_tm + RESID_STAND=fix + RESID_EXTRA=art + RESID_MONTH=hyb:01 + RESID_FULL_THR=0.15,
without rule 1 of appendix A; OOF = cvtg_tm.oof.npy (date-grouped 5-fold CV on the 62 days, grid-cell fallback not applied on the test
side; CELLFB=pred differs by only +0.017 +- 0.014 pp on the regular CV); folds = see folds() below (seed 0).
pp = delta SSE / baseline SSE over all nonA rows of the 62 days; +- is the day-paired SE. Labels are used only for
evaluation, classification and offsets (offsets strictly fold-wise).
Run from the working directory (with the python of the project venv):
  rows   --out gap/track_rows.parquet                     per-row table (62 days of 2025 + 62 days of 2026): OOF, classes, anchoring columns, trajectory features recomputed from the crop
  report --rows gap/track_rows.parquet --out gap/track_diag.json      zero-training diagnostics (1) to (5)
  fusion --rows gap/track_rows.parquet --out gap/track_fusion.json    training: small LightGBM on the reference prediction plus direct-measurement features, strictly fold-wise
train_resid / build_feats are taken from the code directory (TC_CODE_DIR; with the new switches off they reproduce the reference bit for bit; this script only reads them).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
TC_CODE = Path(os.environ.get('TC_CODE_DIR', str(HERE.parent / 'resid_code_dir')))
for _k, _v in dict(RESID_EXTRA='art', RESID_MONTH='hyb:01', RESID_FULL_THR='0.15', RESID_NG='off', RESID_STAND='fix',
                   RESID_STAND_DIR='feats_tm', RESID_THREADS='4').items():
    os.environ.setdefault(_k, _v)
sys.path.insert(0, str(TC_CODE))
import train_resid as T  # noqa: E402
import build_feats as BF  # noqa: E402

FE = os.environ.get('RESID_FEATS', 'feats_tm')
OOF = os.environ.get('TC_OOF', 'cvtg_tm.oof.npy')
OOF8 = os.environ.get('TC_OOF8', 'resid_cv/cv_tm_s8.oof.npy')
TMA = os.environ.get('TC_TMA', 'feats_tma')
CROP = os.environ.get('TC_CROP', 'crop')
STANDS = os.environ.get('TC_STANDS', 'ext/xplane-apt/stand_coords_merged.parquet')
GROUND, MISSING = -1000, -9999
APTS = T.APTS
NAMES = sorted(json.load(open(TC_CODE / 'airports.json')))
assert NAMES == APTS, NAMES
N26 = 344841
QS = (.01, .05, .25, .5, .75, .95, .99)


def folds(dates, nfold=5, seed=0):
    """Day-grouped folds: the sorted dates are permuted with seed 0 and assigned to folds in turn."""
    days = sorted(set(dates))
    perm = np.random.default_rng(seed).permutation(len(days))
    fold_of = {days[i]: k % nfold for k, i in enumerate(perm)}
    return np.array([fold_of[d] for d in dates])


def hav(lat1, lon1, lat2, lon2):
    return BF.hav(lat1, lon1, lat2, lon2)


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def angdiff(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------- trajectory recomputation (label-free)
def load_pts(date, hexes):
    d = np.datetime64(date)
    fs = [Path(CROP) / str(x)[:4] / f'{x}.parquet' for x in (d - 1, d)]
    fs = [f for f in fs if f.exists()]
    if not (Path(CROP) / date[:4] / f'{date}.parquet').exists():
        return None
    cols = ['hex', 'apt', 't', 'lat', 'lon', 'alt', 'gs', 'trk']
    hx = hexes.implode()  # filter by the matched hex first, then de-duplicate (same keep rule as build_feats.load_points: the first one in concatenation order)
    p = pl.concat([pl.scan_parquet(f).select(cols).filter(pl.col('hex').is_in(hx)).collect() for f in fs])
    return p.unique(['hex', 't'], keep='first').sort(['hex', 't'])


def _first(mask, arr):
    i = np.flatnonzero(mask)
    return float(arr[i[0]]) if len(i) else np.nan


def row_track(t, lat, lon, gs, alt, trk, t_end, mvt, sll):
    """Trajectory features of one row. t etc. = all points of the same hex and airport (sorted by t); t_end = last ground point before the take-off event (mvt - ab_last_ground).
    All returned times are converted to obs = mvt - t (same sign convention as ab_obs_*); NaN = not available."""
    o = {}
    i_end = int(np.searchsorted(t, t_end + 1e-3, side='right') - 1)
    if i_end < 0 or abs(t[i_end] - t_end) > 0.01:
        o['tk_ok'] = 0
        return o
    o['tk_ok'] = 1
    lo = max(int(np.searchsorted(t, t[i_end] - 64800.0, side='left')) - 1, 0)   # 18 h window
    ts, la, ln, al = t[lo:i_end + 1], lat[lo:i_end + 1], lon[lo:i_end + 1], alt[lo:i_end + 1]
    g0 = np.nan_to_num(gs[lo:i_end + 1].astype(float), nan=0.0)
    tr = trk[lo:i_end + 1].astype(float)
    e = len(ts) - 1
    dt = np.diff(ts)
    # --- 1) current segment logic (same rule as build_feats.seg_feats: gap >= 600 s, > 3 h, cut at a non-ground point), and record why the trace stopped
    b_gap = dt >= 600
    b_time = (ts[e] - ts[:-1]) > 10800
    b_alt = al[:-1] != GROUND
    idx = np.flatnonzero(b_gap | b_time | b_alt)
    if len(idx):
        q = idx[-1]
        j = q + 1
        o['stop'] = 'gap' if b_gap[q] else ('max3h' if b_time[q] else ('air' if al[q] > GROUND else 'miss'))
        if b_gap[q]:
            o['stop_gap_s'] = float(dt[q])
            o['stop_gap_m'] = float(hav(la[q], ln[q], la[j], ln[j]))
    else:
        j = 0
        o['stop'] = 'start' if lo == 0 else 'win'
    st, sla, sln, sg, strk = ts[j:], la[j:], ln[j:], g0[j:], tr[j:]
    o['first'] = mvt - st[0]
    for k in (1, 5, 8):
        o[f'gs{k}'] = mvt - _first(sg >= k, st)
    o['disp30'] = mvt - _first(hav(sla[0], sln[0], sla, sln) > 30, st)
    o['rev'] = np.nan
    o['turn'] = np.nan
    o['trk_agree'] = np.nan
    if len(st) >= 2:
        dd = hav(sla[:-1], sln[:-1], sla[1:], sln[1:])
        b = bearing(sla[:-1], sln[:-1], sla[1:], sln[1:])
        ok = (dd >= 3) & (dd <= 300) & (np.diff(st) <= 30)
        t2 = strk[1:]
        df_ = angdiff(b, t2)
        okt = ok & ~np.isnan(t2)
        if okt.any():
            o['trk_agree'] = float((df_[okt] <= 45).mean())
        rev = okt & (df_ > 135) & (sg[1:] <= 12)
        o['rev'] = mvt - _first(rev, st[:-1])
        oki = np.flatnonzero(ok)
        if len(oki):
            t0, b0 = st[oki[0]], b[oki[0]]
            later = oki[st[oki] - t0 <= 600]
            if (angdiff(b[later], b0) > 135).any():
                o['turn'] = mvt - t0
    # --- 2) extended segment: standstill bridges gaps (neighbouring points < 50 m apart, gs < 1, < 6 h); points without altitude and gs < 40 count as ground; upper limit 12 h
    dpv = hav(la[:-1], ln[:-1], la[1:], ln[1:])
    bridge = (dpv < 50) & (g0[:-1] < 1) & (g0[1:] < 1) & (dt < 21600)
    x_gap = (dt >= 600) & ~bridge
    x_time = (ts[e] - ts[:-1]) > 43200
    x_air = (al[:-1] > GROUND) | ((al[:-1] == MISSING) & (g0[:-1] >= 40)) | ((al[:-1] != GROUND) & (al[:-1] != MISSING) & (al[:-1] <= GROUND))
    idx = np.flatnonzero(x_gap | x_time | x_air)
    if len(idx):
        q = idx[-1]
        jx = q + 1
        o['stop_x'] = 'gap' if x_gap[q] else ('max12h' if x_time[q] else 'air')
    else:
        jx = 0
        o['stop_x'] = 'start' if lo == 0 else 'win'
    xt, xla, xln, xg = ts[jx:], la[jx:], ln[jx:], g0[jx:]
    o['first_x'] = mvt - xt[0]
    o['push_x'] = np.nan
    still = xg < 1
    if len(xg) > 2 and still[0]:
        k = int(np.argmax(~still)) if (~still).any() else len(xg)
        if k < len(xg) - 1 and xt[k - 1] - xt[0] >= 60 and xg[k + 1] >= 1:
            o['push_x'] = mvt - xt[k]
    o['sleave'] = np.nan
    o['sdist_first_x'] = np.nan
    if sll:
        ds = hav(xla, xln, sll[0], sll[1])
        o['sdist_first_x'] = float(ds[0])
        m = (ds <= 100) & (xg < 1)
        if m.any():
            q = int(np.flatnonzero(m)[-1])
            if q < len(xt) - 1:
                o['sleave'] = mvt - xt[q + 1]
    # --- 3) "movement from standstill" segment after arrival (class A / signs hours ahead): ground points after the last airborne point in the 18 h window
    ia = np.flatnonzero(al > GROUND)
    s0 = ia[-1] + 1 if len(ia) else 0
    o['arr'] = mvt - ts[ia[-1]] if len(ia) else np.nan
    gm = (al == GROUND) | ((al == MISSING) & (g0 < 40))
    sel = np.arange(s0, len(ts))[gm[s0:]]
    o['ep_n'] = 0
    o['ep_early_n'] = 0
    o['ep_first'] = np.nan
    o['ep_early_last'] = np.nan
    o['ep_list'] = ''
    o['ep_early_d'] = np.nan
    if len(sel) >= 2:
        tg, gg, lag, lng = ts[sel], g0[sel], la[sel], ln[sel]
        o['gwin_first'] = mvt - tg[0]
        mov = gg >= 3
        dtg = np.r_[0.0, np.diff(tg)]
        new = mov & (~np.r_[False, mov[:-1]] | (dtg > 120))
        starts = np.flatnonzero(new)
        eps = []
        prev_end = tg[0]
        for r_, s_ in enumerate(starts):
            e_ = s_
            while e_ + 1 < len(tg) and mov[e_ + 1] and dtg[e_ + 1] <= 120:
                e_ += 1
            disp = float(hav(lag[s_], lng[s_], lag[e_], lng[e_]))
            still_before = tg[s_] - prev_end
            if e_ > s_ and disp >= 30 and still_before >= 300:
                d_seg = float(hav(lag[s_], lng[s_], sla[0], sln[0]))  # distance from the start of this segment to the first point of the current ground segment (did it return to the same place?)
                eps.append((mvt - tg[s_], still_before, disp, d_seg))
            prev_end = tg[e_]
        if eps:
            obs = np.array([x[0] for x in eps])
            o['ep_n'] = len(eps)
            o['ep_first'] = float(obs.max())
            early = obs[obs > 10800]
            o['ep_early_n'] = int(len(early))
            if len(early):
                o['ep_early_last'] = float(early.min())
                o['ep_early_d'] = float([x[3] for x in eps if x[0] == early.min()][0])
            o['ep_list'] = ';'.join(f'{x[0]:.0f}/{x[1]:.0f}/{x[2]:.0f}/{x[3]:.0f}' for x in eps[-8:])
    return o


def track_frame(df, stab):
    """df: MVT_ID_mvt, date, ADEP_mvt, STAND_mvt, MVT_TIME_UTC_mvt, ab_hex, ab_last_ground (already filtered to non-null). Recomputed day by day."""
    df = df.with_columns((pl.col('MVT_TIME_UTC_mvt').dt.epoch('ms') / 1000).alias('mvt'), pl.col('ab_hex').cast(pl.String),
                         pl.col('ADEP_mvt').replace_strict(NAMES, list(range(len(NAMES))), default=None,
                                                           return_dtype=pl.Int8).alias('apt_i'))
    out = []
    t0 = time.time()
    for n_, date in enumerate(sorted(df['date'].unique().to_list())):
        rows = df.filter(pl.col('date') == date)
        p = load_pts(date, rows['ab_hex'].unique())
        if p is None:
            continue
        groups = p.partition_by(['hex', 'apt'], as_dict=True)
        arrs = {}
        for r in rows.iter_rows(named=True):
            key = (r['ab_hex'], r['apt_i'])
            if key not in groups:
                out.append(dict(MVT_ID_mvt=r['MVT_ID_mvt'], tk_ok=0))
                continue
            if key not in arrs:
                g = groups[key]
                arrs[key] = tuple(g[c].to_numpy() for c in ('t', 'lat', 'lon', 'gs', 'alt', 'trk'))
            an = r['ADEP_mvt']
            sll = stab[0].get(an, {}).get(r['STAND_mvt']) or stab[1].get(an, {}).get(BF.canon(r['STAND_mvt']))
            o = row_track(*arrs[key], r['mvt'] - r['ab_last_ground'], r['mvt'], sll)
            o['MVT_ID_mvt'] = r['MVT_ID_mvt']
            out.append(o)
        if n_ % 10 == 0:
            print('track', date, rows.height, f'{time.time() - t0:.0f}s', flush=True)
    return pl.DataFrame(out, infer_schema_length=None)


# ---------------------------------------------------------------- rows
def klass(df):
    B = ((pl.col('flt_null') == 1) & (pl.col('mvt_delay') > 3600)).fill_null(False)
    C = ((pl.col('flt_null') == 0) & (pl.col('aobt_sched') > 1200) & (pl.col('mvt_delay') > 1200)).fill_null(False)
    S = (pl.col('synth') == 1).fill_null(False)
    return df.with_columns(pl.when(B).then(pl.lit('B')).when(C).then(pl.lit('C')).when(S).then(pl.lit('S'))
                           .otherwise(pl.lit('D')).alias('cls'))


def cmd_rows(a):
    t0 = time.time()
    v = T.load_train('n2', FE)
    oof = np.load(OOF)
    assert len(oof) == v.height, (len(oof), v.height)
    add = [pl.Series('oof', oof), pl.Series('fold', folds(v['date'].to_list())), pl.lit(2025).alias('yr')]
    if Path(OOF8).exists():
        o8 = np.load(OOF8)
        assert len(o8) == v.height
        add.append(pl.Series('oof8', o8))
    v = v.with_columns(add).with_row_index('rid')
    print('load25', v.height, f'{time.time() - t0:.0f}s', flush=True)
    keep = ['rid', 'yr', 'MVT_ID_mvt', 'date', 'ADEP_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'base', 'oof'] + \
           (['oof8'] if 'oof8' in v.columns else []) + \
           ['fold', 'gate', 'isA', 'flt_null', 'mvt_delay', 'mvt_aobt3', 'synth', 'al', 'hour', 'dep10', 'arr15'] + \
           [c for c in T.AB if c in v.columns] + ['ab_hex']
    v = v.select(keep)
    full = pl.read_parquet('n2/feat.parquet', columns=['MVT_ID_mvt', 'ADEP_mvt', 'STAND_mvt', 'src', 'MVT_TIME_UTC_mvt',
                                                       'flt_null', 'mvt_delay', 'mvt_aobt3', 'synth', 'aobt_sched', 'al', 'hour',
                                                       'dep10', 'arr15'])
    v = v.join(full.select('MVT_ID_mvt', 'aobt_sched').unique('MVT_ID_mvt'), on='MVT_ID_mvt', how='left').sort('rid')
    # 2026 (no labels)
    d26 = full.filter(pl.col('src') == 99).with_columns(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d').alias('date'))
    ab26 = pl.concat([pl.read_parquet(f) for f in sorted(Path(FE).glob('2026-*.parquet'))], how='diagonal_relaxed').unique('MVT_ID_mvt')
    if T.STAND != 'off':
        st = pl.concat([pl.read_parquet(f, columns=['MVT_ID_mvt', 'ab_dist_stand']) for f in sorted(Path(T.STAND_DIR).glob('2026-*.parquet'))]).unique('MVT_ID_mvt')
        ab26 = ab26.drop('ab_dist_stand').join(st, on='MVT_ID_mvt', how='left')
    r26 = d26.join(ab26.drop('date'), on='MVT_ID_mvt', how='left').with_columns(
        pl.lit(2026).alias('yr'), pl.col('ab_obs_first').is_not_null().alias('gate'), pl.lit(False).alias('isA'),
        pl.lit(None, pl.Float64).alias('TAXITIME_SEC_mvt'))
    r26 = r26.with_row_index('rid', offset=10_000_000).with_columns(pl.col('rid').cast(v['rid'].dtype))
    rows = pl.concat([v, r26.select([c for c in v.columns if c in r26.columns])], how='diagonal_relaxed')
    rows = klass(rows)
    # Anchoring columns (feats_tma: segment start anchored to the stationary period at the stand)
    tma = pl.concat([pl.read_parquet(f, columns=['MVT_ID_mvt', 'ab_anchor', 'ab_obs_first', 'ab_obs_push'])
                     for f in sorted(Path(TMA).glob('20*.parquet'))], how='vertical_relaxed').unique('MVT_ID_mvt').rename(
        {'ab_obs_first': 'tma_obs_first', 'ab_obs_push': 'tma_obs_push'})
    rows = rows.join(tma, on='MVT_ID_mvt', how='left')
    print('rows', rows.height, f'{time.time() - t0:.0f}s', flush=True)
    stab = BF.load_stand_tables(STANDS, str(Path(STANDS).with_name('xplane_runways.csv')))
    sub = rows.filter(pl.col('ab_hex').is_not_null() & pl.col('ab_last_ground').is_not_null()).select(
        'MVT_ID_mvt', 'date', 'ADEP_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'ab_hex', 'ab_last_ground')
    if a.days:
        keepd = sorted(sub['date'].unique().to_list())
        keepd = [d for d in keepd if d[:4] == '2025'][:a.days] + [d for d in keepd if d[:4] == '2026'][:a.days]
        sub = sub.filter(pl.col('date').is_in(keepd))
    tk = track_frame(sub, stab)
    rows = rows.join(tk.rename({c: f'tk_{c}' for c in tk.columns if c not in ('MVT_ID_mvt', 'tk_ok')}), on='MVT_ID_mvt',
                     how='left').sort('rid')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(a.out)
    chk = rows.filter(pl.col('tk_ok') == 1)
    print('done', rows.height, 'track_ok', chk.height, 'first_mismatch>1s',
          int(((chk['tk_first'] - chk['ab_obs_first']).abs() > 1).sum()), f'{time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- report helpers
EST = [  # (name, obs column, offset method)
    ('first', 'ab_obs_first', 'apt'), ('move3', 'ab_obs_move', 'apt'), ('gs1', 'tk_gs1', 'apt'), ('gs5', 'tk_gs5', 'apt'),
    ('gs8', 'tk_gs8', 'apt'), ('disp30', 'tk_disp30', 'apt'), ('push', 'ab_obs_push', 'apt'), ('rev', 'tk_rev', 'apt'),
    ('turn', 'tk_turn', 'apt'), ('pb', 'pb_obs', 'apt'), ('first_x', 'tk_first_x', 'apt'), ('push_x', 'tk_push_x', 'apt'),
    ('sleave', 'tk_sleave', 'apt'), ('anch_first', 'tma_obs_first', 'apt'), ('first_cell', 'ab_obs_first', 'cell'),
    ('push_stand', 'ab_obs_push', 'stand'), ('best_rule', 'rule_obs', 'apt'),
]


def fold_offset(key, obs, y, mask, fold, kind, cell=None, stand=None, m_s=8.0):
    """Strictly fold-wise offset: apt = median of (y - obs) per (airport, month), falling back to the airport and then to all rows; cell/stand = smoothed mean (m=8, target clipped to (-1800, 3600), as in te_targets)."""
    off = np.full(len(y), np.nan)
    gap = y - obs
    for k in range(5):
        tr, te = mask & (fold != k), mask & (fold == k)
        if not te.any():
            continue
        if kind == 'apt':
            dtr = pl.DataFrame({'k': key[tr], 'a': [x[:4] for x in key[tr]], 'g': gap[tr]})
            m1 = dict(dtr.group_by('k').agg(pl.col('g').median()).iter_rows())
            m2 = dict(dtr.group_by('a').agg(pl.col('g').median()).iter_rows())
            gm = float(np.median(gap[tr]))
            off[te] = [m1.get(k_, m2.get(k_[:4], gm)) for k_ in key[te]]
        else:
            kk = cell if kind == 'cell' else stand
            gg = np.where((gap > -1800) & (gap < 3600), gap, np.nan)
            ok = tr & ~np.isnan(gg)
            apt = np.array([x[:4] for x in key])
            dtr = pl.DataFrame({'k': kk[ok], 'a': apt[ok], 'g': gg[ok]})
            am = dict(dtr.group_by('a').agg(pl.col('g').mean()).iter_rows())
            ss = {k_: (s_, n_, a_) for k_, a_, s_, n_ in dtr.group_by('k', 'a').agg(pl.col('g').sum().alias('s'), pl.len().alias('n')).iter_rows()}
            vals = []
            for k_, a_ in zip(kk[te], apt[te]):
                x = ss.get(k_)
                vals.append((x[0] + m_s * am.get(a_, 0.0)) / (x[1] + m_s) if x else am.get(a_, np.nan))
            off[te] = vals
    return off


def day_paired(date, d, mask, base_sse):
    s = pl.DataFrame({'d': date, 'x': np.where(mask, d, 0.0)}).group_by('d').agg(pl.col('x').sum())['x'].to_numpy()
    return dict(pp=round(float(-100 * s.sum() / base_sse), 4), se=round(float(100 * s.std(ddof=1) * np.sqrt(len(s)) / base_sse), 4))


def per_fold(fold, d, mask, base_sse):
    x = [float(-100 * d[mask & (fold == k)].sum() / base_sse) for k in range(5)]
    return dict(per_fold_pp=[round(z, 4) for z in x], folds_positive=int(sum(z > 0 for z in x)))


def qd(x):
    return [round(float(q), 1) for q in np.quantile(x, QS)] if len(x) else None


def cats(r):
    """Classification columns of the gated rows (label-based classes are used for diagnostics only)."""
    y, of = r['y'], r['ab_obs_first']
    cs = r['ab_cs_match']
    ds = r['ab_dist_stand']
    c = {}
    c['cls'] = r['cls']
    c['match'] = np.char.add(np.where(np.isin(cs, [1, 3]), 'cs', 'time'), np.where(np.isin(cs, [2, 3]), '_climb', '_ground'))
    c['at_stand'] = np.where(np.isnan(ds), 'na', np.where(ds <= 150, 'near', 'far'))
    c['dwell'] = np.where(r['ab_dwell_seen'] == 1, 'push_seen', 'no_push')
    c['pb_seen'] = np.where(~np.isnan(r['pb_obs']), 'pb', 'nopb')
    lg = y - of
    c['lab_vs_ground'] = np.where(lg < -300, 'short>5m', np.where(lg > 300, 'long>5m', 'within5m'))
    c['anchor'] = np.where(r['ab_anchor'] == 1, 'anchored', 'no')
    late = (r['ab_gs_first'] >= 3) & ~(ds <= 150)
    c['late_xpdr'] = np.where(late, 'first_seen_moving_away', 'ok')
    c['stop'] = r['tk_stop']
    c['apt'] = r['apt']
    c['sign'] = np.where(r['p20'] > y, 'over', 'under')
    bs = (r['mvt_delay'] > 1200) & (np.abs(y - r['mvt_delay']) <= 60)
    et = np.where(bs, 'block_sched', np.where(lg < -300, 'short>5m', np.where(lg > 1800, 'blind>30m',
                  np.where(lg > 300, 'blind5-30m', 'other'))))
    c['etype'] = et
    return c


def breakdown(cat, se, sel, tot_sel):
    """Per class within sel: share of rows and share of SSE; also the share within tot_sel (all gated rows), to show enrichment."""
    out = {}
    s_sel, s_tot = se[sel].sum(), se[tot_sel].sum()
    for v_ in sorted(set(cat[tot_sel].tolist())):
        m = cat == v_
        out[str(v_)] = dict(rows=int((m & sel).sum()), sse_sh=round(float(se[m & sel].sum() / s_sel), 4),
                            rows_sh_all=round(float((m & tot_sel).sum() / tot_sel.sum()), 4),
                            sse_sh_all=round(float(se[m & tot_sel].sum() / s_tot), 4))
    return out


def load_report_frame(path):
    R = pl.read_parquet(path)
    r25 = R.filter(pl.col('yr') == 2025).sort('rid')
    r26 = R.filter(pl.col('yr') == 2026)
    return R, r25, r26


def arrays(r25):
    f = lambda c: r25[c].cast(pl.Float64).fill_nan(None).to_numpy().astype(float) if c in r25.columns else np.full(r25.height, np.nan)
    r = {c: f(c) for c in r25.columns if r25[c].dtype.is_numeric() and c not in ('gate', 'isA')}
    r['y'] = f('TAXITIME_SEC_mvt')
    r['gate'] = r25['gate'].to_numpy().astype(bool)
    r['isA'] = r25['isA'].to_numpy().astype(bool)
    r['fold'] = r25['fold'].to_numpy().astype(int)
    r['apt'] = r25['ADEP_mvt'].to_numpy()
    r['mo'] = r25['date'].str.slice(5, 2).to_numpy()
    r['date'] = r25['date'].to_numpy()
    r['cls'] = r25['cls'].to_numpy()
    r['tk_stop'] = r25['tk_stop'].fill_null('none').to_numpy() if 'tk_stop' in r25.columns else np.full(r25.height, 'none')
    r['p20'] = r['base'] + np.nan_to_num(r['oof'])
    r['p8s'] = r['base'] + np.nan_to_num(r['oof8']) if 'oof8' in r25.columns else None
    pb = np.fmax(r['tk_rev'], r['tk_turn'])  # take the earlier time (larger obs); fmax: if one side is NaN take the other
    r['pb_obs'] = np.where(np.isnan(r['ab_obs_push']), pb, r['ab_obs_push'])
    # Combination rule (label-free): push-back after standing still at the segment start -> push; else push-back by reversing/turning -> pb; else the segment start
    r['rule_obs'] = np.where(~np.isnan(r['ab_obs_push']), r['ab_obs_push'], np.where(~np.isnan(pb), pb, r['ab_obs_first']))
    r['key'] = np.array([f'{a_}-{m_}' for a_, m_ in zip(r['apt'], r['mo'])])
    r['cell'] = r25.select(pl.concat_str([pl.col('ADEP_mvt'), (pl.col('ab_first_lat') / 0.0015).round().cast(pl.Int64).cast(pl.String),
                                          (pl.col('ab_first_lon') / 0.002).round().cast(pl.Int64).cast(pl.String)],
                                         separator='|'))[:, 0].to_numpy()
    r['stand'] = r25.select(pl.concat_str([pl.col('ADEP_mvt'), pl.col('STAND_mvt')], separator='|'))[:, 0].to_numpy()
    return r


# ---------------------------------------------------------------- report
def cmd_report(a):
    R, r25, r26 = load_report_frame(a.rows)
    r = arrays(r25)
    y, b, p20, G0, nA, fold = r['y'], r['base'], r['p20'], r['gate'], ~r['isA'], r['fold']
    EX = np.isin(r['key'], [x for x in a.excl.split(',') if x])  # --excl: cells removed from the gated set (e.g. LIRF-07: no 2026 coverage)
    G = G0 & nA & ~EX
    e0, e20 = (b - y) ** 2, (p20 - y) ** 2
    base_sse = float(e0[nA].sum())
    out = dict(rows25=int(r25.height), rows26=int(r26.height), base_nonA_sse=base_sse, excl=a.excl, n_excl_gated=int((G0 & nA & EX).sum()),
               check=dict(total_red=round(float(1 - e20[nA].sum() / base_sse), 5),
                          gated_red=round(float(1 - e20[G].sum() / e0[G].sum()), 5),
                          gated_sse_share_refcfg=round(float(e20[G].sum() / e20[nA].sum()), 4),
                          track_ok=int(np.nansum(r['tk_ok'] == 1)),
                          first_mismatch_gt1s=int(np.nansum(np.abs(r['tk_first'] - r['ab_obs_first']) > 1))))
    print('check', out['check'], flush=True)
    # (1) airport x month f
    t1 = {}
    for key in sorted(set(r['key'][G])) + ['ALL-01', 'ALL-07', 'ALL']:
        if key == 'ALL':
            m = G
        elif key.startswith('ALL-'):
            m = G & (r['mo'] == key[4:])
        else:
            m = G & (r['key'] == key)
        mk = nA & ((r['key'] == key) if not key.startswith('ALL') else (np.ones_like(nA) if key == 'ALL' else r['mo'] == key[4:]))
        row = dict(n_nonA=int(mk.sum()), n_gated=int(m.sum()), gate=round(float(m.sum() / max(mk.sum(), 1)), 3),
                   rmse_ensavg=round(float(np.sqrt(e0[m].mean())), 1), rmse_refcfg=round(float(np.sqrt(e20[m].mean())), 1),
                   f=round(float(e20[m].sum() / e0[m].sum()), 3), gated_sse_pp_of_base=round(float(100 * e20[m].sum() / base_sse), 3))
        if r['p8s'] is not None:
            row['f_s8'] = round(float(((r['p8s'] - y) ** 2)[m].sum() / e0[m].sum()), 3)
        t1[key] = row
    out['t1_f'] = t1
    # (2) direct-measurement estimates
    t2 = {}
    est_pred = {}
    for name, col, kind in EST:
        obs = r[col] if col in r else np.full(len(y), np.nan)
        D = G & ~np.isnan(obs)
        if D.sum() < 500:
            t2[name] = dict(n=int(D.sum()))
            continue
        off = fold_offset(r['key'], obs, y, D, fold, kind, cell=r['cell'], stand=r['stand'])
        px = obs + off
        D = D & ~np.isnan(px)
        est_pred[name] = (px, D)
        ex, ev = px - y, p20 - y
        ex2, ev2 = ex ** 2, ev ** 2
        better = D & (np.abs(ev) - np.abs(ex) >= 120) & (np.abs(ex) <= 0.5 * np.abs(ev))
        worse = D & (np.abs(ex) - np.abs(ev) >= 120) & (np.abs(ev) <= 0.5 * np.abs(ex))
        # Per-fold, per-airport least-squares blend weight w (refcfg + w*(X - refcfg)); the weight is determined by the training folds only
        pb_ = p20.copy()
        wlog = {}
        for k in range(5):
            tr, te = D & (fold != k), D & (fold == k)
            for ap in APTS:
                ta, ea = tr & (r['apt'] == ap), te & (r['apt'] == ap)
                if ta.sum() < 200 or not ea.any():
                    continue
                dx = px[ta] - p20[ta]
                w = float(np.clip(np.dot(dx, y[ta] - p20[ta]) / max(np.dot(dx, dx), 1e-9), 0, 1))
                pb_[ea] = p20[ea] + w * (px[ea] - p20[ea])
                wlog.setdefault(ap, []).append(round(w, 3))
        dblend = (pb_ - y) ** 2 - e20
        per = {}
        for key in sorted(set(r['key'][D])):
            m = D & (r['key'] == key)
            if m.sum() >= 200:
                per[key] = dict(n=int(m.sum()), rmse_x=round(float(np.sqrt(ex2[m].mean())), 1),
                                rmse_refcfg=round(float(np.sqrt(ev2[m].mean())), 1), mad_x=round(float(np.median(np.abs(ex[m]))), 1),
                                mad_refcfg=round(float(np.median(np.abs(ev[m]))), 1))
        t2[name] = dict(
            n=int(D.sum()), cov=round(float(D.sum() / G.sum()), 4), rmse_x=round(float(np.sqrt(ex2[D].mean())), 1),
            rmse_refcfg=round(float(np.sqrt(ev2[D].mean())), 1), rmse_ensavg=round(float(np.sqrt(e0[D].mean())), 1),
            mad_x=round(float(np.median(np.abs(ex[D]))), 1), mad_refcfg=round(float(np.median(np.abs(ev[D]))), 1),
            q_x=qd(ex[D]), q_refcfg=qd(ev[D]),
            better=dict(rows=int(better.sum()), rows_sh=round(float(better.sum() / D.sum()), 4),
                        refcfg_sse_sh_gated=round(float(ev2[better].sum() / e20[G].sum()), 4),
                        refcfg_sse_pp=round(float(100 * ev2[better].sum() / base_sse), 3),
                        x_sse_pp=round(float(100 * ex2[better].sum() / base_sse), 3)),
            worse=dict(rows=int(worse.sum()), refcfg_sse_pp=round(float(100 * ev2[worse].sum() / base_sse), 3),
                       x_sse_pp=round(float(100 * ex2[worse].sum() / base_sse), 3)),
            oracle_min_pp=round(float(100 * (ev2[D] - np.minimum(ex2[D], ev2[D])).sum() / base_sse), 3),
            replace_all_pp=round(float(100 * (ev2[D] - ex2[D]).sum() / base_sse), 3),
            blend=dict(day_paired(r['date'], dblend, nA, base_sse), **per_fold(fold, dblend, nA, base_sse), w=wlog,
                       per_apt={ap: day_paired(r['date'], dblend, nA & (r['apt'] == ap), base_sse) for ap in APTS
                                if (D & (r['apt'] == ap)).any()}),
            per_cell=per)
        print('est', name, {k: t2[name][k] for k in ('n', 'cov', 'rmse_x', 'rmse_refcfg', 'mad_x', 'mad_refcfg', 'oracle_min_pp')},
              'blend', t2[name]['blend']['pp'], t2[name]['blend']['se'], flush=True)
    out['t2_direct'] = t2
    # Good-observation subset: push-back after standing still is seen + call-sign match + segment start within 150 m of the stand -- is refcfg already close to the measurement floor?
    sub = {}
    for nm, m in (('push_cs_near', G & (r['ab_dwell_seen'] == 1) & np.isin(r['ab_cs_match'], [1, 3]) & (r['ab_dist_stand'] <= 150)),
                  ('push_any', G & (r['ab_dwell_seen'] == 1)), ('no_push', G & (r['ab_dwell_seen'] != 1)),
                  ('pb_any', G & ~np.isnan(r['pb_obs'])), ('nopb', G & np.isnan(r['pb_obs']))):
        if m.sum() < 200:
            continue
        o = dict(n=int(m.sum()), rows_sh=round(float(m.sum() / G.sum()), 4), sse_sh=round(float(e20[m].sum() / e20[G].sum()), 4),
                 rmse_refcfg=round(float(np.sqrt(e20[m].mean())), 1), rmse_ensavg=round(float(np.sqrt(e0[m].mean())), 1),
                 mad_refcfg=round(float(np.median(np.abs(p20[m] - y[m]))), 1))
        for en in ('push', 'pb', 'first_cell', 'best_rule'):
            if en in est_pred:
                px, D = est_pred[en]
                mm = m & D
                if mm.sum() >= 200:
                    o[f'rmse_{en}'] = round(float(np.sqrt(((px - y) ** 2)[mm].mean())), 1)
                    o[f'rmse_refcfg_on_{en}'] = round(float(np.sqrt(e20[mm].mean())), 1)
                    o[f'mad_{en}'] = round(float(np.median(np.abs(px[mm] - y[mm]))), 1)
        per = {}
        for ap in APTS:
            ma = m & (r['apt'] == ap)
            if ma.sum() >= 200:
                per[ap] = dict(n=int(ma.sum()), rmse_refcfg=round(float(np.sqrt(e20[ma].mean())), 1),
                               mad_refcfg=round(float(np.median(np.abs(p20[ma] - y[ma]))), 1),
                               iqr_y_minus_obs_first=[round(float(q), 1) for q in np.quantile((y - r['ab_obs_first'])[ma], [.25, .5, .75])])
                if 'push' in est_pred:
                    px, D = est_pred['push']
                    mm = ma & D
                    if mm.sum() >= 100:
                        per[ap]['rmse_push'] = round(float(np.sqrt(((px - y) ** 2)[mm].mean())), 1)
                        per[ap]['iqr_y_minus_obs_push'] = [round(float(q), 1) for q in np.quantile((y - r['ab_obs_push'])[mm], [.25, .5, .75])]
        o['per_apt'] = per
        sub[nm] = o
    out['t2b_subsets'] = sub
    out['t2c_detector'] = {ap: dict(n=int((G & (r['apt'] == ap)).sum()),
                                    trk_agree_med=round(float(np.nanmedian(r['tk_trk_agree'][G & (r['apt'] == ap)])), 3)
                                    if np.isfinite(r['tk_trk_agree'][G & (r['apt'] == ap)]).any() else None,
                                    push_rate=round(float((~np.isnan(r['ab_obs_push'][G & (r['apt'] == ap)])).mean()), 3),
                                    rev_rate=round(float((~np.isnan(r['tk_rev'][G & (r['apt'] == ap)])).mean()), 3),
                                    turn_rate=round(float((~np.isnan(r['tk_turn'][G & (r['apt'] == ap)])).mean()), 3),
                                    pb_rate=round(float((~np.isnan(r['pb_obs'][G & (r['apt'] == ap)])).mean()), 3),
                                    stop_share={s_: round(float((r['tk_stop'][G & (r['apt'] == ap)] == s_).mean()), 3)
                                                for s_ in ('gap', 'max3h', 'air', 'miss', 'start', 'none')})
                            for ap in APTS if (G & (r['apt'] == ap)).sum() >= 300}
    # (2d) Is there still airline / stand / time-of-day level systematic structure in the refcfg residuals of gated rows (strictly fold-wise, smoothed mean shrunk towards 0, m=20; no training)
    alv = r25['al'].cast(pl.String).fill_null('NA').to_numpy()
    hrs = np.nan_to_num(r['hour'], nan=-1).astype(int).astype(str)
    res20 = y - p20
    rawres = p20 - b

    def te_corr(keys, mask, tgt, m_s=20.0):
        corr = np.zeros(len(y))
        for k in range(5):
            tr, te = mask & (fold != k), mask & (fold == k)
            df = pl.DataFrame({'k': keys[tr], 'r': tgt[tr]}).group_by('k').agg(pl.col('r').sum().alias('s'), pl.len().alias('n'))
            mp = {k_: s_ / (n_ + m_s) for k_, s_, n_ in df.iter_rows()}
            corr[te] = [mp.get(k_, 0.0) for k_ in keys[te]]
        return corr
    ks = {'apt_al': np.char.add(np.char.add(r['apt'].astype(str), '|'), alv.astype(str)),
          'apt_stand': r['stand'].astype(str), 'apt_hour': np.char.add(np.char.add(r['apt'].astype(str), '|'), hrs),
          'cell': r['cell'].astype(str),
          'apt_al_push': np.char.add(np.char.add(np.char.add(r['apt'].astype(str), '|'), alv.astype(str)),
                                     np.where(r['ab_dwell_seen'] == 1, '|p', '|n')),
          'apt_mo_CONTROL': r['key'].astype(str)}
    t2d = {}
    for nm, kk in ks.items():
        corr = te_corr(kk, G, res20)
        d = (p20 + np.where(G, corr, 0.0) - y) ** 2 - e20
        t2d[nm] = dict(day_paired(r['date'], d, nA, base_sse), **per_fold(fold, d, nA, base_sse), n_keys=int(len(set(kk[G].tolist()))))
    # Residual scaling lambda (per-fold, per-airport least squares): is the refcfg residual layer shrunk too much?
    pl_ = p20.copy()
    lam = {}
    for k in range(5):
        for ap in APTS:
            tr, te = G & (fold != k) & (r['apt'] == ap), G0 & ~EX & (fold == k) & (r['apt'] == ap)
            if tr.sum() < 500 or not te.any():
                continue
            l_ = float(np.dot(rawres[tr], (y - b)[tr]) / max(np.dot(rawres[tr], rawres[tr]), 1e-9))
            pl_[te] = b[te] + l_ * rawres[te]
            lam.setdefault(ap, []).append(round(l_, 3))
    d = (pl_ - y) ** 2 - e20
    t2d['lambda_scale'] = dict(day_paired(r['date'], d, nA, base_sse), **per_fold(fold, d, nA, base_sse), lam=lam,
                               per_apt={ap: day_paired(r['date'], d, nA & (r['apt'] == ap), base_sse) for ap in APTS
                                        if (G & (r['apt'] == ap)).any()})
    # How much of the label minus physical push-back offset (y - obs_push, clipped to (-1800, 3600)) is explained by airline / stand (fold-wise smoothed mean, prior = airport x month median)
    push = G & ~np.isnan(r['ab_obs_push'])
    gp = y - r['ab_obs_push']
    okp = push & (gp > -1800) & (gp < 3600)
    prior = fold_offset(r['key'], r['ab_obs_push'], y, okp, fold, 'apt')
    offv = {}
    for nm in ('apt_al', 'apt_stand'):
        adj = te_corr(ks[nm], okp, np.where(okp, gp - prior, 0.0), m_s=8.0)
        per = {}
        for ap in APTS:
            ma = okp & (r['apt'] == ap)
            if ma.sum() >= 300:
                per[ap] = dict(n=int(ma.sum()), rmse_prior=round(float(np.sqrt(((gp - prior)[ma] ** 2).mean())), 1),
                               rmse_te=round(float(np.sqrt(((gp - prior - adj)[ma] ** 2).mean())), 1),
                               iqr_prior=[round(float(q), 1) for q in np.quantile((gp - prior)[ma], [.25, .75])])
        offv[nm] = per
    t2d['push_offset_explained'] = offv
    out['t2d_resid_structure'] = t2d
    print('t2d', {k_: (v_['pp'], v_['se']) for k_, v_ in t2d.items() if 'pp' in v_}, flush=True)
    # (3) sources of the head errors
    c = cats(r)
    order = np.argsort(-np.where(G, e20, -1.0))
    nG = int(G.sum())
    t3 = {}
    for q in (0.01, 0.05, 0.20):
        top = np.zeros(len(y), bool)
        top[order[:int(round(q * nG))]] = True
        top &= G
        o = dict(rows=int(top.sum()), sse_sh_gated=round(float(e20[top].sum() / e20[G].sum()), 4),
                 sse_pp_of_base=round(float(100 * e20[top].sum() / base_sse), 3),
                 sse_sh_nonA_refcfg=round(float(e20[top].sum() / e20[nA].sum()), 4),
                 abs_err_min=round(float(np.abs(p20 - y)[top].min()), 1))
        for cn, cv in c.items():
            o[cn] = breakdown(cv, e20, top, G)
        for en in ('push', 'pb', 'first_cell', 'best_rule', 'first_x', 'sleave', 'anch_first'):
            if en in est_pred:
                px, D = est_pred[en]
                mm = top & D
                o[f'fixable_by_{en}_pp'] = round(float(100 * np.maximum(e20[mm] - (px[mm] - y[mm]) ** 2, 0).sum() / base_sse), 3)
        t3[f'top{int(q * 100)}'] = o
    out['t3_top'] = t3
    # Class means over all gated rows (refcfg MSE per row)
    out['t3_cat_mse'] = {cn: {str(v_): dict(n=int((G & (cv == v_)).sum()), rmse_refcfg=round(float(np.sqrt(e20[G & (cv == v_)].mean())), 1),
                                             rmse_ensavg=round(float(np.sqrt(e0[G & (cv == v_)].mean())), 1))
                              for v_ in sorted(set(cv[G].tolist())) if (G & (cv == v_)).sum() > 0}
                         for cn, cv in c.items() if cn not in ('sign',)}
    # (4) class A
    A = r['isA']
    t4 = dict(n=int(A.sum()), sse_refcfg=float(e20[A].sum()), sse_ensavg=float(e0[A].sum()),
              mse_contrib_62d=round(float(e20[A].sum() / len(y)), 1))
    byk = {}
    for key in sorted(set(r['key'][A])):
        m = A & (r['key'] == key)
        byk[key] = dict(n=int(m.sum()), gated=int((m & G0).sum()), matched=int((m & ~np.isnan(r['tk_ok'])).sum()),
                        sse_sh=round(float(e20[m].sum() / e20[A].sum()), 4), y_med=round(float(np.median(y[m])), 0))
    t4['by_cell'] = byk
    lst = []
    idxA = np.flatnonzero(A & ~np.isnan(r['tk_ok']))
    for i in idxA:
        eps = r25['tk_ep_list'][int(i)] if 'tk_ep_list' in r25.columns else ''
        vis = None
        if eps:
            obs = np.array([float(x.split('/')[0]) for x in eps.split(';')])
            vis = round(float(np.min(np.abs(obs - y[i]))), 0)
        lst.append(dict(key=r['key'][i], date=r['date'][i], cls=r['cls'][i], y=round(float(y[i])), p20=round(float(p20[i])),
                        mvt_delay=None if np.isnan(r['mvt_delay'][i]) else round(float(r['mvt_delay'][i])),
                        obs_first=None if np.isnan(r['ab_obs_first'][i]) else round(float(r['ab_obs_first'][i])),
                        stop=r['tk_stop'][i], arr=None if np.isnan(r['tk_arr'][i]) else round(float(r['tk_arr'][i])),
                        ep_first=None if np.isnan(r['tk_ep_first'][i]) else round(float(r['tk_ep_first'][i])),
                        ep_early_last=None if np.isnan(r['tk_ep_early_last'][i]) else round(float(r['tk_ep_early_last'][i])),
                        ep_list=eps, block_vs_nearest_ep_s=vis))
    t4['matched_rows'] = lst
    t4['block_visible_600s'] = int(sum(1 for x in lst if x['block_vs_nearest_ep_s'] is not None and x['block_vs_nearest_ep_s'] <= 600))
    # Signature S: a "movement from standstill" appears > 3 h before take-off, after arrival (label-free)
    S = (np.nan_to_num(r['tk_ep_early_n']) > 0)
    sig = dict(n_S=int(S.sum()), n_S_A=int((S & A).sum()), n_A_matched=int((A & ~np.isnan(r['tk_ok'])).sum()),
               precision=round(float((S & A).sum() / max(S.sum(), 1)), 4))
    S_ret = S & (r['tk_ep_early_d'] <= 150)
    sig['n_S_ret'] = int(S_ret.sum())
    sig['n_S_ret_A'] = int((S_ret & A).sum())
    for nm, obs, SS in (('early_last', r['tk_ep_early_last'], S), ('ep_first', r['tk_ep_first'], S),
                        ('early_last_ret', r['tk_ep_early_last'], S_ret)):
        m = SS & ~np.isnan(obs)
        d_all = (obs - y) ** 2 - e20
        sig[f'rule_{nm}'] = dict(rows=int(m.sum()), d_sse_A=float(d_all[m & A].sum()), d_sse_nonA=float(d_all[m & nA].sum()),
                                 d_pp_nonA=round(float(100 * d_all[m & nA].sum() / base_sse), 3),
                                 d_sse_A_share_of_A=round(float(d_all[m & A].sum() / max(e20[A].sum(), 1)), 4))
    y_ = y
    sig['y_dist_S_nonA'] = qd(y_[S & nA])
    sig['by_apt'] = {ap: dict(S=int((S & (r['apt'] == ap)).sum()), S_A=int((S & A & (r['apt'] == ap)).sum()),
                              A=int((A & (r['apt'] == ap)).sum())) for ap in APTS}
    s26 = r26.filter(pl.col('tk_ep_early_n').fill_null(0) > 0) if 'tk_ep_early_n' in r26.columns else r26.head(0)
    sig['n_S_2026_by_cell'] = {f'{a_}-{m_}': int(n_) for a_, m_, n_ in
                               s26.group_by(pl.col('ADEP_mvt'), pl.col('date').str.slice(5, 2).alias('m')).len().iter_rows()} if s26.height else {}
    n26 = r26.group_by(pl.col('ADEP_mvt'), pl.col('date').str.slice(5, 2).alias('m')).len()
    sig['n_2026_by_cell'] = {f'{a_}-{m_}': int(n_) for a_, m_, n_ in n26.iter_rows()}
    t4['signature'] = sig
    out['t4_A'] = t4
    # (5) 2026 weights: n26 x gating rate x 2025 refcfg MSE of gated rows (cells with too few samples fall back to the airport, then to all rows); likewise for non-gated rows
    g26 = r26.group_by(pl.col('ADEP_mvt').alias('a'), pl.col('date').str.slice(5, 2).alias('m')).agg(
        pl.len().alias('n'), pl.col('gate').mean().alias('g'))
    mse_g, mse_ng = {}, {}
    for key in set(r['key']):
        m = nA & (r['key'] == key)
        if (m & G0).sum() >= 300:
            mse_g[key] = float(e20[m & G0].mean())
        if (m & ~G0).sum() >= 300:
            mse_ng[key] = float(e20[m & ~G0].mean())
    fb_g = {ap: float(e20[G & (r['apt'] == ap)].mean()) for ap in APTS if (G & (r['apt'] == ap)).sum() >= 300}
    fb_ng = {ap: float(e20[nA & ~G0 & (r['apt'] == ap)].mean()) for ap in APTS if (nA & ~G0 & (r['apt'] == ap)).sum() >= 300}
    allg, allng = float(e20[G].mean()), float(e20[nA & ~G0].mean())
    w26 = {}
    for a_, m_, n_, g_ in g26.iter_rows():
        key = f'{a_}-{m_}'
        wg = n_ * g_ * mse_g.get(key, fb_g.get(a_, allg))
        wn = n_ * (1 - g_) * mse_ng.get(key, fb_ng.get(a_, allng))
        w26[key] = dict(n=int(n_), gate=round(float(g_), 3), w_g=wg, w_ng=wn,
                        src_g='cell' if key in mse_g else ('apt' if a_ in fb_g else 'all'))
    tot = sum(x['w_g'] + x['w_ng'] for x in w26.values())
    out['t5_w26'] = dict(cells=w26, gated_share_of_nonA_sse_2026=round(sum(x['w_g'] for x in w26.values()) / tot, 4),
                         note='2025 refcfg CV MSE per row x 2026 row count and gating rate; class B is not recomputed per cell')
    # (6) Observation pattern of gated rows, 2025 vs 2026 (label-free): push-back seen, call-sign match, segment start next to the stand, reason the segment stopped
    mix = R.filter(pl.col('gate')).with_columns(pl.col('date').str.slice(5, 2).alias('m'))
    if 'tk_rev' in mix.columns:
        mix = mix.with_columns(((pl.col('ab_dwell_seen') == 1).fill_null(False) | pl.col('tk_rev').fill_nan(None).is_not_null() |
                                pl.col('tk_turn').fill_nan(None).is_not_null()).alias('pbx'))
    else:
        mix = mix.with_columns((pl.col('ab_dwell_seen') == 1).fill_null(False).alias('pbx'))
    t6 = mix.group_by('yr', 'ADEP_mvt', 'm').agg(
        pl.len().alias('n'), (pl.col('ab_dwell_seen') == 1).mean().round(3).alias('push'), pl.col('pbx').mean().round(3).alias('pb'),
        pl.col('ab_cs_match').is_in([1, 3]).mean().round(3).alias('cs'), (pl.col('ab_dist_stand') <= 150).mean().round(3).alias('near'),
        pl.col('ab_npts').median().alias('npts_med'), (pl.col('tk_stop') == 'gap').mean().round(3).alias('stop_gap'),
        (pl.col('ab_gs_first') >= 3).mean().round(3).alias('first_moving')).sort('ADEP_mvt', 'm', 'yr')
    out['t6_mix'] = t6.to_dicts()
    json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str)
    print('wrote', a.out, flush=True)


# ---------------------------------------------------------------- fusion (training)
FUS_OLD = ['p20', 'base', 'ab_obs_first', 'ab_obs_move', 'ab_obs_push', 'ab_dwell_seen', 'ab_dist_stand', 'ab_gs_first',
           'ab_npts', 'ab_max_gap', 'ab_cs_match', 'ab_etype', 'ab_last_ground', 'ab_dlift', 'ab_path_len', 'ab_stop_after_move',
           'ab_hold_last', 'ab_taxi_now10', 'mvt_delay', 'mvt_aobt3', 'flt_null', 'synth', 'hour', 'apt_code', 'mo_code']
FUS_NEW = ['tk_gs1', 'tk_gs5', 'tk_gs8', 'tk_disp30', 'tk_rev', 'tk_turn', 'tk_trk_agree', 'tk_first_x', 'tk_push_x',
           'tk_sleave', 'tk_sdist_first_x', 'stop_code', 'tk_stop_gap_s', 'tk_stop_gap_m', 'tk_arr', 'tk_ep_n', 'tk_ep_first',
           'ab_anchor', 'tma_obs_first', 'tma_obs_push']


def cmd_fusion(a):
    import lightgbm as lgb
    R, r25, _ = load_report_frame(a.rows)
    r = arrays(r25)
    y, b, p20, G0, nA, fold = r['y'], r['base'], r['p20'], r['gate'], ~r['isA'], r['fold']
    EX = np.isin(r['key'], [x for x in a.excl.split(',') if x])
    G = G0 & nA & ~EX
    G0 = G0 & ~EX
    e20 = (p20 - y) ** 2
    base_sse = float(((b - y) ** 2)[nA].sum())
    r['apt_code'] = np.array([APTS.index(x) for x in r['apt']], dtype=float)
    r['mo_code'] = (r['mo'] == '07').astype(float)
    r['stop_code'] = np.array([['gap', 'max3h', 'air', 'miss', 'start', 'win', 'none'].index(s) for s in r['tk_stop']], dtype=float)
    params = dict(objective='regression', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=int(os.environ.get("RESID_THREADS", "2")))
    out = dict(base_nonA_sse=base_sse, excl=a.excl, gated_sse_refcfg=float(e20[G].sum()))
    for nm, feats in (('ctrl', ['p20', 'base', 'apt_code', 'mo_code']), ('old', FUS_OLD), ('new', FUS_OLD + FUS_NEW)):
        X = np.column_stack([r[c] if c in r else np.full(len(y), np.nan) for c in feats]).astype(np.float32)
        for rounds in (200, 600):
            pf = p20.copy()
            imp = np.zeros(len(feats))
            for k in range(5):
                tr, te = G & (fold != k), G0 & (fold == k)
                m = lgb.train(dict(params, seed=k), lgb.Dataset(X[tr], (y - p20)[tr]), num_boost_round=rounds)
                pf[te] = p20[te] + m.predict(X[te])
                imp += m.feature_importance('gain')
            d = (pf - y) ** 2 - e20
            res = dict(all=dict(day_paired(r['date'], d, nA, base_sse), **per_fold(fold, d, nA, base_sse)),
                       jan=day_paired(r['date'], d, nA & (r['mo'] == '01'), base_sse),
                       jul=day_paired(r['date'], d, nA & (r['mo'] == '07'), base_sse),
                       per_apt={ap: day_paired(r['date'], d, nA & (r['apt'] == ap), base_sse) for ap in APTS
                                if (G & (r['apt'] == ap)).sum() > 0},
                       gated_red_rel=round(float(-d[G].sum() / e20[G].sum()), 4),
                       per_cell={key: dict(n=int((G & (r['key'] == key)).sum()), sse20=float(e20[G & (r['key'] == key)].sum()),
                                           red=round(float(-d[G & (r['key'] == key)].sum() / e20[G & (r['key'] == key)].sum()), 4))
                                 for key in sorted(set(r['key'][G])) if (G & (r['key'] == key)).sum() >= 300},
                       imp_top={feats[i]: round(float(imp[i] / imp.sum()), 4) for i in np.argsort(-imp)[:15]})
            out[f'{nm}_{rounds}'] = res
            print('fusion', nm, rounds, res['all'], res['gated_red_rel'], flush=True)
    json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('wrote', a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['rows', 'report', 'fusion'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', default='gap/track_rows.parquet')
    ap.add_argument('--days', type=int, default=0, help='rows: only compute the trajectories of the first N days per year (for trial runs)')
    ap.add_argument('--excl', default='', help='report/fusion: airport-months removed from the gated set (comma-separated, e.g. LIRF-07)')
    a = ap.parse_args()
    {'rows': cmd_rows, 'report': cmd_report, 'fusion': cmd_fusion}[a.cmd](a)


if __name__ == '__main__':
    main()
