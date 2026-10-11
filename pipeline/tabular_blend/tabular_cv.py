"""tabular_cv: formal CV of the tabular candidate families. Plugs the families that were positive in the feature screening into members with the structure of the averaged baseline, confirmed on two folds.

Method (same basis as the earlier feature-screening scripts):
- Features = FEATSETS['srp'] of the baseline script (feature set of the main member of ensemble B, srp_g5; the SRP list is copied by hand from the baseline's common module);
  new arm = SRP + the candidate columns of the chosen families (pf.parquet) + (when family d is included) residual target encodings (TE) computed per fold
  (residuals of the 12-fold OOF; training rows leave their own month out, validation rows use all 11 training months; same definition as in the screening).
- Two folds: hold out 2025-01 / 2025-07 and train on the other 11 months; the taxi target and the delta target are averaged (delta path = mvt_aobt3 + prediction,
  rows without a match use the taxi path); LightGBM 255 leaves, lr .1 (--lr), ff .7, lambda2 10, min_data 100, max_bin 63, early stopping 50 on the validation fold
  (the same mild optimism as the baseline members, identical for both arms).
- Both arms use the same seed and parameters; B rows are always replaced by the averaged baseline.
- Evaluation: member-level delta-SSE (new - base) and the delta-SSE of "averaged baseline + (new - base)" transplanted onto the averaged baseline, day-paired SE,
  by airport x month x row class; the 2026 projection (ADDR26 weights) follows the screening script.
Usage: python tabular_cv.py <storedir: baseline feature-store dir, n2/> <pf.parquet> <out.json> --fam b,c [--te] [--threads 4]
"""
import argparse
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
CAT0 = ['ADEP_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'ADES_mvt', 'AIRCRAFT_OPERATOR_flt', 'WK_TBL_CAT_flt', 'MARKET_SEGMENT_flt', 'FLIGHT_TYPE_flt']
CAT = set(CAT0 + ['al', 'a_rule_mvt', 'a_rule_flt', 'b_pref', 'c_cfg'])
BASE = ['hour', 'dow', 'mod', 'mvt_delay', 'mvt_aobt3', 'mvt_eobt1', 'aobt_eobt', 'flt_null']
CONG = ['dep10', 'dep30', 'arr15', 'rwy_gap_prev', 'rwy_gap_next', 'rwy_dep10', 'sched30', 'aobt_rank10']
NEW5 = ['overtake', 'overtake_rwy', 'turn_aobt', 'turn_land', 'same_type']
P15 = ['mvt_sec', 'mvt_sec0', 'aobt_sec0', 'arvt3_sec0', 'sched_sec', 'mvt_lobt', 'mvt_iobt', 'lobt_eobt1', 'aobt_lobt', 'arvt1_eobt1', 'arvt3_mvt', 'm1_m3_dur', 'month']
SYN = ['synth', 'ma_dev', 'synth_share', 'arr_taxi_med', 'arr_taxi_med60', 'arr20', 'rwy_dep20_fwd']
TRANS = ['cs_prev_ae', 'cs_next_ae', 'day_null_rate', 'day_n', 'stand_hour_n', 'day_ma_dev_med']
MET = ['tc', 'sknt', 'vsby', 'wx_frz', 'wx_rain', 'wx_fog', 'deice_risk']
STAND = ['mvt_inblk', 'inblk_sched', 'eobt_inblk', 'aobt_inblk', 'arr_taxi_in_prev', 'prev_same_al', 'prev_same_type', 'arr_fltnull_prev',
         'next_inblk_mvt', 'next_inblk_sched', 'arr_blk_eq_sched_prev']
REGIME = ['ex', 'rwy_ex5_m', 'rwy_ex5_n', 'rwy_ex15_m', 'rwy_ex15_n', 'rwy_ex60_m', 'rwy_ex60_n', 'rwy_ae15_m', 'rwy_syn15_m', 'apt_ex15_m',
          'apt_ex15_n', 'al_ex60_m', 'al_ex60_n', 'ex_prev', 'ex_next', 'ae_prev', 'ae_next', 'same_al_prev', 'arr_ex5_m', 'arr_ex5_n']
SRP = list(dict.fromkeys(CAT0 + BASE + CONG + NEW5 + ['al'] + P15 + SYN + TRANS + MET + STAND + REGIME + ['pi_sched']))
APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']
# ADDR26: label-free estimate of each airport-month cell's share (%) of the 2026 non-extreme SSE that the ADS-B layer cannot reach.
# Per-row MSE of the averaged-baseline 2025 month-held-out OOF by airport x month x row class, times the 2026 row counts per cell and class;
# the artefact class (B) is subtracted and the remainder scaled by (1 - 2026 gated share). Used only to express CV gains as a 2026 projection.
ADDR26 = {'LIRF-07': 28.0 - 12.0, 'LIRF-01': (8.7 - 3.6) * (1 - 0.79), 'LTFM-07': 8.8 - 0.2, 'LTFM-01': 7.6 - 0.4,
          'LFPG-01': (6.7 - 0.6) * (1 - 0.23), 'LFPG-07': (6.2 - 0.5) * (1 - 0.19)}
TE_KEYS = {'te_fl': ['ADEP_mvt', 'FLIGHT_mvt'], 'te_flst': ['ADEP_mvt', 'FLIGHT_mvt', 'STAND_mvt'], 'te_alst': ['ADEP_mvt', 'al', 'STAND_mvt'],
           'te_strw': ['ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt'], 'te_tyst': ['ADEP_mvt', 'AIRCRAFT_TYPE_mvt', 'STAND_mvt'],
           'te_alde': ['ADEP_mvt', 'al', 'ADES_mvt'], 'te_cs': ['ADEP_mvt', 'k_cs'],
           # same definition as TE_EXTRA of the screening script (second batch)
           'te_alstrw': ['ADEP_mvt', 'al', 'STAND_mvt', 'RUNWAY_mvt'], 'te_alhr': ['ADEP_mvt', 'al', 'e_loc_hour'],
           'te_sthr': ['ADEP_mvt', 'STAND_mvt', 'e_loc_hour'], 'te_rwhr': ['ADEP_mvt', 'RUNWAY_mvt', 'e_loc_hour'],
           'te_alty': ['ADEP_mvt', 'al', 'AIRCRAFT_TYPE_mvt'], 'te_derw': ['ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt'],
           'te_aldb': ['ADEP_mvt', 'al', 'dbk'], 'te_stdb': ['ADEP_mvt', 'STAND_mvt', 'dbk']}


def log(*x):
    print(time.strftime('%H:%M:%S'), *x, flush=True)



def block_se(d, blocks):
    s = np.bincount(blocks, weights=d)
    s = s[np.bincount(blocks) > 0]
    n = len(s)
    return float(np.sqrt(n / (n - 1) * ((s - s.mean()) ** 2).sum())) if n > 1 else float('nan')


def te_fold(d, r, trm, keys, k=20.0):
    x = d.select(keys + ['mo']).with_columns(pl.Series('_v', np.where(trm, r, 0.0)), pl.Series('_c', trm.astype(np.float64)))
    gm = x.group_by(keys + ['mo']).agg(pl.col('_v').sum().alias('_sm'), pl.col('_c').sum().alias('_nm'))
    ga = gm.group_by(keys).agg(pl.col('_sm').sum().alias('_sa'), pl.col('_nm').sum().alias('_na'))
    j = x.with_row_index('_k').join(ga, on=keys, how='left').join(gm, on=keys + ['mo'], how='left').sort('_k')
    sa = j['_sa'].fill_null(0).to_numpy(); na = j['_na'].fill_null(0).to_numpy()
    sm = j['_sm'].fill_null(0).to_numpy(); nm = j['_nm'].fill_null(0).to_numpy()
    s = np.where(trm, sa - sm, sa); n = np.where(trm, na - nm, na)
    return (s / (n + k)).astype(np.float32), n.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('storedir'); ap.add_argument('pf'); ap.add_argument('out')
    ap.add_argument('--fam', default='b'); ap.add_argument('--te', action='store_true'); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--lr', type=float, default=0.08); ap.add_argument('--cols', default='', help='comma-separated whitelist of extra columns (if given, only these candidate columns are used)')
    ap.add_argument('--tekeys', default='', help='comma-separated TE names (e.g. te_alst,te_strw); empty = all TE_KEYS')
    a = ap.parse_args()
    t0 = time.time()
    fams = [x for x in a.fam.split(',') if x and x != 'none']
    aux = ['MVT_ID_mvt', 'src', 'ym', 'TAXITIME_SEC_mvt', 'delta', 'aobt_sched', 'MVT_TIME_UTC_mvt', 'FLIGHT_mvt']
    d = pl.read_parquet(os.path.join(a.storedir, 'feat.parquet'), columns=list(dict.fromkeys(SRP + aux))).filter(pl.col('src') != 99).with_row_index('_i')
    pf = pl.read_parquet(a.pf)
    if a.cols:
        new = [c for c in a.cols.split(',') if c]
    else:
        new = [c for c in pf.columns if any(c.startswith(f + '_') for f in fams)]
    keep = ['MVT_ID_mvt'] + new + (['k_cs', 'e_loc_hour'] if a.te else [])
    d = d.join(pf.select(list(dict.fromkeys(keep))), on='MVT_ID_mvt', how='left').sort('_i')
    del pf
    d = d.with_columns((pl.col('ym') - 202500).alias('mo'), pl.col('MVT_TIME_UTC_mvt').dt.date().cast(pl.Int32).alias('dayn'))
    dm_ = pl.col('mvt_delay').cast(pl.Float64)
    d = d.with_columns(pl.when(dm_ <= 1200).then(0).when(dm_ <= 2400).then(1).when(dm_ <= 3600).then(2).when(dm_ <= 5400).then(3)
                       .when(dm_ <= 7200).then(4).when(dm_ <= 10800).then(5).otherwise(6).alias('dbk'))
    y = d['TAXITIME_SEC_mvt'].to_numpy().astype(np.float64)
    mo = d['mo'].to_numpy()
    p12 = np.load(os.path.join(a.storedir, 'oof12', 'srp_g5_avg.npy'))
    assert len(p12) == d.height
    B = ((d['flt_null'] == 1) & (d['ADEP_mvt'] == 'LIRF') & (d['mvt_delay'] > 3600)).fill_null(False).to_numpy()
    A = (y > 10800) & ~B
    C = ((d['flt_null'] == 0) & (d['aobt_sched'] > 1200) & (d['mvt_delay'] > 1200)).fill_null(False).to_numpy() & ~A & ~B
    S = (d['synth'] == 1).fill_null(False).to_numpy() & ~A & ~B & ~C
    cls = np.where(B, 'B', np.where(A, 'A', np.where(C, 'C', np.where(S, 'S', 'D'))))
    for k in [k for k in TE_KEYS if k not in (a.tekeys.split(',') if a.tekeys else ['te_fl', 'te_flst', 'te_alst', 'te_strw', 'te_tyst', 'te_alde', 'te_cs'])]:
        TE_KEYS.pop(k)
    tecols = list(TE_KEYS) + [f'n_{k}' for k in TE_KEYS] if a.te else []
    cols_new = SRP + new + tecols
    log(f'rows {d.height:,}; new cols {len(new)} te {len(tecols)}: {new + tecols}')
    # float32 matrix (categories encoded uniformly)
    X = np.full((d.height, len(cols_new)), np.nan, dtype=np.float32)
    for j, c in enumerate(cols_new):
        if c in tecols:
            continue
        s = d[c]
        if c in CAT:
            s = s.cast(pl.String)
            u = s.drop_nulls().unique().sort().to_list()
            X[:, j] = s.replace_strict(dict(zip(u, range(len(u)))), default=None, return_dtype=pl.Float32).to_numpy()
        else:
            X[:, j] = s.cast(pl.Float32).to_numpy()
    keepc = list(dict.fromkeys(['ADEP_mvt', 'FLIGHT_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'al', 'AIRCRAFT_TYPE_mvt', 'ADES_mvt', 'mo', 'dayn', 'delta', 'mvt_aobt3', 'dbk']
                               + (['k_cs', 'e_loc_hour'] if a.te else [])))
    d = d.select(keepc)  # features are in X now; free the large polars table (about 2.5 GB with 194 columns)
    gc.collect()
    log(f'X built {X.shape} {X.nbytes / 1e9:.2f}GB')
    nb = len(SRP)
    cat_b = [j for j, c in enumerate(SRP) if c in CAT]
    cat_n = [j for j, c in enumerate(cols_new) if c in CAT]
    dl = d['delta'].to_numpy().astype(np.float64)
    mao = d['mvt_aobt3'].to_numpy().astype(np.float64)
    P = dict(objective='regression', metric='rmse', learning_rate=a.lr, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=a.threads, seed=0, max_bin=63)
    pred = {'base': np.full(d.height, np.nan), 'new': np.full(d.height, np.nan)}
    info = {}
    r12 = y - p12
    keep_te = (cls != 'A') & (cls != 'B')
    for fo in (1, 7):
        va = mo == fo; tr = ~va
        if a.te:
            trm = tr & keep_te
            for k, keys in TE_KEYS.items():
                t_, n_ = te_fold(d, r12, trm, keys)
                X[:, cols_new.index(k)] = t_; X[:, cols_new.index('n_' + k)] = n_
        for arm, ncol, cat in (('base', nb, cat_b), ('new', len(cols_new), cat_n)):
            out = {}
            for tgt in ('taxi', 'delta'):
                if tgt == 'taxi':
                    mt, mv, yt, yv = tr, va, y, y
                else:
                    mt = tr & np.isfinite(dl); mv = va & np.isfinite(dl); yt, yv = dl, dl
                dtr = lgb.Dataset(X[mt, :ncol], yt[mt], categorical_feature=cat, free_raw_data=True)
                dva = lgb.Dataset(X[mv, :ncol], yv[mv], reference=dtr)
                m = lgb.train(P, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
                pv = m.predict(X[mv, :ncol], num_iteration=m.best_iteration)
                out[tgt] = (mv, pv, m.best_iteration)
                if arm == 'new' and tgt == 'taxi':
                    g = m.feature_importance('gain'); o = np.argsort(-g)[:40]
                    info[f'imp_{fo}'] = [(cols_new[j], round(float(g[j] / g.sum() * 100), 2)) for j in o if cols_new[j] not in SRP][:20]
                del dtr, dva, m; gc.collect()
                log(f'fold {fo} {arm} {tgt}: iters {out[tgt][2]} {time.time() - t0:.0f}s')
            pt = np.full(d.height, np.nan); pt[out['taxi'][0]] = out['taxi'][1]
            pd_ = pt.copy(); mv = out['delta'][0]; pd_[mv] = mao[mv] + out['delta'][1]
            pred[arm][va] = ((pt + pd_) / 2)[va]
            info[f'iters_{fo}_{arm}'] = [out['taxi'][2], out['delta'][2]]
    m17 = np.isin(mo, (1, 7))
    ensavg = np.full(d.height, np.nan)
    for fo in (1, 7):
        mk = mo == fo
        ensavg[mk] = (np.load(os.path.join(a.storedir, 'oof', f'ens_v5b_{fo}.npy')) + np.load(os.path.join(a.storedir, 'oof', f'ens_v6_{fo}.npy'))) / 2
    for k in pred:
        pred[k][cls == 'B'] = ensavg[cls == 'B']
    day = d['dayn'].to_numpy(); day = day - day.min()
    apt = d['ADEP_mvt'].to_numpy()
    nonA = (cls != 'A') & m17
    eb = (y - pred['base']) ** 2; en = (y - pred['new']) ** 2
    e8 = (y - ensavg) ** 2
    tr8 = ensavg + (pred['new'] - pred['base'])
    et = (y - tr8) ** 2
    tot_b = eb[nonA].sum(); tot8 = e8[nonA].sum()
    res = {'fam': fams, 'te': a.te, 'new_cols': new + tecols, 'info': info, 'sec': None,
           'member': {'base_nonA_mse': round(float(tot_b / nonA.sum()), 1), 'new_nonA_mse': round(float(en[nonA].sum() / nonA.sum()), 1),
                      'd_pct': round(float((eb - en)[nonA].sum() / tot_b * 100), 3), 'se_pct': round(block_se((eb - en)[nonA], day[nonA]) / tot_b * 100, 3),
                      'cls': {c: round(float((eb - en)[nonA & (cls == c)].sum() / tot_b * 100), 3) for c in 'CDS'}},
           'transplant_ensavg': {'ensavg_nonA_mse': round(float(tot8 / nonA.sum()), 1), 'd_pct': round(float((e8 - et)[nonA].sum() / tot8 * 100), 3),
                             'se_pct': round(block_se((e8 - et)[nonA], day[nonA]) / tot8 * 100, 3),
                             'cls': {c: round(float((e8 - et)[nonA & (cls == c)].sum() / tot8 * 100), 3) for c in 'CDS'}},
           'avgblend': {'d_pct_vs_avgblendbase': round(float((((y - (ensavg + pred['base']) / 2) ** 2) - ((y - (ensavg + pred['new']) / 2) ** 2))[nonA].sum() / tot8 * 100), 3)}}
    cells = {}
    for ap_ in APTS:
        for mm in (1, 7):
            mk = nonA & (apt == ap_) & (mo == mm) & (cls != 'B')
            cells[f'{ap_}-{mm:02d}'] = {'n': int(mk.sum()), 'member_rel': round(float((eb - en)[mk].sum() / eb[mk].sum() * 100), 2),
                                        'member_se': round(block_se((eb - en)[mk], day[mk]) / eb[mk].sum() * 100, 2),
                                        'ensavg_rel': round(float((e8 - et)[mk].sum() / e8[mk].sum() * 100), 2),
                                        'ensavg_se': round(block_se((e8 - et)[mk], day[mk]) / e8[mk].sum() * 100, 2)}
    res['cells'] = cells
    p26m = sum(cells[c]['member_rel'] / 100 * w for c, w in ADDR26.items())
    p26t = sum(cells[c]['ensavg_rel'] / 100 * w for c, w in ADDR26.items())
    res['p26'] = {'member_pct': round(p26m, 3), 'transplant_pct': round(p26t, 3)}
    res['sec'] = round(time.time() - t0)
    with open(a.out, 'w') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    np.savez_compressed(a.out.replace('.json', '_pred.npz'), base=pred['base'][m17], new=pred['new'][m17], idx=np.where(m17)[0])
    log('member', res['member']); log('transplant', res['transplant_ensavg']); log('p26', res['p26'])


if __name__ == '__main__':
    main()
