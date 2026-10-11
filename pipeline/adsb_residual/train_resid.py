"""ADS-B residual model: learns the residual over the averaged baseline from ADS-B features; either date-grouped CV to
estimate the local gain, or predictions for the test rows (the ranking file).

Usage:
  evaluation:  python train_resid.py --storedir n2 --feats feats --eval --out resid_eval.json
  prediction:  python train_resid.py --storedir n2 --feats feats --predict --tag 13 --out-dir sub
Baseline: training side = (oof/ens_v5b_{1,7} + oof/ens_v6_{1,7})/2, i.e. the month-held-out OOF of the two ensembles A and B
      (row order verified to align with feat.parquet); test side = v8_sub_copy.parquet (full-year average of A and B).
Metric: nonA RMSE = sqrt(nonA MSE); isA = y > 10800 and not (LIRF & flt_null & mvt_delay > 3600).
Prediction: airports whose 2025 training gating rate (share of non-extreme training rows with a ground segment) is >= RESID_FULL_THR
      (default 0.3; 0.15 in the submitted configuration) use the full model; the other airports (with the default threshold LEMD/LIRF/LFPG, which
      only have ground coverage in 2026) use the generic model (no airport code / coordinates / encodings; validated by leave-one-airport-out).
Gating: the residual is added only on rows with an ADS-B ground segment (ab_obs_first not null); all other rows = baseline
      (adding a residual to rows without a ground segment made the result worse in the first evaluation).
Per-month models: RESID_MONTH = off (default) | only (each predicted month uses only training days of the same month: residual model,
      grid encoding, full/generic assignment all per month) | w<k> (all training days, samples of the same month get LightGBM
      weight k; encodings and assignment use all days) | hyb:01[+07] (first predict all rows as with off, then for the listed
      months replace airports whose same-month training gating rate is >= the threshold by a same-month-only full model;
      airports without same-month ground samples, such as LEMD/LIRF in January, still use the model trained on all days).
      RESID_FULL_THR is the gating-rate threshold for the full model (default 0.3; 0.15 admits LEMD).
Target encoding: "blind-time" of the first observed grid cell (~150 m) and of the stand (y - obs_first / y - obs_push); training rows use
      date-nested OOF, test rows use all training rows.
Flow features: when RESID_EXTRA contains flow, flow/<date>.parquet (flow_feats.py: landing / take-off / go-around window rates) enters the full and
      generic models; RESID_FLOW_MINCOV (default 0.6): rows whose +-60 min ADS-B landing coverage fl_cov is below this value get the other fl_* set to null
      (receiver coverage differs between 2025 and 2026).
      RESID_NG = off (default) | flow | ctl: trains a separate residual model for non-gated rows (no ground segment; otherwise = baseline); features = baseline-type
      (base / airport code / hour / delay / counts / art) + flow (ctl has no flow, as a control); target = residual minus the airport x month mean of
      the training rows (so that the whole-month offset of the baseline's month-held-out OOF is not learned, which would be the same-month optimism of "correct all rows");
      RESID_NG_TRAIN = ng (default, only non-gated training rows) | all.
Air-traffic features: RESID_NG = air | flowair (non-gated model features additionally use air/<date>.parquet, air_feats.py: aircraft in the air within 150 km,
      holding, approach extension distance, approach spacing, ...); when RESID_EXTRA contains air the air features also enter the full/generic models;
      RESID_AIR_DIR (default air); RESID_AIR_MINCOV (default 0.5): rows whose ar_cov (number of approaches identified in the previous 120 minutes / number of ARR rows
      in the dataset) is below this value get the other ar_*/ac_* set to null. Default (RESID_NG=off, EXTRA without air): air is not read and the output is unchanged.
Stand-table correction: RESID_STAND = off (default: unchanged) | fix (ab_dist_stand is replaced by the value rebuilt with build_feats.py --stands in
      RESID_STAND_DIR (default feats_stand): merged OSM + apt.dat stand table, new normalisation rule; the other ADS-B columns still come from --feats) | fixrwy (fix plus
      ab_dist_stand_rwy = distance from the stand to the end of the active take-off runway, entering the full model and GENERIC). Overrides by MVT_ID; rows missing from the rebuilt directory are set to null.
Hierarchical fallback of the grid encoding: RESID_CELLFB = off (default: unchanged) | nb (when te_cell_gap_first finds no entry for the cell, fall back
      to the merged mean of the 3x3 neighbouring cells -> 7x7 -> airport mean, same m=8 smoothing, neighbours exclude the cell itself) | nbaug (nb + training rows are randomly treated as
      "not found" at the rate of missing cells among the gated rows of each airport of the prediction set, so that the model has seen fallback values; the rate is computed from the te keys, label-free)
      | nbaugf (nbaug + fallback-level feature te_cell_lvl: 0 own cell / 1 3x3 / 2 7x7 / 3 airport mean). RESID_CELLFB_EXCL (default LSZH, comma-separated) = airports without fallback
      (they still fall back to the airport mean and are not subjected to cell dropping); RESID_CELLFB_SEED (default 0) = random seed of the cell dropping.
      pred (fallback on the prediction side only): training is identical to off (training-row encodings do not fall back); the same boosters first predict once with the off
      encoding, then once more with only the missing te_cell_gap_first of the test rows replaced by fallback values (dual(): the first pass trains and caches the boosters, the second reuses
      them in call order without retraining), so rows whose cell is found are bit-identical; when writing predictions both <tag>_off (the check) and <tag> are written.
Airline ICAO prefix: RESID_ARTICAO = off (default) | on (the al of add_art first takes the 3-letter prefix of CALLSIGN_flt in raw (of the form ABC followed by digits),
      and falls back to the original al = letter prefix of FLIGHT_mvt; RESID_RAW_DIR defaults to raw).
Aircraft-type check of time matches: RESID_TYPEGATE = off (default: unchanged) | fam (reads RESID_TYPECHK (default typechk.parquet, produced by type_chk.py): for rows with a pure time match
      whose ADS-B type family differs from the dataset type family (tmis), all ADS-B columns obtained from the pairing are set to null = treated as unmatched, not gated, back to
      baseline; airport-level columns such as ab_taxi_now10 are left alone). Training and test rows are treated alike.
Further switches (all off by default: output unchanged):
  RESID_WINTER = off | <months, e.g. 11+12 or 06+11+12>: with hyb predictions, rows of January at the airports in RESID_WINTER_APTS (default LEMD,LIRF) use a full
      model trained on "2025-01 training rows (averaged baseline) + a 2025 winter pool of the same pattern (DEP rows of all airports in these months, baseline = oof12 linearly
      calibrated per airport to the averaged-baseline scale)". The pool comes from RESID_POOL_FEATS (default feats_tmfy, rebuilt with build_feats --stands --typefilter), RESID_POOL_BASE
      (default srp_g5_delta), RESID_POOL_CAL (lin | none); load_pool() loads it into the global POOL (main loads it automatically when WINTER != off).
  RESID_GENSWAP = <airport-month,...> (e.g. EGLL-07): with hyb predictions, the rows of these airport x months use the generic model (no airport code / coordinates / encodings),
      trained on the gated training rows of the other airports (leave-one-airport-out, excluding the airport itself).
  RESID_STANDFB = off | pred: prediction side only; when the stand of a test row is not in the encoding pool (te_stand_key_*, otherwise falling back to the airport mean), use the merged blind-time mean of the
      pool stands of the same airport within RESID_STANDFB_R metres (default 250) (same m=8 smoothing, coordinates looked up in the merged stand table RESID_STANDS); if none is found, still fall back to the airport mean.
      Shares dual() with CELLFB=pred: if either is pred the model runs twice, and the second pass swaps only the test-side encodings whose own switch is on.
  RESID_RANK_FEATS = <directory> (swaps only the test side): when predicting, the ADS-B features of the test rows (including ab_dist_stand) are read from this directory (e.g. feats_tma =
      build_feats --anchor segment-start anchoring); training rows are still read from --feats / RESID_STAND_DIR, so training is identical and only rows whose features differ between the two directories change.
  RESID_NULLAB = <column,...> (risk ablation, e.g. ab_taxi_now10,ab_npts,ab_max_gap): after prep, these coverage-dependent columns are set to null (training, pool and test rows alike).
  RESID_TK = off | gap | all | family names joined with + (gap, motion, stand): track-context features before the segment start (the label-free tk_* trajectory features) enter the full-model
      BASEF and the generic-model GENERIC (appended to the end of both lists; gap = 6 columns, all = 17 columns). Joined by MVT_ID: training and test rows read RESID_TK_FILE (default v22/tk_main.parquet,
      produced by build_track_context.py main), winter-pool rows read RESID_TK_POOL (default v22/tk_pool.parquet, produced by build_track_context.py pool); not found = null.
      RESID_TK_LOAD=1: also append the 17 columns to the data frame when TK=off (the feature table is then switched with set_tk() while an evaluation script runs).
  RESID_BASE_SUB = <path> (default <storedir>/v8_sub_copy.parquet; relative paths are resolved against the --storedir directory): baseline file of the test side (here the baseline retrained on the final files).
  RESID_N26 = 344841 | auto: expected number of test rows; auto = row count of RESID_RAW_DIR/submitting.parquet (and asserts that it equals the row count of the baseline file and of the joined table).
  RESID_WINTER_TEST = 01 | 01,02: test months for which the winter-pool full model is used when WINTER is on (the training set is always "2025-01 training rows + pool", same WINTER_APTS);
      default 01. With 01,02 February is treated like January: it is also 2 to 3 months from pool months 11/12 and also winter (a rule-level argument, not validated on February).
"""
import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

BASE_COLS = ['MVT_ID_mvt', 'ADEP_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'src', 'ym', 'TAXITIME_SEC_mvt', 'MVT_TIME_UTC_mvt',
             'hour', 'mvt_aobt3', 'mvt_delay', 'flt_null', 'dep10', 'arr15', 'synth', 'al',
             'tc', 'sknt', 'vsby', 'wx_frz', 'deice_risk']
# Of the tail SSE, 20% is a BLOCK~=SCHED artefact and 20% is long blind time -> give the residual model the airline artefact-rate encoding and weather
EXTRA = os.environ.get('RESID_EXTRA', 'none').split(',')  # default = none; RESID_EXTRA=art is recommended (weather does not help)
WX = ['tc', 'sknt', 'vsby', 'wx_frz', 'deice_risk']
ART = ['te_art_apt_al', 'te_art_al', 'd_sched']
FLOW = ['fl_arr10', 'fl_arr20', 'fl_arr40', 'fl_arr_n10', 'fl_dep20', 'fl_rwy_arr20', 'fl_gap_min20',
        'fl_gap_mean20', 'fl_vfin20', 'fl_ga40', 'fl_cov']
NG = os.environ.get('RESID_NG', 'off')
NG_TRAIN = os.environ.get('RESID_NG_TRAIN', 'ng')
FLOW_DIR = os.environ.get('RESID_FLOW_DIR', 'flow')
FLOW_MINCOV = float(os.environ.get('RESID_FLOW_MINCOV', '0.6'))
USE_FLOW = 'flow' in EXTRA or NG in ('flow', 'flowair')
AIR = ['ac_r30', 'ac_r60', 'ac_r100', 'ac_r150', 'ac_inb100', 'ac_hold', 'ac_hold30',
       'ar_n40', 'ar_ext20', 'ar_ext40', 'ar_ext_max40', 'ar_t100_40', 'ar_hold_frac40', 'ar_hturn40',
       'ar_x60_40', 'ar_t60_40', 'ar_vterm40', 'ar_vfin20', 'ar_gap_min20', 'ar_gap_mean20', 'ar_gap_std40', 'ar_cov']
AIR_DIR = os.environ.get('RESID_AIR_DIR', 'air')
AIR_MINCOV = float(os.environ.get('RESID_AIR_MINCOV', '0.5'))
USE_AIR = 'air' in EXTRA or NG in ('air', 'flowair')


def art_table(tr, m=20.0):
    """Share of flights delayed by > 20 minutes with |y - mvt_delay| <= 60 (BLOCK~=SCHED), smoothed per airport x airline and per airline."""
    c = tr.filter(pl.col('mvt_delay') > 1200).with_columns(
        ((pl.col('TAXITIME_SEC_mvt') - pl.col('mvt_delay')).abs() <= 60).cast(pl.Float64).alias('art'))
    g0 = c['art'].mean()
    a1 = c.group_by('ADEP_mvt', 'al').agg(pl.col('art').sum().alias('s'), pl.len().alias('n')).with_columns(
        ((pl.col('s') + m * g0) / (pl.col('n') + m)).alias('te_art_apt_al')).select('ADEP_mvt', 'al', 'te_art_apt_al')
    a2 = c.group_by('al').agg(pl.col('art').sum().alias('s'), pl.len().alias('n')).with_columns(
        ((pl.col('s') + m * g0) / (pl.col('n') + m)).alias('te_art_al')).select('al', 'te_art_al')
    return a1, a2, g0


ARTICAO = os.environ.get('RESID_ARTICAO', 'off')
assert ARTICAO in ('off', 'on'), ARTICAO
RAW_DIR = os.environ.get('RESID_RAW_DIR', 'raw')


def al_icao(d):
    """RESID_ARTICAO=on: al <- ICAO 3-letter prefix of CALLSIGN_flt, falling back to the original al. Mapped by MVT_ID (no join, row order is kept)."""
    if ARTICAO == 'off':
        return d
    fs = sorted(Path(RAW_DIR).glob('training_*.parquet')) + [Path(RAW_DIR) / 'ranking.parquet']
    cs = pl.concat([pl.read_parquet(f, columns=['MVT_ID_mvt', 'CALLSIGN_flt']) for f in fs], how='vertical_relaxed').with_columns(
        pl.col('CALLSIGN_flt').str.strip_chars().str.to_uppercase().str.extract(r'^([A-Z]{3})[0-9]').alias('csp')
    ).drop_nulls('csp').unique('MVT_ID_mvt').join(d.select('MVT_ID_mvt').unique(), on='MVT_ID_mvt', how='semi')
    csp = pl.col('MVT_ID_mvt').replace_strict(cs['MVT_ID_mvt'], cs['csp'], default=None, return_dtype=pl.String)
    return d.with_columns(pl.coalesce(csp, pl.col('al')).alias('al'))


def add_art(v, full_train, exclude_ym=None):
    tr = full_train if exclude_ym is None else full_train.filter(pl.col('ym') != exclude_ym)
    a1, a2, g0 = art_table(tr)
    return v.join(a1, on=['ADEP_mvt', 'al'], how='left').join(a2, on='al', how='left').with_columns(
        pl.col('te_art_apt_al').fill_null(g0), pl.col('te_art_al').fill_null(g0))
STAND = os.environ.get('RESID_STAND', 'off')
assert STAND in ('off', 'fix', 'fixrwy'), STAND
STAND_DIR = os.environ.get('RESID_STAND_DIR', 'feats_stand')
TYPEGATE = os.environ.get('RESID_TYPEGATE', 'off')
assert TYPEGATE in ('off', 'fam'), TYPEGATE
TYPECHK = os.environ.get('RESID_TYPECHK', 'typechk.parquet')
MATCH_COLS = ['ab_hex', 'ab_dlift', 'ab_cs_match', 'ab_etype', 'ab_obs_first', 'ab_obs_move', 'ab_obs_push', 'ab_dwell_seen',
              'ab_dist_first_lift', 'ab_path_len', 'ab_stop_after_move', 'ab_gs_first', 'ab_dist_stand', 'ab_max_gap',
              'ab_npts', 'ab_first_lat', 'ab_first_lon', 'ab_hold_last', 'ab_last_ground']
AB = ['ab_dlift', 'ab_cs_match', 'ab_obs_first', 'ab_obs_move', 'ab_dwell_seen', 'ab_dist_first_lift',
      'ab_path_len', 'ab_stop_after_move', 'ab_gs_first', 'ab_dist_stand', 'ab_max_gap', 'ab_npts',
      'ab_first_lat', 'ab_first_lon', 'ab_taxi_now10', 'ab_hold_last', 'ab_obs_push', 'ab_etype', 'ab_last_ground'] + \
     (['ab_dist_stand_rwy'] if STAND == 'fixrwy' else [])
TE = [('cell', 'gap_first'), ('stand_key', 'gap_first'), ('stand_key', 'gap_push')]
TE_NAMES = [f'te_{k}_{t}' for k, t in TE]
PF = ['pf_cell', 'pf_stand', 'pp_stand']
# RESID_TK: column names, family definitions and column order (TK_ALL, FAM) are the same as in track_context_rows.py, so that CV results are comparable
TK_ALL = ['tk_gs1', 'tk_gs5', 'tk_gs8', 'tk_disp30', 'tk_rev', 'tk_turn', 'tk_trk_agree', 'tk_first_x', 'tk_push_x', 'tk_sleave',
          'tk_sdist_first_x', 'tk_stop_gap_s', 'tk_stop_gap_m', 'tk_arr', 'tk_ep_n', 'tk_ep_first', 'tk_stop_code']
TK_FAM = {'gap': ['tk_stop_gap_s', 'tk_stop_gap_m', 'tk_stop_code', 'tk_arr', 'tk_ep_n', 'tk_ep_first'],
          'motion': ['tk_gs1', 'tk_gs5', 'tk_gs8', 'tk_disp30', 'tk_rev', 'tk_turn', 'tk_trk_agree'],
          'stand': ['tk_sleave', 'tk_sdist_first_x', 'tk_first_x', 'tk_push_x']}
TK_STOPS = ['gap', 'max3h', 'air', 'miss', 'start', 'win']


def tk_cols(spec):
    return [] if spec in ('', 'off', 'none') else (list(TK_ALL) if spec == 'all' else sum((TK_FAM[f] for f in spec.split('+')), []))


TK = os.environ.get('RESID_TK', 'off')
TKC = tk_cols(TK)
TK_FILE = os.environ.get('RESID_TK_FILE', 'v22/tk_main.parquet')
TK_POOL_FILE = os.environ.get('RESID_TK_POOL', 'v22/tk_pool.parquet')
TK_LOAD = bool(TKC) or os.environ.get('RESID_TK_LOAD', '') == '1'
BASEF0 = ['base', 'apt_code', 'hour', 'mvt_aobt3', 'mvt_delay', 'flt_null', 'dep10', 'arr15', 'synth',
          'd_move', 'd_first', 'd_push'] + AB + (ART if 'art' in EXTRA else []) + (WX if 'wx' in EXTRA else []) + \
         (FLOW if 'flow' in EXTRA else []) + (AIR if 'air' in EXTRA else [])
BASEF = BASEF0 + TKC
CELLFB = os.environ.get('RESID_CELLFB', 'off')
assert CELLFB in ('off', 'nb', 'nbaug', 'nbaugf', 'pred'), CELLFB
CELLFB_EXCL = set(os.environ.get('RESID_CELLFB_EXCL', 'LSZH').split(',')) - {''}
CELLFB_SEED = int(os.environ.get('RESID_CELLFB_SEED', '0'))
LVL = ['te_cell_lvl'] if CELLFB == 'nbaugf' else []
FEATS = BASEF + TE_NAMES + PF + LVL
WINTER = os.environ.get('RESID_WINTER', 'off')
WINTER_APTS = [x for x in os.environ.get('RESID_WINTER_APTS', 'LEMD,LIRF').split(',') if x]
POOL_FEATS = os.environ.get('RESID_POOL_FEATS', 'feats_tmfy')
POOL_BASE = os.environ.get('RESID_POOL_BASE', 'srp_g5_delta')
POOL_CAL = os.environ.get('RESID_POOL_CAL', 'lin')
WINTER_TEST = [x for x in os.environ.get('RESID_WINTER_TEST', '01').split(',') if x]
BASE_SUB = os.environ.get('RESID_BASE_SUB', 'v8_sub_copy.parquet')
N26 = os.environ.get('RESID_N26', '344841')
POOL = None  # winter pool (filled by load_pool)
GENSWAP = [x for x in os.environ.get('RESID_GENSWAP', '').split(',') if x]
STANDFB = os.environ.get('RESID_STANDFB', 'off')
assert STANDFB in ('off', 'pred'), STANDFB
STANDFB_R = float(os.environ.get('RESID_STANDFB_R', '250'))
STANDS_FILE = os.environ.get('RESID_STANDS', 'ext/xplane-apt/stand_coords_merged.parquet')
STAND_LOG = []  # statistics of each stand fallback
RANK_FEATS = os.environ.get('RESID_RANK_FEATS', '')  # feature directory read for the test rows (e.g. feats_tma); empty = same as --feats
CELL_LOG = []  # grid-cell fallback statistics of te in each fit_predict (recorded when CELLFB != off)
PRED_FB = False  # CELLFB=pred: when True the test-side grid encoding uses the fallback values (switched by dual())
_BC = dict(mode=None, models=[], i=0)  # booster cache: store = train and keep; reuse = fetch in call order, no retraining


def _train(params, Xtr, r, w, rounds):
    if _BC['mode'] == 'reuse':
        m = _BC['models'][_BC['i']]
        _BC['i'] += 1
        return m
    m = lgb.train(params, lgb.Dataset(Xtr, r, weight=w), num_boost_round=rounds)
    if _BC['mode'] == 'store':
        _BC['models'].append(m)
    return m


def dual(fn, *args, **kw):
    """CELLFB=pred: runs fn twice -- the first pass with the off encoding (trains and caches the boosters), the second with the test-side fallback (reuses the boosters). Returns (off, fb)."""
    global PRED_FB
    _BC.update(mode='store', models=[], i=0)
    PRED_FB = False
    try:
        o = fn(*args, **kw)
        _BC.update(mode='reuse', i=0)
        PRED_FB = True
        f = fn(*args, **kw)
        assert _BC['i'] == len(_BC['models']), (_BC['i'], len(_BC['models']))
    finally:
        PRED_FB = False
        _BC.update(mode=None, models=[], i=0)
    return o, f
PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, num_threads=int(os.environ.get('RESID_THREADS', '6')))  # use RESID_THREADS to match the number of cores available to the process
PARAMS_ORIG = dict(PARAMS)
MONTH = os.environ.get('RESID_MONTH', 'off')
FULL_THR = float(os.environ.get('RESID_FULL_THR', '0.3'))
APTS = ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LSZH', 'LTFM']


def read_ab(feats, year):
    ab = _read_ab(feats, year)
    if TYPEGATE == 'off':
        return ab
    tm = pl.read_parquet(TYPECHK, columns=['MVT_ID_mvt', 'tmis']).filter(pl.col('tmis')).select('MVT_ID_mvt')
    bad = pl.col('MVT_ID_mvt').is_in(tm['MVT_ID_mvt'].implode())
    cols = [c for c in MATCH_COLS if c in ab.columns]
    print(f'TYPEGATE {year}: nulled {ab.filter(bad).height} rows', flush=True)
    return ab.with_columns([pl.when(bad).then(None).otherwise(pl.col(c)).alias(c) for c in cols])


def _read_ab(feats, year):
    fs = sorted(Path(feats).glob(f'{year}-*.parquet'))
    ab = pl.concat([pl.read_parquet(f) for f in fs], how='diagonal_relaxed').unique('MVT_ID_mvt')
    if STAND == 'off':
        return ab
    cols = ['ab_dist_stand'] + (['ab_dist_stand_rwy'] if STAND == 'fixrwy' else [])
    st = pl.concat([pl.read_parquet(f, columns=['MVT_ID_mvt'] + cols) for f in sorted(Path(STAND_DIR).glob(f'{year}-*.parquet'))]
                   ).unique('MVT_ID_mvt')
    return ab.drop('ab_dist_stand').join(st, on='MVT_ID_mvt', how='left')


def add_flow(v, year):
    if not USE_FLOW:
        return v
    fs = sorted(Path(FLOW_DIR).glob(f'{year}-*.parquet'))
    fl = pl.concat([pl.read_parquet(f) for f in fs]).drop('date').unique('MVT_ID_mvt')
    v = v.join(fl, on='MVT_ID_mvt', how='left')
    low = pl.col('fl_cov').fill_null(0) < FLOW_MINCOV
    return v.with_columns([pl.when(low).then(None).otherwise(pl.col(c)).alias(c) for c in FLOW if c != 'fl_cov'])


def add_air(v, year):
    if not USE_AIR:
        return v
    fs = sorted(f for f in Path(AIR_DIR).glob(f'{year}-*.parquet') if not f.stem.endswith('_arr'))
    fl = pl.concat([pl.read_parquet(f) for f in fs]).drop('date').unique('MVT_ID_mvt')
    v = v.join(fl, on='MVT_ID_mvt', how='left')
    low = pl.col('ar_cov').fill_null(0) < AIR_MINCOV
    return v.with_columns([pl.when(low).then(None).otherwise(pl.col(c)).alias(c) for c in AIR if c != 'ar_cov'])


def load_train(storedir, feats):
    d = al_icao(pl.read_parquet(Path(storedir) / 'feat.parquet', columns=BASE_COLS))
    parts = []
    for mo in (1, 7):
        v = d.filter((pl.col('src') != 99) & (pl.col('ym') == 202500 + mo))
        b = (np.load(Path(storedir) / 'oof' / f'ens_v5b_{mo}.npy') + np.load(Path(storedir) / 'oof' / f'ens_v6_{mo}.npy')) / 2
        assert len(b) == v.height
        v = v.with_columns(pl.Series('base', b))
        parts.append(add_art(v, d.filter(pl.col('src') != 99), exclude_ym=202500 + mo))
    v = pl.concat(parts).with_columns(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d').alias('date'))
    ab = read_ab(feats, 2025)
    v = v.filter(pl.col('date').is_in(ab['date'].unique().to_list())).join(ab.drop('date'), on='MVT_ID_mvt', how='left')
    return add_tk(prep(add_air(add_flow(v, 2025), 2025)), TK_FILE)


def tk_frame(path):
    """Per-row tk_* table -> MVT_ID_mvt + 17 columns (Float64, NaN -> null; tk_stop string -> tk_stop_code), as in track_context_rows.tk_frame."""
    cols = [c for c in TK_ALL if c != 'tk_stop_code']
    R = pl.read_parquet(path, columns=['MVT_ID_mvt', 'tk_stop'] + cols)
    R = R.with_columns([pl.col(c).cast(pl.Float64).fill_nan(None) for c in cols] +
                       [pl.col('tk_stop').replace_strict(TK_STOPS, list(range(len(TK_STOPS))), default=None,
                                                         return_dtype=pl.Float64).alias('tk_stop_code')]).drop('tk_stop')
    return R.unique('MVT_ID_mvt', keep='first', maintain_order=True)


def add_tk(v, path):
    """When RESID_TK != off (or RESID_TK_LOAD=1), left-joins the 17 tk_* columns by MVT_ID with row order unchanged; otherwise returns the input as is (off = unchanged)."""
    if not TK_LOAD:
        return v
    n = v.height
    v = v.with_row_index('_tkr').join(tk_frame(path), on='MVT_ID_mvt', how='left').sort('_tkr').drop('_tkr')
    assert v.height == n, (v.height, n)
    print('tk joined', path, 'rows', n, 'tk_arr non-null', int(v['tk_arr'].is_not_null().sum()),
          'gated with tk_ok-like', int((v['gate'] & v['tk_stop_code'].is_not_null()).sum()), 'gated', int(v['gate'].sum()), flush=True)
    return v


def set_tk(cols):
    """Switch the tk features at run time (for evaluation scripts; the data frame must already carry the 17 columns via TK_LOAD). cols = [] means off."""
    global BASEF, FEATS, GENERIC, TKC
    TKC = list(cols)
    BASEF = BASEF0 + TKC
    FEATS = BASEF + TE_NAMES + PF + LVL
    GENERIC = GENERIC0 + TKC


def load_pool(storedir, v, months):
    """Winter / new-month pool: the training DEP rows of these months in feat.parquet + POOL_FEATS features; baseline = oof12 (per airport, calibrated on the rows of v in months 1 and 7
    by fitting averaged baseline ~= alpha + beta * oof12; both are OOF, labels are not used); isA rows removed; column order and types aligned to v. Returns (pool, calibration table)."""
    d = al_icao(pl.read_parquet(Path(storedir) / 'feat.parquet', columns=BASE_COLS))
    b12 = np.load(Path(storedir) / 'oof12' / f'{POOL_BASE}.npy')
    dtr = d.filter(pl.col('src') != 99)
    assert len(b12) == dtr.height, (len(b12), dtr.height)
    dtr = dtr.with_columns(pl.Series('b12', b12))
    j = v.select('MVT_ID_mvt', 'ADEP_mvt', 'base').join(dtr.select('MVT_ID_mvt', 'b12'), on='MVT_ID_mvt', how='inner')
    cal = {}
    for x in APTS:
        s = j.filter(pl.col('ADEP_mvt') == x)
        if POOL_CAL == 'lin' and s.height > 100:
            be, al_ = np.polyfit(s['b12'].to_numpy(), s['base'].to_numpy(), 1)
            cal[x] = (float(al_), float(be))
        else:
            cal[x] = (0.0, 1.0)
    fs = [f for f in sorted(Path(POOL_FEATS).glob('2025-*.parquet')) if f.stem[5:7] in months]
    ab = pl.concat([pl.read_parquet(f) for f in fs], how='diagonal_relaxed').unique('MVT_ID_mvt')
    parts = []
    for mm in months:
        ym = 202500 + int(mm)
        days = [x for x in ab['date'].unique().to_list() if x[5:7] == mm]
        if not days:
            print('pool: no feats for month', mm, flush=True)
            continue
        p = add_art(dtr.filter(pl.col('ym') == ym), d.filter(pl.col('src') != 99), exclude_ym=ym).with_columns(
            pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d').alias('date'))
        parts.append(p.filter(pl.col('date').is_in(days)).join(ab.drop('date'), on='MVT_ID_mvt', how='left'))
    p = pl.concat(parts, how='diagonal_relaxed')
    a_, b = p['ADEP_mvt'].to_numpy(), p['b12'].to_numpy().astype(float).copy()
    for x, (al_, be) in cal.items():
        b[a_ == x] = al_ + be * b[a_ == x]
    p = add_tk(prep(p.with_columns(pl.Series('base', b))).filter(~pl.col('isA')), TK_POOL_FILE)
    p = p.select([pl.col(c).cast(t) for c, t in v.schema.items()])
    print('pool rows', p.height, 'days', p['date'].n_unique(), 'gate', round(float(p['gate'].mean()), 3),
          {x: int(((p['ADEP_mvt'] == x) & p['gate']).sum()) for x in APTS}, 'cal', cal, flush=True)
    return p, cal


def load_rank(storedir, feats):
    full = al_icao(pl.read_parquet(Path(storedir) / 'feat.parquet', columns=BASE_COLS))
    d = full.filter(pl.col('src') == 99)
    ensavg = pl.read_parquet(Path(storedir) / BASE_SUB).rename({'TAXITIME_SEC_mvt': 'base'})
    v = d.join(ensavg.with_columns(pl.col('base').cast(pl.Float64)), on='MVT_ID_mvt', how='inner')
    v = add_art(v, full.filter(pl.col('src') != 99))
    n26 = pl.scan_parquet(Path(RAW_DIR) / 'submitting.parquet').select(pl.len()).collect().item() if N26 == 'auto' else int(N26)
    assert v.height == ensavg.height == n26, (v.height, ensavg.height, n26)
    v = v.with_columns(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d').alias('date'))
    ab = read_ab(feats, 2026)
    v = v.join(ab.drop('date'), on='MVT_ID_mvt', how='left')
    return add_tk(prep(add_air(add_flow(v, 2026), 2026)), TK_FILE)


NULLAB = [c for c in os.environ.get('RESID_NULLAB', '').split(',') if c]  # these coverage-type features are set to null as whole columns (default: none)


def prep(v):
    v = _prep(v)
    return v.with_columns([pl.lit(None, v.schema[c]).alias(c) for c in NULLAB]) if NULLAB else v


def _prep(v):
    for c in AB + ['ab_hex']:
        if c not in v.columns:
            v = v.with_columns(pl.lit(None, pl.Float64).alias(c))
    return v.with_columns(
        (pl.col('ab_obs_move') - pl.col('base')).alias('d_move'),
        (pl.col('ab_obs_first') - pl.col('base')).alias('d_first'),
        (pl.col('ab_obs_push') - pl.col('base')).alias('d_push'),
        (pl.col('mvt_delay') - pl.col('base')).alias('d_sched'),
        pl.concat_str([pl.col('ADEP_mvt'), (pl.col('ab_first_lat') / 0.0015).round().cast(pl.Int64).cast(pl.String),
                       (pl.col('ab_first_lon') / 0.002).round().cast(pl.Int64).cast(pl.String)], separator='|').alias('cell'),
        pl.concat_str([pl.col('ADEP_mvt'), pl.col('STAND_mvt')], separator='|').alias('stand_key'),
        pl.col('ADEP_mvt').replace_strict(APTS, list(range(10)), default=None, return_dtype=pl.Int8).alias('apt_code'),
        pl.col('ab_obs_first').is_not_null().alias('gate'),
        ((pl.col('TAXITIME_SEC_mvt') > 10800) &
         ~((pl.col('flt_null') == 1) & (pl.col('ADEP_mvt') == 'LIRF') & (pl.col('mvt_delay') > 3600))).fill_null(False).alias('isA'))


def te_map(keys, tgt, apt, m=8.0):
    """Smoothed mean: (sum + m * airport mean) / (n + m). Rows with NaN tgt are not counted."""
    ok = ~np.isnan(tgt)
    df = pl.DataFrame({'k': keys[ok], 'a': apt[ok], 'y': tgt[ok]})
    am = dict(df.group_by('a').agg(pl.col('y').mean()).iter_rows())
    g = df.group_by('k', 'a').agg(pl.col('y').sum().alias('s'), pl.len().alias('n'))
    return {k: (s_ + m * am[a_]) / (n + m) for k, a_, s_, n in g.iter_rows()}, am


def te_apply(keys, apt, mp, am):
    return np.array([mp.get(k, am.get(a_, np.nan)) if k is not None else np.nan for k, a_ in zip(keys, apt)])


def cell_fb_table(keys, tgt, apt):
    """(airport, i, j) -> (sum, count) used for the grid fallback; keys look like 'APT|i|j'; rows with NaN tgt are not counted."""
    ok = ~np.isnan(tgt)
    df = pl.DataFrame({'k': keys[ok], 'y': tgt[ok]}).group_by('k').agg(pl.col('y').sum().alias('s'), pl.len().alias('n'))
    out = {}
    for k, s_, n in df.iter_rows():
        a_, i, j = k.split('|')
        out[(a_, int(i), int(j))] = (s_, n)
    return out


def te_apply_fb(keys, apt, mp, am, fb, force=None, m=8.0):
    """Hierarchical-fallback version of te_apply: own cell (skipped for force rows) -> 3x3 -> 7x7 merged neighbour mean (excluding the own cell, same m smoothing) -> airport mean.
    CELLFB_EXCL airports fall back directly to the airport mean. Returns (value, level 0/1/2/3, NaN for an empty key)."""
    val, lvl = np.full(len(keys), np.nan), np.full(len(keys), np.nan)
    for r, (k, a_) in enumerate(zip(keys, apt)):
        if k is None:
            continue
        if (force is None or not force[r]) and k in mp:
            val[r], lvl[r] = mp[k], 0
            continue
        mu = am.get(a_, np.nan)
        val[r], lvl[r] = mu, 3
        if a_ in CELLFB_EXCL:
            continue
        _, i, j = k.split('|')
        i, j = int(i), int(j)
        for lv, rad in ((1, 1), (2, 3)):
            s_, n = 0.0, 0
            for di in range(-rad, rad + 1):
                for dj in range(-rad, rad + 1):
                    if di or dj:
                        x = fb.get((a_, i + di, j + dj))
                        if x is not None:
                            s_ += x[0]
                            n += x[1]
            if n > 0:
                val[r], lvl[r] = (s_ + m * mu) / (n + m), lv
                break
    return val, lvl


_STAB = {}


def _canon(s):
    """Stand-name normalisation, the same rule as build_feats.canon()."""
    import re
    if s is None:
        return None
    s = re.sub(r'\(.*?\)', '', str(s).upper().strip())
    s = re.sub(r'^(GATE|RAMP|STAND|PARKING|POSITION|POS|APRON)\b\s*', '', s)
    s = re.sub(r'[\s\-_]', '', s)
    m = re.fullmatch(r'([A-Z]*)(\d+)([A-Z]*)', s)
    if m:
        s = m.group(1) + str(int(m.group(2))) + m.group(3)
    return s or None


def stand_ll(k):
    """stand_key 'APT|STAND' -> (lat, lon): look up the raw stand name in the merged stand table first, then the canon key (as in build_feats --stands)."""
    if not _STAB:
        m = pl.read_parquet(STANDS_FILE).filter(pl.col('lat').is_not_null())
        st, sk = {}, {}
        for apt, stand, key, la, lo in m.select('apt', 'stand', 'stand_key', 'lat', 'lon').iter_rows():
            st.setdefault(apt, {})[stand] = (la, lo)
            if key:
                sk.setdefault(apt, {}).setdefault(key, (la, lo))
        _STAB.update(st=st, sk=sk)
    if k is None:
        return None
    a_, s = k.split('|', 1)
    return _STAB['st'].get(a_, {}).get(s) or _STAB['sk'].get(a_, {}).get(_canon(s))


def _hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def stand_fb(kte, apt_te, mp, am, ktr, tgt, cur, m=8.0):
    """RESID_STANDFB=pred: when the stand of a test row is not in the encoding pool (mp), use the merged mean of the pool stands of the same airport within STANDFB_R metres (m-smoothed towards the airport mean)."""
    ok = ~np.isnan(tgt)
    g = pl.DataFrame({'k': ktr[ok], 'y': tgt[ok]}).group_by('k').agg(pl.col('y').sum().alias('s'), pl.len().alias('n'))
    seen = {}
    for k, s_, n in g.iter_rows():
        ll = stand_ll(k)
        if ll:
            seen.setdefault(k.split('|', 1)[0], []).append((ll[0], ll[1], s_, n))
    arr = {a_: np.array(x, dtype=float) for a_, x in seen.items()}
    out, cache, log = cur.copy(), {}, {}
    for r, (k, a_) in enumerate(zip(kte, apt_te)):
        if k is None or k in mp:
            continue
        if k not in cache:
            ll, val, dmin = stand_ll(k), None, None
            if ll and a_ in arr:
                A = arr[a_]
                dd = _hav(ll[0], ll[1], A[:, 0], A[:, 1])
                dmin = float(dd.min())
                w = dd <= STANDFB_R
                if w.any():
                    val = (A[w, 2].sum() + m * am.get(a_, np.nan)) / (A[w, 3].sum() + m)
            cache[k] = (val, dmin, ll is not None)
        val, dmin, has = cache[k]
        L = log.setdefault(a_, dict(rows=0, has_ll=0, filled=0, keys=set(), dmin=[]))
        L['rows'] += 1
        L['has_ll'] += int(has)
        if val is not None:
            out[r] = val
            L['filled'] += 1
        if k not in L['keys']:
            L['keys'].add(k)
            if dmin is not None:
                L['dmin'].append(dmin)
    STAND_LOG.append({a_: dict(rows=L['rows'], has_ll=L['has_ll'], filled=L['filled'], n_keys=len(L['keys']),
                               dmin_q=[round(float(q), 1) for q in np.quantile(L['dmin'], [.1, .5, .9])] if L['dmin'] else None)
                      for a_, L in sorted(log.items())})
    return out


def te_targets(v):
    y = v['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    t = {'gap_first': y - v['ab_obs_first'].to_numpy(), 'gap_push': y - v['ab_obs_push'].to_numpy()}
    return {n: np.where((x > -1800) & (x < 3600), x, np.nan) for n, x in t.items()}


# By default the "first-observation blind time per stand" is dropped: it depends on the receiver distribution (more receivers in 2026 -> the same stand is seen earlier -> the blind time learned on 2025 is overestimated);
# in CV this costs only 0.23 percentage points (ablation 15.35% -> 15.12%). The per-grid-cell blind time is a physical quantity and is kept. RESID_DROP=none restores it.
_d = os.environ.get('RESID_DROP', 'te_stand_key_gap_first,pf_stand')
DROP = set() if _d == 'none' else set(_d.split(',')) - {''}


def matrix(v, T):
    X0 = v.select(BASEF).to_numpy().astype(np.float32)
    of, op, b = v['ab_obs_first'].to_numpy(), v['ab_obs_push'].to_numpy(), v['base'].to_numpy()
    extra = [T[n] for n in TE_NAMES] + [of + T['te_cell_gap_first'] - b, of + T['te_stand_key_gap_first'] - b,
                                       op + T['te_stand_key_gap_push'] - b]
    names = TE_NAMES + PF
    extra = [np.full(v.height, np.nan) if n in DROP else x for n, x in zip(names, extra)]
    extra += [T[n] for n in LVL]
    return np.column_stack([X0] + extra).astype(np.float32)


def fit_predict(tr, te, inner=4, rounds=600, seeds=(0,), w=None):
    """tr: training DataFrame (with labels, isA removed); te: prediction DataFrame. Returns the residual prediction for te (not yet gated)."""
    apt_tr, apt_te = tr['ADEP_mvt'].to_numpy(), te['ADEP_mvt'].to_numpy()
    tg = te_targets(tr)
    days = sorted(set(tr['date'].to_list()))
    fin = np.array([{d: i % inner for i, d in enumerate(days)}[d] for d in tr['date'].to_list()])
    Ttr = {n: np.full(tr.height, np.nan) for n in TE_NAMES + LVL}
    Tte = {}
    for (key, tn), name in zip(TE, TE_NAMES):
        ktr, kte = tr[key].to_numpy(), te[key].to_numpy()
        mp, am = te_map(ktr, tg[tn], apt_tr)
        if key == 'cell' and CELLFB in ('nb', 'nbaug', 'nbaugf'):
            Tte[name], lv_te = te_apply_fb(kte, apt_te, mp, am, cell_fb_table(ktr, tg[tn], apt_tr))
            if LVL:
                Tte['te_cell_lvl'] = lv_te
            force = np.zeros(tr.height, bool)
            p_apt = {}
            if CELLFB in ('nbaug', 'nbaugf'):  # cell dropping: at the rate of missing own cells among the gated te rows of each airport (label-free)
                for x in set(apt_te[~np.isnan(lv_te)]) - CELLFB_EXCL:
                    mx = (apt_te == x) & ~np.isnan(lv_te)
                    p_apt[x] = float((lv_te[mx] > 0).mean())
                u = np.random.default_rng(CELLFB_SEED).random(tr.height)
                force = np.array([k is not None and u_ < p_apt.get(a_, 0.0) for k, a_, u_ in zip(ktr, apt_tr, u)])
            CELL_LOG.append(dict(n_te=int(len(kte)), aug_rate={x: round(p, 4) for x, p in sorted(p_apt.items())},
                                 n_force=int(force.sum()), te_lvl={x: [int(((apt_te == x) & (lv_te == L)).sum()) for L in range(4)]
                                                                  for x in sorted(set(apt_te[~np.isnan(lv_te)]))}))
            for j in range(inner):
                tj = np.where(fin != j, tg[tn], np.nan)
                mp, am = te_map(ktr, tj, apt_tr)
                s = fin == j
                Ttr[name][s], lv = te_apply_fb(ktr[s], apt_tr[s], mp, am, cell_fb_table(ktr, tj, apt_tr), force=force[s])
                if LVL:
                    Ttr['te_cell_lvl'][s] = lv
            continue
        Tte[name] = te_apply(kte, apt_te, mp, am)
        if key == 'cell' and CELLFB == 'pred' and PRED_FB:  # test side only; rows whose own cell is found / LSZH / empty key are bit-identical to te_apply
            fbv, lv_te = te_apply_fb(kte, apt_te, mp, am, cell_fb_table(ktr, tg[tn], apt_tr))
            CELL_LOG.append(dict(n_te=int(len(kte)), n_changed=int((~((fbv == Tte[name]) | (np.isnan(fbv) & np.isnan(Tte[name])))).sum()),
                                 te_lvl={x: [int(((apt_te == x) & (lv_te == L)).sum()) for L in range(4)]
                                         for x in sorted(set(apt_te[~np.isnan(lv_te)]))}))
            Tte[name] = fbv
        if key == 'stand_key' and STANDFB == 'pred' and PRED_FB and name not in DROP:  # test side only; stands in the pool are bit-identical
            Tte[name] = stand_fb(kte, apt_te, mp, am, ktr, tg[tn], Tte[name])
        for j in range(inner):
            mp, am = te_map(ktr, np.where(fin != j, tg[tn], np.nan), apt_tr)
            Ttr[name][fin == j] = te_apply(ktr[fin == j], apt_tr[fin == j], mp, am)
    Xtr, Xte = matrix(tr, Ttr), matrix(te, Tte)
    r = (tr['TAXITIME_SEC_mvt'].to_numpy().astype(float) - tr['base'].to_numpy())
    out = np.zeros(te.height)
    imp = np.zeros(len(FEATS))
    for s in seeds:
        m = _train(dict(PARAMS, seed=s), Xtr, r, w, rounds)
        out += m.predict(Xte) / len(seeds)
        imp += m.feature_importance('gain')
    return out, dict(zip(FEATS, (imp / imp.sum()).round(4)))


GENERIC0 = ['base', 'hour', 'mvt_aobt3', 'mvt_delay', 'flt_null', 'dep10', 'arr15', 'synth', 'd_move', 'd_first',
            'd_push'] + [c for c in AB if c not in ('ab_first_lat', 'ab_first_lon')] + \
           (ART if 'art' in EXTRA else []) + (WX if 'wx' in EXTRA else []) + (FLOW if 'flow' in EXTRA else []) + \
           (AIR if 'air' in EXTRA else [])
GENERIC = GENERIC0 + TKC  # tk columns are appended to the end of the generic-model features


def fit_predict_generic(tr, te, rounds=600, seeds=(0,), w=None):
    """Generic residual model without airport-specific information (airport code / coordinates / grid and stand encodings), for airports without ground samples in 2025 (LEMD/LIRF etc.)."""
    Xtr = tr.select(GENERIC).to_numpy().astype(np.float32)
    Xte = te.select(GENERIC).to_numpy().astype(np.float32)
    r = tr['TAXITIME_SEC_mvt'].to_numpy().astype(float) - tr['base'].to_numpy()
    out = np.zeros(te.height)
    for s in seeds:
        m = _train(dict(PARAMS, seed=s), Xtr, r, w, rounds)
        out += m.predict(Xte) / len(seeds)
    return out


def fit_mixed(trv, te, mode=None, **kw):
    """Prediction logic: airports whose training gating rate is >= FULL_THR use the full model, the others the generic model; with RESID_MONTH != off the models are trained separately per month of te.
    trv has isA removed. Returns (residual prediction for te (not yet gated), per-group info)."""
    mode = mode or MONTH
    if mode.startswith('hyb'):
        months = mode.split(':')[1].split('+') if ':' in mode else ['01', '07']
        res, info = fit_mixed(trv, te, mode='off', **kw)
        mo_te, mo_tr = te['date'].str.slice(5, 2).to_numpy(), trv['date'].str.slice(5, 2).to_numpy()
        apt_te = te['ADEP_mvt'].to_numpy()
        for mm in months:
            tm = mo_te == mm
            if not tm.any():
                continue
            tr = trv.filter(pl.Series(mo_tr == mm))
            tg = {x: float(tr.filter(pl.col('ADEP_mvt') == x)['gate'].mean() or 0) for x in APTS}
            fa = [x for x in APTS if tg[x] >= FULL_THR]
            r_full, _ = fit_predict(tr, te.filter(pl.Series(tm)), **kw)
            res[tm] = np.where(np.isin(apt_te[tm], fa), r_full, res[tm])
            info[f'month_{mm}'] = dict(train_days=int(tr['date'].n_unique()), full_apts=fa,
                                       train_gate={x: round(tg[x], 3) for x in APTS})
        # January WINTER_APTS rows use the "January + winter pool" full model; RESID_WINTER_TEST can extend the same training set to February (fitted separately per test month; default only 01)
        for wmm in (WINTER_TEST if WINTER != 'off' and '01' in months else []):
            tm = (mo_te == wmm) & np.isin(apt_te, WINTER_APTS)
            if tm.any():
                assert POOL is not None, 'RESID_WINTER requires load_pool first'
                wm = WINTER.split('+')
                pw = POOL.filter(pl.col('date').str.slice(5, 2).is_in(wm))
                tr = pl.concat([trv.filter(pl.Series(mo_tr == '01')), pw], how='vertical_relaxed')
                res[tm], _ = fit_predict(tr, te.filter(pl.Series(tm)), **kw)
                info['winter' if wmm == '01' else f'winter_{wmm}'] = dict(
                    months=wm, apts=WINTER_APTS, rows=int(tm.sum()), pool_rows=pw.height,
                    train_rows=tr.height, pool_gate={x: int(((pw['ADEP_mvt'] == x) & pw['gate']).sum()) for x in WINTER_APTS})
        for s_ in GENSWAP:  # the given airport x months use the generic model (leave-one-airport-out: training excludes the airport itself)
            x, mm = s_.split('-')
            tm = (mo_te == mm) & (apt_te == x)
            if tm.any():
                tr = trv.filter(pl.col('gate') & (pl.col('ADEP_mvt') != x))
                res[tm] = fit_predict_generic(tr, te.filter(pl.Series(tm)), **kw)
                info[f'genswap_{s_}'] = dict(rows=int(tm.sum()), train_rows=tr.height)
        return res, info
    mo_te = te['date'].str.slice(5, 2).to_numpy()
    mo_tr = trv['date'].str.slice(5, 2).to_numpy()
    apt_te = te['ADEP_mvt'].to_numpy()
    res = np.zeros(te.height)
    info = {}
    for mm in (['all'] if mode == 'off' else sorted(set(mo_te))):
        tm = np.ones(te.height, bool) if mm == 'all' else mo_te == mm
        tr, w = trv, None
        if mode == 'only':
            tr = trv.filter(pl.Series(mo_tr == mm))
        elif mode != 'off':
            w = np.where(mo_tr == mm, float(mode[1:]), 1.0)
        tes = te.filter(pl.Series(tm))
        r_full, _ = fit_predict(tr, tes, w=w, **kw)
        g = tr['gate'].to_numpy()
        r_gen = fit_predict_generic(tr.filter(pl.col('gate')), tes, w=None if w is None else w[g], **kw)
        tg = {x: float(tr.filter(pl.col('ADEP_mvt') == x)['gate'].mean() or 0) for x in APTS}
        fa = [x for x in APTS if tg[x] >= FULL_THR]
        res[tm] = np.where(np.isin(apt_te[tm], fa), r_full, r_gen)
        info[mm] = dict(train_days=int(tr['date'].n_unique()), full_apts=fa, train_gate={x: round(tg[x], 3) for x in APTS})
    return res, info


NGF = ['base', 'apt_code', 'hour', 'mvt_aobt3', 'mvt_delay', 'flt_null', 'dep10', 'arr15', 'synth'] + \
      (ART if 'art' in EXTRA else []) + (FLOW if NG in ('flow', 'flowair') else []) + (AIR if NG in ('air', 'flowair') else [])


def fit_ng(trv, te, rounds=600, seeds=(0,)):
    """Residual model for non-gated rows (RESID_NG): target = residual minus the airport x month mean of the training rows, so it learns only the fluctuations with flow rate / time of day within the same airport and month."""
    tr = trv if NG_TRAIN == 'all' else trv.filter(~pl.col('gate'))
    r = tr['TAXITIME_SEC_mvt'].to_numpy().astype(float) - tr['base'].to_numpy()
    key = tr.select(pl.concat_str([pl.col('ADEP_mvt'), pl.col('date').str.slice(5, 2)]))[:, 0].to_numpy()
    mu = pl.DataFrame({'k': key, 'r': r}).group_by('k').agg(pl.col('r').mean())
    r = r - pl.DataFrame({'k': key}).join(mu, on='k', how='left')['r'].to_numpy()
    Xtr = tr.select(NGF).to_numpy().astype(np.float32)
    Xte = te.select(NGF).to_numpy().astype(np.float32)
    out = np.zeros(te.height)
    for s in seeds:
        m = _train(dict(PARAMS, seed=s), Xtr, r, None, rounds)
        out += m.predict(Xte) / len(seeds)
    return out


def fit_all(trv, te, **kw):
    """fit_mixed (gated rows) + RESID_NG (non-gated rows). Returns (residual of the gated-row model (not yet gated), residual of the non-gated-row model or None, info)."""
    res, info = fit_mixed(trv, te, **kw)
    r_ng = fit_ng(trv, te, **kw) if NG != 'off' else None
    info['ng'] = dict(mode=NG, train=NG_TRAIN, feats=NGF) if r_ng is not None else None
    return res, r_ng, info


def loao(v, **kw):
    """Leave-one-airport-out: only on airports with ground coverage; a generic model is trained on the other airports and tested on this airport (gated rows)."""
    y = v['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    out = {}
    for a in ['EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LSZH']:
        tr = v.filter((pl.col('ADEP_mvt') != a) & ~pl.col('isA') & pl.col('gate'))
        m = (v['ADEP_mvt'] == a).to_numpy() & ~v['isA'].to_numpy() & v['gate'].to_numpy()
        te = v.filter(pl.Series(m))
        res = fit_predict_generic(tr, te, **kw)
        b = te['base'].to_numpy()
        yy = y[m]
        out[a] = dict(n=int(m.sum()), base=round(float(np.sqrt(np.mean((b - yy) ** 2))), 1),
                      loao_generic=round(float(np.sqrt(np.mean((b + res - yy) ** 2))), 1))
        print(a, out[a], flush=True)
    return out


def cv(v, nfold=5, seed=0, **kw):
    days = sorted(v['date'].unique().to_list())
    perm = np.random.default_rng(seed).permutation(len(days))
    fold_of = {days[i]: k % nfold for k, i in enumerate(perm)}
    fd = np.array([fold_of[d] for d in v['date'].to_list()])
    res = np.zeros(v.height)
    isA = v['isA'].to_numpy()
    imps = []
    for k in range(nfold):
        tr = v.filter(pl.Series((fd != k) & ~isA))
        te_mask = fd == k
        res[te_mask], imp = fit_predict(tr, v.filter(pl.Series(te_mask)), **kw)
        imps.append(imp)
    return res, {f: round(float(np.mean([i[f] for i in imps])), 4) for f in FEATS}


def full_apts(v):
    """Airports with a training gating rate >= 30% are full-model airports."""
    t = v.filter(~pl.col('isA')).group_by('ADEP_mvt').agg(pl.col('gate').mean())
    return [a for a, g in t.iter_rows() if g >= 0.3]


def report(v, res):
    y = v['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    b = v['base'].to_numpy()
    nA = ~v['isA'].to_numpy()
    gate = v['gate'].to_numpy()
    apt_ = v['ADEP_mvt'].to_numpy()
    full = np.isin(apt_, full_apts(v))
    preds = {'all': b + res, 'gated': b + res * gate, 'fullall': b + res * (gate | full)}
    mse = lambda p, m: float(np.mean((p[m] - y[m]) ** 2))
    out = dict(n_rows=int(len(y)), n_nonA=int(nA.sum()), days=int(v['date'].n_unique()), gate_cover=float(gate.mean()),
               base_nonA_mse=mse(b, nA))
    out['base_nonA_rmse'] = float(np.sqrt(out['base_nonA_mse']))
    for k, p in preds.items():
        out[f'{k}_nonA_mse'] = mse(p, nA)
        out[f'{k}_rel_sse_red'] = 1 - out[f'{k}_nonA_mse'] / out['base_nonA_mse']
        out[f'{k}_nonA_rmse'] = float(np.sqrt(out[f'{k}_nonA_mse']))
    apt = v['ADEP_mvt'].to_numpy()
    p = preds['gated']
    out['apt'] = {a: dict(n=int(((apt == a) & nA).sum()), gate=round(float(gate[apt == a].mean()), 3),
                          base=round(np.sqrt(mse(b, (apt == a) & nA)), 1), new=round(np.sqrt(mse(p, (apt == a) & nA)), 1),
                          sse_share_red=round((mse(b, (apt == a) & nA) - mse(p, (apt == a) & nA)) * ((apt == a) & nA).sum()
                                              / (out['base_nonA_mse'] * nA.sum()), 4))
                  for a in APTS}
    return out


def main():
    global POOL, STAND_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--storedir', required=True)
    ap.add_argument('--feats', required=True)
    ap.add_argument('--eval', action='store_true')
    ap.add_argument('--predict', action='store_true')
    ap.add_argument('--loao', action='store_true')
    ap.add_argument('--tag')
    ap.add_argument('--out', default='resid_eval.json')
    ap.add_argument('--out-dir', default='sub')
    ap.add_argument('--rounds', type=int, default=600)
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--mode', default='gated', choices=['gated', 'fullall'])
    a = ap.parse_args()
    v = load_train(a.storedir, a.feats)
    kw = dict(rounds=a.rounds, seeds=tuple(range(a.seeds)))
    if a.loao:
        json.dump(loao(v, **kw), open('loao.json', 'w'), indent=1)
    if a.eval:
        res, imp = cv(v, **kw)
        rep = report(v, res)
        rep['importance_top'] = sorted(imp.items(), key=lambda x: -x[1])[:20]
        print(json.dumps(rep, indent=1))
        json.dump(rep, open(a.out, 'w'), indent=1)
        np.save(Path(a.out).with_suffix('.oof.npy'), res)
    if a.predict:
        if RANK_FEATS:  # swap only the ADS-B feature directory of the test rows; training is unchanged
            sd, STAND_DIR = STAND_DIR, RANK_FEATS
            rk = load_rank(a.storedir, RANK_FEATS)
            STAND_DIR = sd
        else:
            rk = load_rank(a.storedir, a.feats)
        trv = v.filter(~pl.col('isA'))
        if WINTER != 'off':
            POOL, _ = load_pool(a.storedir, v, WINTER.split('+'))
        if CELLFB == 'pred' or STANDFB == 'pred':
            (res0, rng0, g0), (res1, rng1, g1) = dual(fit_all, trv, rk, **kw)
            runs = [(f'{a.tag}_off', res0, rng0, g0), (a.tag, res1, rng1, g1)]
        else:
            runs = [(a.tag, *fit_all(trv, rk, **kw))]
        for tag, res, r_ng, groups in runs:
            write_sub(a, v, trv, rk, tag, res, r_ng, groups)


def write_sub(a, v, trv, rk, tag, res, r_ng, groups):
    """Write one prediction file + .res.npy + .json (with CELLFB=pred main calls this twice: <tag>_off and <tag>)."""
    tr_gate = {x: float(trv.filter(pl.col('ADEP_mvt') == x)['gate'].mean() or 0) for x in APTS}
    fa = sorted(set(sum([g['full_apts'] for g in groups.values() if g and 'full_apts' in g], [])))
    apt = rk['ADEP_mvt'].to_numpy()
    gate = rk['gate'].to_numpy()
    if a.mode == 'fullall':  # all rows of full-model airports are corrected, other airports only rows with a ground segment (not supported with per-month assignment)
        assert MONTH == 'off'  # noqa
        gate = gate | np.isin(apt, fa)
    res = res * gate if r_ng is None else np.where(gate, res, r_ng)
    p = np.clip(rk['base'].to_numpy() + res, 0, None)
    sub = pl.DataFrame({'MVT_ID_mvt': rk['MVT_ID_mvt'], 'TAXITIME_SEC_mvt': np.round(p).astype(np.int32)})
    Path(a.out_dir).mkdir(exist_ok=True)
    fn = Path(a.out_dir) / (f'upstanding-firefly_v{tag}.parquet' if STAND == 'off' else f'upstanding-firefly_{tag}.parquet')
    sub.write_parquet(fn)
    np.save(fn.with_suffix('.res.npy'), res)
    info = dict(rows=sub.height, gate_cover=float(gate.mean()), days_2026=int(rk.filter(pl.col('gate'))['date'].n_unique()),
                full_model_apts=fa, mode=a.mode, train_gate=tr_gate, month_split=MONTH, full_thr=FULL_THR,
                groups=groups, extra=EXTRA, flow_mincov=FLOW_MINCOV if USE_FLOW else None,
                air_mincov=AIR_MINCOV if USE_AIR else None, ng=NG, stand=STAND,
                cellfb=CELLFB, cellfb_excl=sorted(CELLFB_EXCL), articao=ARTICAO, cell_log=CELL_LOG,
                winter=WINTER, winter_apts=WINTER_APTS, winter_test=WINTER_TEST, base_sub=BASE_SUB, n26=N26, pool_feats=POOL_FEATS if WINTER != 'off' else None, genswap=GENSWAP,
                standfb=STANDFB, standfb_r=STANDFB_R, stand_log=STAND_LOG, rank_feats=RANK_FEATS or None, nullab=NULLAB,
                gate_by_apt={x: round(float(gate[apt == x].mean()), 3) for x in APTS},
                n_by_apt={x: int((apt == x).sum()) for x in APTS},
                mean_change=float(np.mean(res)), rms_change=float(np.sqrt(np.mean(res ** 2))),
                rms_change_by_apt={x: round(float(np.sqrt(np.mean(res[apt == x] ** 2))), 1) for x in APTS},
                train_days=int(v['date'].n_unique()), rounds=a.rounds, seeds=a.seeds,
                tk=TK, tk_cols=TKC, tk_file=TK_FILE if TK_LOAD else None, tk_pool=TK_POOL_FILE if TK_LOAD and WINTER != 'off' else None,
                code_file=str(Path(__file__).resolve()))
    json.dump(info, open(fn.with_suffix('.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
