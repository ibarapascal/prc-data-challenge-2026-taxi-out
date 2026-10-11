"""schedule_match_classifier (output column pi_sched): binary classifier for "BLOCK ~ SCHED (+-60 s)" -- leave-one-month-out 12-fold OOF probabilities (training rows) + full-year model probabilities (evaluation rows), written back as the new column pi_sched of work/feat.parquet.
Rationale: two-fold AUC 0.876 / 0.897; blending it in post-processing did not help, so it is passed as a feature to the main model and the LIRF sub-model instead.
Memory: float32 matrix of all rows ~1.0 GB + fold copies ~0.9 GB + write-back of feat ~2.5 GB (released in stages), peak < 5 GB; 6 threads; 12 folds x ~45 s + 1 full-year fit ~ 12 minutes.
Usage: python schedule_match_classifier.py <work dir> [--threads 6] [--rounds 250]    (the work directory of baseline_models.py, which holds feat.parquet)
Output: work/feat.parquet (column pi_sched added; the original file is backed up to work/feat_before_pi.parquet on the first run), work/pi_sched.npy (aligned with the feat row order), work/pi_sched_meta.json (AUC / positive rate per fold)
Nested mode: --outer MO [MO ...] -- for each outer month MO, the other 11 training months are cross-fitted leave-one-month-out internally to obtain pi,
  and pi for month MO and the evaluation rows is fitted on these 11 months only; written to work/pi_sched_o<MO>.npy (aligned with all rows of feat), **feat.parquet is not modified**. baseline_models.py reads it with PI_NESTED=1 (not adopted).
  Each outer month takes ~ 12 fits ~ the same time (about 7 minutes) and memory as the default mode.
"""
import os, sys, json, time, shutil, argparse, numpy as np, polars as pl, lightgbm as lgb
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
ap = argparse.ArgumentParser(); ap.add_argument('work'); ap.add_argument('--threads', type=int, default=6); ap.add_argument('--rounds', type=int, default=250); ap.add_argument('--outer', type=int, nargs='*', default=None); a = ap.parse_args()
FEAT = os.path.join(a.work, 'feat.parquet'); BAK = os.path.join(a.work, 'feat_before_pi.parquet')
def log(*x): print(*x, flush=True)
CAT0 = ['ADEP_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','ADES_mvt','AIRCRAFT_OPERATOR_flt','WK_TBL_CAT_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt']
BASE = ['hour','dow','mod','mvt_delay','mvt_aobt3','mvt_eobt1','aobt_eobt','flt_null']
CONG = ['dep10','dep30','arr15','rwy_gap_prev','rwy_gap_next','rwy_dep10','sched30','aobt_rank10']
NEW5 = ['overtake','overtake_rwy','turn_aobt','turn_land','same_type']
P15 = ['mvt_sec','mvt_sec0','aobt_sec0','arvt3_sec0','sched_sec','mvt_lobt','mvt_iobt','lobt_eobt1','aobt_lobt','arvt1_eobt1','arvt3_mvt','m1_m3_dur','month']
SYN = ['synth','ma_dev','synth_share','arr_taxi_med','arr_taxi_med60','arr20','rwy_dep20_fwd']
TRANS = ['cs_prev_ae','cs_next_ae','day_null_rate','day_n','stand_hour_n','day_ma_dev_med']
MET = ['tc','sknt','vsby','wx_frz','wx_rain','wx_fog','deice_risk']
STAND = ['mvt_inblk','inblk_sched','eobt_inblk','aobt_inblk','arr_taxi_in_prev','prev_same_al','prev_same_type','arr_fltnull_prev','next_inblk_mvt','next_inblk_sched','arr_blk_eq_sched_prev']
REGIME = ['ex','rwy_ex5_m','rwy_ex5_n','rwy_ex15_m','rwy_ex15_n','rwy_ex60_m','rwy_ex60_n','rwy_ae15_m','rwy_syn15_m','apt_ex15_m','apt_ex15_n','al_ex60_m','al_ex60_n','ex_prev','ex_next','ae_prev','ae_next','same_al_prev','arr_ex5_m','arr_ex5_n']
CAT = CAT0 + ['al']; SR = CAT0 + BASE + CONG + NEW5 + ['al'] + P15 + SYN + TRANS + MET + STAND + REGIME
t0 = time.time()
if not os.path.exists(BAK): shutil.copy(FEAT, BAK); log('backup ->', BAK)
d = pl.read_parquet(FEAT, columns=SR + ['ym', 'src', 'TAXITIME_SEC_mvt', 'mvt_sched'])
if 'pi_sched' in pl.read_parquet_schema(FEAT): log('feat already has pi_sched; will overwrite the column')
n = d.height; ym = d['ym'].to_numpy(); src = d['src'].to_numpy(); tr = src != 99
y = d['TAXITIME_SEC_mvt'].to_numpy().astype(float); ms = np.nan_to_num(d['mvt_sched'].to_numpy().astype(float), nan=-1e9)
lab = (np.abs(ms - y) <= 60).astype(int); lab[~tr] = -1
Xdf = d.select(SR).to_pandas(); del d
X = np.empty((n, len(SR)), dtype=np.float32)
for j, c in enumerate(SR):
    if c in CAT:
        codes = Xdf[c].astype('category').cat.codes.to_numpy().astype(np.float32); codes[codes < 0] = np.nan; X[:, j] = codes
    else: X[:, j] = Xdf[c].to_numpy(dtype=np.float32, na_value=np.nan)
del Xdf; log(f'X {X.shape} float32 {X.nbytes/1e9:.2f}GB, train rows {tr.sum():,}, pos rate {lab[tr].mean():.4f} ({time.time()-t0:.0f}s)')
cidx = [i for i, c in enumerate(SR) if c in CAT]
LP = dict(objective='binary', learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=a.threads, seed=0)
from sklearn.metrics import roc_auc_score
if a.outer:
    for MO in a.outer:
        assert 1 <= MO <= 12; hold = tr & (ym == 202500 + MO); T = tr & ~hold; pi = np.full(n, np.nan); meta = {}; t2 = time.time()
        for mo in range(1, 13):
            if mo == MO: continue
            te = T & (ym == 202500 + mo); fit = T & ~te; t1 = time.time()
            m = lgb.train(LP, lgb.Dataset(X[fit], lab[fit], feature_name=SR, categorical_feature=cidx, free_raw_data=True), a.rounds)
            pi[te] = m.predict(X[te]); auc = roc_auc_score(lab[te], pi[te]); meta[f'inner{mo}'] = dict(auc=round(float(auc), 4), n=int(te.sum())); log(f'  outer {MO:2d} inner {mo:2d}: AUC {auc:.4f} ({time.time()-t1:.0f}s)'); del m
        m = lgb.train(LP, lgb.Dataset(X[T], lab[T], feature_name=SR, categorical_feature=cidx, free_raw_data=True), a.rounds)
        rest = hold | ~tr; pi[rest] = m.predict(X[rest]); auc = roc_auc_score(lab[hold], pi[hold]); meta['outer'] = dict(month=MO, auc=round(float(auc), 4), n=int(hold.sum()), mean_pi_hold=round(float(pi[hold].mean()), 4), mean_pi_inner=round(float(pi[T].mean()), 4)); del m
        assert not np.isnan(pi).any(); np.save(os.path.join(a.work, f'pi_sched_o{MO}.npy'), pi); json.dump(meta, open(os.path.join(a.work, f'pi_sched_o{MO}_meta.json'), 'w'), indent=1)
        log(f'outer {MO}: hold AUC {auc:.4f} inner AUCs', {k: v['auc'] for k, v in meta.items() if k.startswith('inner')}, f'({time.time()-t2:.0f}s) -> pi_sched_o{MO}.npy')
    log(f'nested DONE ({time.time()-t0:.0f}s); feat.parquet untouched'); sys.exit(0)
pi = np.full(n, np.nan); meta = {}
for mo in range(1, 13):
    te = tr & (ym == 202500 + mo); fit = tr & ~te; t1 = time.time()
    m = lgb.train(LP, lgb.Dataset(X[fit], lab[fit], feature_name=SR, categorical_feature=cidx, free_raw_data=True), a.rounds)
    pi[te] = m.predict(X[te]); auc = roc_auc_score(lab[te], pi[te]); meta[f'fold{mo}'] = dict(auc=round(float(auc), 4), n=int(te.sum()), pos=round(float(lab[te].mean()), 4))
    log(f'  month {mo:2d}: AUC {auc:.4f} n {te.sum():,} ({time.time()-t1:.0f}s)'); del m
m = lgb.train(LP, lgb.Dataset(X[tr], lab[tr], feature_name=SR, categorical_feature=cidx, free_raw_data=True), a.rounds)
pi[~tr] = m.predict(X[~tr]); meta['rank'] = dict(n=int((~tr).sum()), mean_pi=round(float(pi[~tr].mean()), 4), mean_pi_train_oof=round(float(pi[tr].mean()), 4)); del m, X
assert not np.isnan(pi).any(); np.save(os.path.join(a.work, 'pi_sched.npy'), pi); json.dump(meta, open(os.path.join(a.work, 'pi_sched_meta.json'), 'w'), indent=1)
log('OOF AUC by month:', {k: v['auc'] for k, v in meta.items() if k.startswith('fold')}, 'rank mean pi', meta['rank'])
full = pl.read_parquet(FEAT)
if 'pi_sched' in full.columns: full = full.drop('pi_sched')
full = full.with_columns(pl.Series('pi_sched', pi.astype(np.float32))); tmp = FEAT + '.tmp'; full.write_parquet(tmp); os.replace(tmp, FEAT)
log(f'wrote {FEAT} with pi_sched ({full.height:,} rows, {len(full.columns)} cols) DONE ({time.time()-t0:.0f}s)')
