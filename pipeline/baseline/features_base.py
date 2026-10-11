"""Experiment: redo of the congestion features + comparison of loss functions / clipping. Two folds (validation November and December, trained on the first two months each).
Usage: python features_base.py <raw_dir>. Memory <= 1 GB. All features use only columns that the DEP rows of ranking also have.
"""
import sys, os, time
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR','load_as_storage')
import polars as pl, numpy as np, lightgbm as lgb
raw = sys.argv[1] if len(sys.argv)>1 else 'raw'
MONTHS = {9:'2025-09-01_2025-10-01',10:'2025-10-01_2025-11-01',11:'2025-11-01_2025-12-01',12:'2025-12-01_2026-01-01'}
CAT = ['ADEP_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','ADES_mvt','AIRCRAFT_OPERATOR_flt','WK_TBL_CAT_flt','MARKET_SEGMENT_flt','FLIGHT_TYPE_flt']
BASE = ['hour','dow','mod','mvt_delay','mvt_aobt3','mvt_eobt1','aobt_eobt','flt_null']
CONG = ['dep10','dep30','arr15','rwy_gap_prev','rwy_gap_next','rwy_dep10','sched30','aobt_rank10']

def feats(df, canonical=False):
    t = pl.col('MVT_TIME_UTC_mvt')
    df = df.with_columns(
        t.dt.hour().alias('hour'), t.dt.weekday().alias('dow'),
        ((t.dt.hour().cast(pl.Int16) if canonical else t.dt.hour())*60+t.dt.minute()).alias('mod'),
        (t-pl.col('SCHED_TIME_UTC_mvt')).dt.total_seconds().alias('mvt_delay'),
        (t-pl.col('AOBT_3_flt')).dt.total_seconds().alias('mvt_aobt3'),
        (t-pl.col('EOBT_1_flt')).dt.total_seconds().alias('mvt_eobt1'),
        (pl.col('AOBT_3_flt')-pl.col('EOBT_1_flt')).dt.total_seconds().alias('aobt_eobt'),
        pl.col('AOBT_3_flt').is_null().cast(pl.Int8).alias('flt_null'),
        pl.when(pl.col('PHASE_mvt')=='DEP').then(pl.col('ADEP_mvt')).otherwise(pl.col('ADES_mvt')).alias('apt'),
    ).sort('apt','MVT_TIME_UTC_mvt')
    # rolling counts at the same airport: departures in the 10/30 minutes before take-off, arrivals in the 15 minutes before (MVT time, available for both row types in ranking)
    isdep = (pl.col('PHASE_mvt')=='DEP').cast(pl.Int32)
    roll = df.rolling(index_column='MVT_TIME_UTC_mvt', period='10m', group_by='apt').agg(isdep.sum().alias('dep10'))
    roll30 = df.rolling(index_column='MVT_TIME_UTC_mvt', period='30m', group_by='apt').agg(isdep.sum().alias('dep30'))
    arr15 = df.rolling(index_column='MVT_TIME_UTC_mvt', period='15m', group_by='apt').agg((1-isdep).sum().alias('arr15'))
    df = df.with_columns(roll['dep10'], roll30['dep30'], arr15['arr15'])
    dep = df.filter(pl.col('PHASE_mvt')=='DEP').sort(['ADEP_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt']+(['MVT_ID_mvt'] if canonical else []))
    # gap to the previous/next take-off on the same runway, departures on the runway in the previous 10 minutes
    dep = dep.with_columns(
        (pl.col('MVT_TIME_UTC_mvt')-pl.col('MVT_TIME_UTC_mvt').shift(1)).dt.total_seconds().over('ADEP_mvt','RUNWAY_mvt').alias('rwy_gap_prev'),
        (pl.col('MVT_TIME_UTC_mvt').shift(-1)-pl.col('MVT_TIME_UTC_mvt')).dt.total_seconds().over('ADEP_mvt','RUNWAY_mvt').alias('rwy_gap_next'),
    )
    r10 = dep.rolling(index_column='MVT_TIME_UTC_mvt', period='10m', group_by=['ADEP_mvt','RUNWAY_mvt']).agg(pl.len().alias('rwy_dep10'))
    dep = dep.with_columns(r10['rwy_dep10'])
    # planned congestion: departures at the same airport within SCHED +-15 minutes (pure prior)
    s = dep.sort('ADEP_mvt','SCHED_TIME_UTC_mvt')
    s30 = s.rolling(index_column='SCHED_TIME_UTC_mvt', period='30m', offset='-15m', group_by='ADEP_mvt').agg(pl.len().alias('sched30'))
    s = s.with_columns(s30['sched30'])
    # pushbacks at the same airport within 10 minutes after pushback (AOBT_3 queue)
    a = s.filter(pl.col('AOBT_3_flt').is_not_null()).sort('ADEP_mvt','AOBT_3_flt')
    a10 = a.rolling(index_column='AOBT_3_flt', period='10m', group_by='ADEP_mvt').agg(pl.len().alias('aobt_rank10'))
    a = a.with_columns(a10['aobt_rank10']).select('MVT_ID_mvt','aobt_rank10')
    out=s.join(a, on='MVT_ID_mvt', how='left')
    return out.sort('MVT_ID_mvt') if canonical else out

def load(m): return feats(pl.read_parquet(f'{raw}/training_{MONTHS[m]}.parquet'))

def to_X(df, cols, cats=None):
    X = df.select(cols).to_pandas()
    for c in CAT:
        if c in X: X[c] = X[c].astype('category') if cats is None else X[c].astype('category').cat.set_categories(cats[c])
    return X

def rmse(p,y): return float(np.sqrt(np.mean((p-y)**2)))

def fit(Xtr, ytr, Xva, yva, obj='regression', clip=None, alpha=None, rounds=3000):
    y = ytr if clip is None else np.clip(ytr, 0, clip)
    params = dict(objective=obj, metric='rmse', learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
    if alpha: params['alpha']=alpha
    cats=[c for c in CAT if c in Xtr]
    m = lgb.train(params, lgb.Dataset(Xtr, y, categorical_feature=cats), num_boost_round=rounds,
                  valid_sets=[lgb.Dataset(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, m.predict(Xva, num_iteration=m.best_iteration)

if __name__=='__main__':
    results=[]
    for val in (11,12):
        t0=time.time()
        tr = pl.concat([load(val-2), load(val-1)], how='vertical_relaxed'); va = load(val)
        ytr = tr['TAXITIME_SEC_mvt'].to_numpy().astype(float); yva = va['TAXITIME_SEC_mvt'].to_numpy().astype(float)
        print(f'\n=== fold val={val}: train {tr.height:,} val {va.height:,} feat {time.time()-t0:.0f}s')
        for name, cols in [('base', CAT+BASE), ('base+cong', CAT+BASE+CONG)]:
            Xtr = to_X(tr, cols); cats={c:Xtr[c].cat.categories for c in CAT}; Xva = to_X(va, cols, cats)
            for lname, kw in [('L2 clip3600', dict(clip=3600)), ('L2 noclip', dict()), ('L2 clip7200', dict(clip=7200)),
                              ('huber a600', dict(obj='huber', alpha=600)), ('huber a1500', dict(obj='huber', alpha=1500))]:
                if name=='base' and lname not in ('L2 clip3600',): continue
                t0=time.time(); m,p = fit(Xtr,ytr,Xva,yva,**kw)
                r = rmse(p,yva); r_n = float(np.sqrt(np.mean(((p-yva)**2)[yva<=3600])))
                # post-processing: add the mean training residual per airport (mean correction for RMSE)
                ptr = m.predict(Xtr, num_iteration=m.best_iteration)
                bias = tr.with_columns(pl.Series('res', ytr-ptr)).group_by('ADEP_mvt').agg(pl.col('res').mean().alias('b'))
                pb = p + va.join(bias,on='ADEP_mvt',how='left')['b'].fill_null(0).to_numpy()
                results.append((val,name,lname,round(r,1),round(r_n,1),round(rmse(pb,yva),1),m.best_iteration,round(time.time()-t0)))
                print(f'{name:10s} {lname:12s} RMSE {r:6.1f}  RMSE(<=1h) {r_n:6.1f}  +apt_bias {rmse(pb,yva):6.1f}  iter {m.best_iteration} {time.time()-t0:.0f}s')
            if name=='base+cong':
                imp = sorted(zip(m.feature_name(), m.feature_importance('gain')), key=lambda x:-x[1])
                print('  gain:', [(n,int(v/1e6)) for n,v in imp[:14]])
                pa = va.with_columns(pl.Series('p',p)).group_by('ADEP_mvt').agg(((pl.col('p')-pl.col('TAXITIME_SEC_mvt'))**2).mean().sqrt().round(0).alias('rmse')).sort('rmse')
                print('  per-airport:', dict(zip(pa['ADEP_mvt'], pa['rmse'])))
    print('\n=== summary (val, feats, loss, RMSE, RMSE<=1h, +bias, iter, s)')
    for r in results: print(r)
