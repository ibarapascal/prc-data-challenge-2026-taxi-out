"""Baseline modelling script (iteration framework; 6 threads, memory < 6 GB). Adds to the earlier framework:
  P0-11 evaluation protocol: record() also logs non-extreme RMSE/MSE and slices of disturbed days; the stage report pools the two folds and computes nonA_RMSE = sqrt(weighted non-extreme MSE)
  P0-12 stand occupancy: the feature store derives in_block_prev/next for all DEP rows, including unmatched ones (STAND); the LIRF sub-model can use stand features (env LIRFSTAND=1)
  P0-13 local regime: excess of mvt_aobt3 among same-runway neighbours within +-5/15/60 minutes (self excluded), the previous/next aircraft, same-airport/same-airline windows, ARR side +-5 minutes (REGIME)
  Minor: empty STAND values are filled with 'NA'; stage_featset defaults to GRID[--grid] (default 1 = 255 leaves, min_data 50)
FEATSETS: stand = metar+STAND; regime = metar+REGIME; sr = metar+STAND+REGIME. The work directory should not be shared with the earlier framework (OOF files are not aligned).
FEATSET srp = sr + pi_sched (the column is written into feat.parquet by schedule_match_classifier.py); with env PI=1 the LIRF sub-model's LSTAND also gets pi_sched; fullens env NSEED (several seeds averaged for LightGBM members, default 1) and CATSEED (several CatBoost seeds, default 1).
Later additions: (1) the feat stage splits each file by year-month before computing feats2 (fixes the cross-month overtake offset in the July part of the ranking file; FEAT_SPLIT=0 restores the old behaviour); (2) after fit_lirf fixes the number of rounds, it refits on the full B subset (LIRF_REFIT=0 restores the old behaviour);
  (3) with PI_NESTED=1, fold(mo) uses work/pi_sched_o<mo>.npy (nested pi, outer-month labels isolated; not adopted); (4) ens objective: ENS_OBJ=mean (default; arithmetic mean of the two-fold RMSE, used for the submitted baseline and set explicitly by runner/run_baseline.py) or ENS_OBJ=proxy (pooled two-fold non-extreme RMSE; optional, not adopted).

Usage: python baseline_models.py <raw_dir> <work_dir> <stage...> [--threads 6]
stages: feat (feature store: 12 training months plus the evaluation set, written to work/feat.parquet)
       base / p15 / p01a / p05 / metar (incremental feature sets, two folds = January/July, component table)
       p01b (physical sub-model for synthetic-AOBT and unmatched rows, override) p03 (class-C anchor two-stage model) p02 (deeper class-B sub-model + isotonic)
       seeds (best feature set x several seeds x two targets, writes OOF) ens (Caruana hill climbing) full (full-year fit that writes the prediction file, no upload)
Output: work/results.tsv (one experiment per row: name, fold, full RMSE, RMSE without >3h rows, component JSON), work/oof/<exp>_<fold>.npy
Component definitions: A y > 3h (not B); B LIRF unmatched MVT-SCHED > 1h (rule set); C matched delta > 20 min; D matched regular; E unmatched regular.
"""
import sys, os, time, json, argparse
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR','load_as_storage')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl, numpy as np, lightgbm as lgb
from features_base import CAT as CAT0, BASE, CONG, rmse
from features_extended import feats2, NEW, P
# Import-safe so other scripts can reuse the exact feature builder without
# parsing another command's argv, creating caches, or dispatching training stages.
a=None
sec=lambda x,y:(pl.col(x)-pl.col(y)).dt.total_seconds()
CAT=CAT0+['al']
COLS0=CAT0+BASE+CONG+NEW[:5]+['al']
P15=['mvt_sec','mvt_sec0','aobt_sec0','arvt3_sec0','sched_sec','mvt_lobt','mvt_iobt','lobt_eobt1','aobt_lobt','arvt1_eobt1','arvt3_mvt','m1_m3_dur','month']
SYN=['synth','ma_dev','synth_share','arr_taxi_med','arr_taxi_med60','arr20','rwy_dep20_fwd']
TRANS=['cs_prev_ae','cs_next_ae','day_null_rate','day_n','stand_hour_n','day_ma_dev_med']
MET=['tc','sknt','vsby','wx_frz','wx_rain','wx_fog','deice_risk']
STAND=['mvt_inblk','inblk_sched','eobt_inblk','aobt_inblk','arr_taxi_in_prev','prev_same_al','prev_same_type','arr_fltnull_prev','next_inblk_mvt','next_inblk_sched','arr_blk_eq_sched_prev']
REGIME=['ex','rwy_ex5_m','rwy_ex5_n','rwy_ex15_m','rwy_ex15_n','rwy_ex60_m','rwy_ex60_n','rwy_ae15_m','rwy_syn15_m','apt_ex15_m','apt_ex15_n','al_ex60_m','al_ex60_n','ex_prev','ex_next','ae_prev','ae_next','same_al_prev','arr_ex5_m','arr_ex5_n']
MSET=COLS0+P15+SYN+TRANS+MET
FEATSETS={'base':COLS0,'p15':COLS0+P15,'p01a':COLS0+P15+SYN,'p05':COLS0+P15+SYN+TRANS,'metar':MSET,'stand':MSET+STAND,'regime':MSET+REGIME,'sr':MSET+STAND+REGIME,'srp':MSET+STAND+REGIME+['pi_sched']}
FEAT=None; RES=None
def configure(raw,work,threads=6,tag='ensa',grid=1):
    global a,FEAT,RES,DF
    a=argparse.Namespace(raw=str(raw),work=str(work),threads=threads,tag=tag,grid=grid)
    FEAT=os.path.join(a.work,'feat.parquet'); RES=os.path.join(a.work,'results.tsv'); DF=None
def log(*x): print(*x,flush=True)

# ---------- Feature store ----------
def metar_table():
    d=os.path.join(a.raw,'..','ext','metar')
    if not os.path.isdir(d): return None
    fs=[pl.read_csv(os.path.join(d,f),null_values=['M'],schema_overrides={k:pl.Float64 for k in ['tmpf','dwpf','relh','drct','sknt','gust','vsby','p01i','skyl1']}|{'wxcodes':pl.Utf8,'metar':pl.Utf8,'skyc1':pl.Utf8}) for f in os.listdir(d) if f.endswith('.csv')]
    m=pl.concat(fs,how='diagonal_relaxed').with_columns(pl.col('valid').str.strptime(pl.Datetime('us'),'%Y-%m-%d %H:%M').dt.replace_time_zone('UTC').alias('t'),((pl.col('tmpf')-32)*5/9).alias('tc'))
    w=pl.col('wxcodes').fill_null('')
    m=m.with_columns(w.str.contains('SN|FZ|IC|PL|SG').cast(pl.Int8).alias('wx_frz'),w.str.contains('RA|DZ|SH|TS').cast(pl.Int8).alias('wx_rain'),w.str.contains('FG|BR|HZ').cast(pl.Int8).alias('wx_fog'))
    return m.select(pl.col('station').alias('ADEP_mvt'),'t','tc','sknt','vsby','wx_frz','wx_rain','wx_fog').sort('t')
def add_stand(df,arrst):
    """P0-12 stand occupancy: for every DEP row (all rows), the most recent arrival at the same airport and stand before this movement (backward) and the first arrival after it (forward)."""
    prev=arrst.select('ADEP_mvt','STAND_mvt',pl.col('in_block').alias('in_block_prev'),pl.col('land').alias('land_prev'),pl.col('arr_al').alias('arr_al_prev'),pl.col('arr_type').alias('arr_type_prev'),pl.col('arr_fltnull').alias('arr_fltnull_prev'),pl.col('arr_blk_eq_sched').alias('arr_blk_eq_sched_prev'))
    nxt=arrst.select('ADEP_mvt','STAND_mvt',pl.col('in_block').alias('in_block_next'))
    df=df.sort('MVT_TIME_UTC_mvt').join_asof(prev,left_on='MVT_TIME_UTC_mvt',right_on='in_block_prev',by=['ADEP_mvt','STAND_mvt'],strategy='backward')
    df=df.join_asof(nxt,left_on='MVT_TIME_UTC_mvt',right_on='in_block_next',by=['ADEP_mvt','STAND_mvt'],strategy='forward')
    df=df.with_columns(sec('MVT_TIME_UTC_mvt','in_block_prev').alias('mvt_inblk'),sec('in_block_prev','SCHED_TIME_UTC_mvt').alias('inblk_sched'),sec('EOBT_1_flt','in_block_prev').alias('eobt_inblk'),sec('AOBT_3_flt','in_block_prev').alias('aobt_inblk'),
        sec('in_block_prev','land_prev').alias('arr_taxi_in_prev'),(pl.col('arr_al_prev')==pl.col('al')).cast(pl.Int8).alias('prev_same_al'),(pl.col('arr_type_prev')==pl.col('AIRCRAFT_TYPE_mvt')).cast(pl.Int8).alias('prev_same_type'),
        sec('in_block_next','MVT_TIME_UTC_mvt').alias('next_inblk_mvt'),sec('in_block_next','SCHED_TIME_UTC_mvt').alias('next_inblk_sched'))
    return df.drop('in_block_prev','land_prev','arr_al_prev','arr_type_prev','in_block_next')
def win(d,keys,lo,hi,vals):
    """Deterministic window statistics (polars rolling with a multi-key group_by does not return rows in input order; only 47% were aligned in a test with polars 1.44, hence numpy).
    Window = rows of the same group with MVT in (t+lo, t+hi], self excluded. vals={name:col}: returns name_s (sum, nulls ignored) and name_c (non-null count); with col=None only the row count name_c. The result is aligned with the rows of d."""
    needed=list(dict.fromkeys(keys+['MVT_TIME_UTC_mvt']+[c for c in vals.values() if c is not None]))
    d=d.select(needed).with_row_index('_rid'); s=d.sort(keys+['MVT_TIME_UTC_mvt'])
    gid=s.select(pl.concat_str([pl.col(k).cast(pl.Utf8).fill_null('~') for k in keys],separator='|').rle_id().alias('g'))['g'].to_numpy().astype(np.int64)
    t=s['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy().astype(np.int64); tg=gid*(1<<40)+t
    L=np.searchsorted(tg,tg+lo,'right'); H=np.searchsorted(tg,tg+hi,'right'); incl=(lo<0)&(hi>=0); rid=s['_rid'].to_numpy(); out={}
    def scat(x): o=np.empty(len(x),x.dtype); o[rid]=x; return o
    for name,col in vals.items():
        if col is None:
            c=(H-L)-(1 if incl else 0); out[name+'_c']=scat(c.astype(np.int32)); continue
        x=s[col].to_numpy().astype(float); nn=~np.isnan(x); x0=np.where(nn,x,0.0)
        cs=np.concatenate([[0.0],np.cumsum(x0)]); cn=np.concatenate([[0],np.cumsum(nn.astype(np.int64))])
        sm=cs[H]-cs[L]-(x0 if incl else 0); c=cn[H]-cn[L]-(nn.astype(np.int64) if incl else 0)
        out[name+'_s']=scat(sm); out[name+'_c']=scat(c.astype(np.int32))
    return out
def _mean(o,name,dst):
    c=o[name+'_c']; return pl.Series(dst+'_m',np.where(c>0,o[name+'_s']/np.maximum(c,1),np.nan)).fill_nan(None),pl.Series(dst+'_n',c)
def add_regime(df,arr,canonical=False):
    """P0-13 local regime: ex = mvt_aobt3 - ref (20th percentile by airport, stand, runway; falls back to airport, runway when < 30 rows), only for non-synthetic matched rows; neighbour-window means exclude self.
    Also recomputes rwy_dep10 (same runway, previous 10 minutes) and rwy_dep20_fwd (same runway, next 20 minutes) with deterministic windows, overriding the misaligned versions from the earlier feature scripts."""
    valid=(pl.col('flt_null')==0)&(pl.col('synth')==0)&pl.col('mvt_aobt3').is_not_null()
    v=df.select('ADEP_mvt','STAND_mvt','RUNWAY_mvt','mvt_aobt3','flt_null','synth').filter(valid)
    r1=v.group_by('ADEP_mvt','STAND_mvt','RUNWAY_mvt').agg(pl.col('mvt_aobt3').quantile(0.2).alias('_q1'),pl.len().alias('_n1'))
    r2=v.group_by('ADEP_mvt','RUNWAY_mvt').agg(pl.col('mvt_aobt3').quantile(0.2).alias('_q2'))
    del v
    df=df.join(r1,on=['ADEP_mvt','STAND_mvt','RUNWAY_mvt'],how='left').join(r2,on=['ADEP_mvt','RUNWAY_mvt'],how='left')
    del r1,r2
    df=df.with_columns(pl.when(pl.col('_n1')>=30).then(pl.col('_q1')).otherwise(pl.col('_q2')).alias('_ref'))
    df=df.with_columns(pl.when(valid).then((pl.col('mvt_aobt3')-pl.col('_ref')).clip(-900,3600)).otherwise(None).alias('ex'),pl.col('aobt_eobt').clip(-1800,7200).alias('_ae'),pl.col('synth').cast(pl.Float64).alias('_sy')).drop('_q1','_n1','_q2','_ref')
    RK=['ADEP_mvt','RUNWAY_mvt']; cols=[]
    for w,h in (('5',300),('15',900),('60',3600)):
        o=win(df,RK,-h,h,{'ex':'ex'}); cols+=list(_mean(o,'ex',f'rwy_ex{w}'))
    o=win(df,RK,-900,900,{'ae':'_ae','sy':'_sy'}); cols+=[_mean(o,'ae','rwy_ae15')[0],_mean(o,'sy','rwy_syn15')[0]]
    o=win(df,RK,-600,0,{'n':None}); cols.append(pl.Series('rwy_dep10',o['n_c']))
    o=win(df,RK,0,1200,{'n':None}); cols.append(pl.Series('rwy_dep20_fwd',o['n_c']))
    o=win(df,['ADEP_mvt'],-900,900,{'ex':'ex'}); cols+=list(_mean(o,'ex','apt_ex15'))
    o=win(df,['ADEP_mvt','al'],-3600,3600,{'ex':'ex'}); cols+=list(_mean(o,'ex','al_ex60'))
    df=df.drop('rwy_dep10','rwy_dep20_fwd').with_columns(cols)
    # previous/next aircraft on the same runway (in MVT order)
    df=df.sort(['ADEP_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt']+(['MVT_ID_mvt'] if canonical else [])).with_columns(pl.col('ex').shift(1).over(RK).alias('ex_prev'),pl.col('ex').shift(-1).over(RK).alias('ex_next'),pl.col('_ae').shift(1).over(RK).alias('ae_prev'),pl.col('_ae').shift(-1).over(RK).alias('ae_next'),(pl.col('al').shift(1).over(RK)==pl.col('al')).cast(pl.Int8).alias('same_al_prev')).drop('_ae','_sy')
    # ARR side +-5 minutes: mean taxi-in excess (excess = taxi_in - airport median) and arrival count
    am=arr.group_by('ADEP_mvt').agg(pl.col('taxi_in').median().alias('_am')); ar=arr.join(am,on='ADEP_mvt').with_columns((pl.col('taxi_in')-pl.col('_am')).alias('_aex'))
    o=win(ar,['ADEP_mvt'],-300,300,{'aex':'_aex'}); m,n=_mean(o,'aex','arr_ex5'); ar=ar.with_columns(m,n).select('ADEP_mvt','MVT_TIME_UTC_mvt','arr_ex5_m','arr_ex5_n')
    # Duplicate ARR timestamps have different leave-self-out means. An arbitrary
    # unique(last) depends on partition/hash-join order; use their visible mean.
    ar=(ar.group_by('ADEP_mvt','MVT_TIME_UTC_mvt').agg(pl.col('arr_ex5_m').mean(),pl.col('arr_ex5_n').first()) if canonical else ar.unique(['ADEP_mvt','MVT_TIME_UTC_mvt'],keep='last')).sort('MVT_TIME_UTC_mvt')
    df=df.sort('MVT_TIME_UTC_mvt').join_asof(ar,on='MVT_TIME_UTC_mvt',by='ADEP_mvt',strategy='nearest',tolerance='10m')
    return df
def build_feat(canonical=False,airport=None):
    files=sorted(f for f in os.listdir(a.raw) if f.startswith('training_'))[:int(os.environ.get('NFILES','99'))]+['ranking.parquet']
    parts=[]; arrs=[]; arrst=[]
    for i,f in enumerate(files):
        scan=pl.scan_parquet(os.path.join(a.raw,f))
        if airport is not None:
            scan=scan.filter(((pl.col('PHASE_mvt')=='DEP')&(pl.col('ADEP_mvt')==airport))|((pl.col('PHASE_mvt')=='ARR')&(pl.col('ADES_mvt')==airport)))
        raw=scan.collect().with_columns(pl.when(pl.col('PHASE_mvt')=='DEP').then(pl.col('STAND_mvt').fill_null('NA')).otherwise(pl.col('STAND_mvt')).alias('STAND_mvt'))
        if not raw.height: continue
        # Fix: the feats2 rank features accumulate over a whole file, and ranking.parquet holds two months (January/July), so July overtake counts also carried the January count of rows lacking AOBT.
        # Training files are one month each; here every file is split by year-month before computing, so that one feats2 call = one month. env FEAT_SPLIT=0 restores the old behaviour (for comparison).
        chunks=[raw] if os.environ.get('FEAT_SPLIT','1')=='0' else raw.with_columns((pl.col('MVT_TIME_UTC_mvt').dt.year()*100+pl.col('MVT_TIME_UTC_mvt').dt.month()).alias('_ym')).partition_by('_ym',maintain_order=True)
        if len(chunks)>1: log(f'  {f}: split into {len(chunks)} month chunks before feats2')
        for ch in chunks:
            if '_ym' in ch.columns: ch=ch.drop('_ym')
            d=feats2(ch,canonical=canonical).with_columns(pl.col('FLIGHT_mvt').str.extract(r'^([A-Z]+)').alias('al'), sec('AOBT_3_flt','BLOCK_TIME_UTC_mvt').alias('delta'), pl.lit(i if f!='ranking.parquet' else 99).cast(pl.Int8).alias('src'))
            parts.append(d)
        arrs.append(raw.filter(pl.col('PHASE_mvt')=='ARR').select(pl.col('ADES_mvt').alias('ADEP_mvt'),'MVT_TIME_UTC_mvt',sec('BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt').clip(0,3600).alias('taxi_in')))
        arrst.append(raw.filter(pl.col('PHASE_mvt')=='ARR').select(pl.col('ADES_mvt').alias('ADEP_mvt'),'STAND_mvt',pl.col('BLOCK_TIME_UTC_mvt').alias('in_block'),pl.col('MVT_TIME_UTC_mvt').alias('land'),pl.col('FLIGHT_mvt').str.extract(r'^([A-Z]+)').alias('arr_al'),pl.col('AIRCRAFT_TYPE_mvt').alias('arr_type'),pl.col('AOBT_3_flt').is_null().cast(pl.Int8).alias('arr_fltnull'),(sec('BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt').abs()<120).cast(pl.Int8).alias('arr_blk_eq_sched')).drop_nulls(['STAND_mvt','in_block']))
    df=pl.concat(parts,how='vertical_relaxed'); arr=pl.concat(arrs).drop_nulls().sort('ADEP_mvt','MVT_TIME_UTC_mvt'); arrst=pl.concat(arrst)
    if canonical: arrst=arrst.sort(['ADEP_mvt','STAND_mvt','in_block','land','arr_al','arr_type','arr_fltnull','arr_blk_eq_sched'],nulls_last=True)
    arrst=arrst.unique(['ADEP_mvt','STAND_mvt','in_block'],keep='first' if canonical else 'any',maintain_order=canonical).sort('in_block'); arr0=arr
    del parts,arrs,raw,chunks,ch,d
    t=pl.col('MVT_TIME_UTC_mvt')
    df=df.with_columns(t.dt.second().alias('mvt_sec'),(t.dt.second()==0).cast(pl.Int8).alias('mvt_sec0'),(pl.col('AOBT_3_flt').dt.second()==0).cast(pl.Int8).alias('aobt_sec0'),(pl.col('ARVT_3_flt').dt.second()==0).cast(pl.Int8).alias('arvt3_sec0'),
        pl.col('SCHED_TIME_UTC_mvt').dt.second().alias('sched_sec'),sec('MVT_TIME_UTC_mvt','LOBT_flt').alias('mvt_lobt'),sec('MVT_TIME_UTC_mvt','IOBT_flt').alias('mvt_iobt'),sec('LOBT_flt','EOBT_1_flt').alias('lobt_eobt1'),sec('AOBT_3_flt','LOBT_flt').alias('aobt_lobt'),
        sec('ARVT_1_flt','EOBT_1_flt').alias('arvt1_eobt1'),sec('ARVT_3_flt','MVT_TIME_UTC_mvt').alias('arvt3_mvt'),(sec('ARVT_1_flt','EOBT_1_flt')-sec('ARVT_3_flt','AOBT_3_flt')).alias('m1_m3_dur'),
        t.dt.month().cast(pl.Int8).alias('month'),(t.dt.year()*100+t.dt.month()).alias('ym'),t.dt.date().alias('day'),
        sec('AOBT_3_flt','SCHED_TIME_UTC_mvt').alias('aobt_sched'),sec('MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt').alias('mvt_sched'))
    # synthetic AOBT: mode over 60-second bins and its share by (airport, runway, year-month), training + evaluation pooled
    df=df.with_columns((pl.col('mvt_aobt3')/60).floor().alias('_b'))
    g=df.drop_nulls('_b').group_by('ADEP_mvt','RUNWAY_mvt','ym','_b').agg(pl.len().alias('_n'))
    tot=df.drop_nulls('_b').group_by('ADEP_mvt','RUNWAY_mvt','ym').agg(pl.len().alias('_t'))
    gs=g.sort(['_n','_b'],descending=[True,False]) if canonical else g.sort('_n',descending=True)
    mode=gs.unique(['ADEP_mvt','RUNWAY_mvt','ym'],keep='first',maintain_order=True).join(tot,on=['ADEP_mvt','RUNWAY_mvt','ym']).select('ADEP_mvt','RUNWAY_mvt','ym',(pl.col('_b')*60).alias('_mode'),(pl.col('_n')/pl.col('_t')).alias('synth_share'))
    df=df.join(mode,on=['ADEP_mvt','RUNWAY_mvt','ym'],how='left').with_columns((pl.col('mvt_aobt3')-pl.col('_mode')).alias('ma_dev'))
    df=df.with_columns(((pl.col('ma_dev')>=-30)&(pl.col('ma_dev')<90)&(pl.col('synth_share')>0.25)).fill_null(False).cast(pl.Int8).alias('synth')).drop('_b','_mode')
    del g,tot,gs,mode
    # rolling median of arrival taxi-in (+-15 minutes, previous 60 minutes) and arrival count in the previous 20 minutes, asof-joined to DEP by airport
    r=arr.rolling(index_column='MVT_TIME_UTC_mvt',period='30m',offset='-15m',group_by='ADEP_mvt').agg(pl.col('taxi_in').median().alias('arr_taxi_med'),pl.len().alias('_c'))
    r60=arr.rolling(index_column='MVT_TIME_UTC_mvt',period='60m',group_by='ADEP_mvt').agg(pl.col('taxi_in').median().alias('arr_taxi_med60'))
    r20=arr.rolling(index_column='MVT_TIME_UTC_mvt',period='20m',group_by='ADEP_mvt').agg(pl.len().alias('arr20'))
    arr=arr.with_columns(r['arr_taxi_med'],r60['arr_taxi_med60'],r20['arr20']).select('ADEP_mvt','MVT_TIME_UTC_mvt','arr_taxi_med','arr_taxi_med60','arr20').unique(['ADEP_mvt','MVT_TIME_UTC_mvt'],keep='last').sort('MVT_TIME_UTC_mvt')
    df=df.sort('MVT_TIME_UTC_mvt').join_asof(arr,on='MVT_TIME_UTC_mvt',by='ADEP_mvt',strategy='nearest',tolerance='30m')
    del r,r60,r20,arr
    # departures in the 20 minutes after, same runway (forward-looking)
    d2=df.select('MVT_ID_mvt','ADEP_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt').sort('ADEP_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt')
    fwd=d2.rolling(index_column='MVT_TIME_UTC_mvt',period='20m',offset='0s',group_by=['ADEP_mvt','RUNWAY_mvt']).agg(pl.len().alias('rwy_dep20_fwd'))
    fwd=d2.select('MVT_ID_mvt').with_columns((fwd['rwy_dep20_fwd']-1).alias('rwy_dep20_fwd'))
    df=df.join(fwd,on='MVT_ID_mvt',how='left')
    del d2,fwd
    # transductive: previous/next flight with the same flight number (aobt_eobt), share of unmatched rows that day at the airport, number of movements that day, stand x hour counts, same-day median of ma_dev
    df=df.sort(['FLIGHT_mvt','MVT_TIME_UTC_mvt']+(['MVT_ID_mvt'] if canonical else [])).with_columns(pl.col('aobt_eobt').shift(1).over('FLIGHT_mvt','ADEP_mvt').alias('cs_prev_ae'),pl.col('aobt_eobt').shift(-1).over('FLIGHT_mvt','ADEP_mvt').alias('cs_next_ae'))
    df=df.with_columns(pl.col('flt_null').mean().over('ADEP_mvt','day').alias('day_null_rate'),pl.len().over('ADEP_mvt','day').alias('day_n'),pl.len().over('ADEP_mvt','STAND_mvt','day','hour').alias('stand_hour_n'),pl.col('ma_dev').median().over('ADEP_mvt','day').alias('day_ma_dev_med'))
    m=metar_table()
    if m is not None:
        df=df.sort('MVT_TIME_UTC_mvt').join_asof(m,left_on='MVT_TIME_UTC_mvt',right_on='t',by='ADEP_mvt',strategy='backward',tolerance='2h').with_columns(((pl.col('tc')<3)&((pl.col('wx_frz')==1)|(pl.col('vsby')<2))).cast(pl.Int8).alias('deice_risk')).drop('t')
    else:
        df=df.with_columns([pl.lit(None,pl.Float64).alias(c) for c in MET])
    del m
    df=add_stand(df,arrst); df=add_regime(df,arr0,canonical=canonical)
    del arrst,arr0
    keep=list(dict.fromkeys(FEATSETS['sr']+['MVT_ID_mvt','TAXITIME_SEC_mvt','delta','src','ym','day','aobt_sched','mvt_sched','mvt_lobt','mvt_iobt','flt_null','mvt_aobt3','mvt_delay','mvt_eobt1','ADEP_mvt','FLIGHT_mvt','ADES_mvt','RUNWAY_mvt','STAND_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','BLOCK_TIME_UTC_mvt','EOBT_1_flt','LOBT_flt','IOBT_flt','AIRCRAFT_TYPE_mvt']))
    miss=[c for c in keep if c not in df.columns]; log('missing feat cols:',miss)
    df=df.select([c for c in keep if c in df.columns]).sort(['src','MVT_TIME_UTC_mvt']+(['MVT_ID_mvt'] if canonical else []))
    df.write_parquet(FEAT); log(f'feat store {df.height:,} rows x {df.width} cols -> {FEAT}')
    log(df.group_by('src').agg(pl.len(),pl.col('synth').mean().round(3).alias('synth_rate')).sort('src'))
    log(df.filter(pl.col('src')==99).group_by('ADEP_mvt').agg(pl.len(),pl.col('synth').mean().round(3).alias('synth_rate'),pl.col('synth_share').mean().round(2)).sort('ADEP_mvt'))

# ---------- Component evaluation ----------
def components(va,p):
    y=va['TAXITIME_SEC_mvt'].to_numpy().astype(float); r=y-p; tot=float((r**2).sum())
    v=va.with_columns(pl.Series('r',r)).with_columns(
        pl.when((pl.col('flt_null')==1)&(pl.col('ADEP_mvt')=='LIRF')&(pl.col('mvt_delay')>3600)).then(pl.lit('B')).when(pl.col('TAXITIME_SEC_mvt')>10800).then(pl.lit('A')).when(pl.col('delta')>1200).then(pl.lit('C')).when(pl.col('flt_null')==1).then(pl.lit('E')).otherwise(pl.lit('D')).alias('cls'))
    agg=lambda g:g.agg(pl.len().alias('n'),((pl.col('r')**2).sum()/tot*100).round(1).alias('sse'),(pl.col('r')**2).mean().sqrt().round(0).alias('rmse'),pl.col('r').mean().round(0).alias('bias'))
    out={'rmse':round(rmse(p,y),2),'rmse_no3h':round(float(np.sqrt(np.mean((r**2)[y<=10800]))),2)}
    nonA=v.select(pl.col('cls')!='A').to_numpy().ravel(); out['nonA_rmse']=round(float(np.sqrt(np.mean(r[nonA]**2))),2); out['nonA_mse']=round(float(np.mean(r[nonA]**2)),1); out['nonA_n']=int(nonA.sum())
    dist=v.with_columns(((pl.col('aobt_sched')>1200).cast(pl.Float64)).mean().over('ADEP_mvt','day').alias('_ds')).select(pl.col('_ds')>0.3).to_numpy().ravel()&nonA
    out['dist_rmse']=round(float(np.sqrt(np.mean(r[dist]**2))),2) if dist.sum() else None; out['dist_n']=int(dist.sum())
    dd_=v.filter(pl.col('cls')=='D'); out['D_apt']={k:int(vv) for k,vv in dd_.group_by('ADEP_mvt').agg((pl.col('r')**2).mean().sqrt().round(0)).sort('ADEP_mvt').iter_rows()}
    out['cls']={k:dict(zip(['n','sse','rmse','bias'],vals)) for k,*vals in agg(v.group_by('cls')).sort('cls').iter_rows()}
    dd=v.filter(pl.col('cls').is_in(['D','E']))
    out['apt']={k:dict(zip(['n','sse','rmse','bias'],vals)) for k,*vals in agg(dd.group_by('ADEP_mvt')).sort('ADEP_mvt').iter_rows()}
    out['synth']={str(k):dict(zip(['n','sse','rmse','bias'],vals)) for k,*vals in agg(dd.group_by('synth')).sort('synth').iter_rows()}
    return out
def record(exp,fold,va,p,extra=None):
    c=components(va,p); np.save(os.path.join(a.work,'oof',f'{exp}_{fold}.npy'),p)
    with open(RES,'a') as f: f.write(f"{exp}\t{fold}\t{c['rmse']}\t{c['rmse_no3h']}\t{json.dumps(c)}\t{json.dumps(extra or {})}\n")
    log(f"[{exp} fold{fold}] RMSE {c['rmse']} no3h {c['rmse_no3h']} nonA {c['nonA_rmse']} dist {c['dist_rmse']}(n={c['dist_n']}) cls {json.dumps(c['cls'])}")
    log('   D by apt:',c['D_apt'])
    log('   D/E synth:',json.dumps(c['synth']),' apt rmse:',{k:v['rmse'] for k,v in c['apt'].items()})
    return c

# ---------- Data and models ----------
DF=None
def data():
    global DF
    if DF is None:
        DF=pl.read_parquet(FEAT)
        # LOWMEM=1 (default) casts Float64 -> Float32 and Int64 -> Int32 (accepted natively by LightGBM and CatBoost); the transient peak of the two-fold srp stage (10.1-10.5 GB) tripped the 9.5 GB memory guard twice. LOWMEM=0 restores the old dtypes.
        if os.environ.get('LOWMEM','1')=='1':
            i64=[c for c,t in DF.schema.items() if t==pl.Int64 and c!='MVT_ID_mvt']; mm=DF.select([pl.col(c).min().alias(c+'_lo') for c in i64]+[pl.col(c).max().alias(c+'_hi') for c in i64]).row(0,named=True)
            ok=[c for c in i64 if mm[c+'_lo'] is not None and mm[c+'_lo']>-2**31 and mm[c+'_hi']<2**31-1]
            DF=DF.with_columns([pl.col(c).cast(pl.Float32) for c,t in DF.schema.items() if t==pl.Float64 and c!='MVT_ID_mvt']+[pl.col(c).cast(pl.Int32) for c in ok])  # MVT_ID_mvt holds large Float64 integers; float32 would lose precision; log(f'  LOWMEM: float64->f32, int64->i32 ({len(ok)}/{len(i64)} cols), est {DF.estimated_size()/1e9:.2f}GB')
        log(f'loaded feat {DF.height:,}')
    return DF
def fold(mo):
    full=data(); ymv=202500+mo
    # With PI_NESTED=1, replace the pi_sched column by work/pi_sched_o<mo>.npy (written by schedule_match_classifier.py --outer mo, aligned with all rows of feat) --
    # so the labels of the outer validation month no longer enter the pi of the training months. Stops with an error if the file is missing instead of silently falling back to the leaking version. with_columns only creates one new column (the rest are shared), not a full copy.
    if os.environ.get('PI_NESTED')=='1' and 'pi_sched' in full.columns:
        fp=os.path.join(a.work,f'pi_sched_o{mo}.npy'); assert os.path.exists(fp),f'PI_NESTED=1 but {fp} missing'
        pi=np.load(fp); assert len(pi)==full.height,(len(pi),full.height); full=full.with_columns(pl.Series('pi_sched',pi.astype(np.float32))); del pi; log(f'  fold {mo}: pi_sched <- {os.path.basename(fp)} (nested)')
    # lower memory: no intermediate copy of all training rows, filter tr/va directly
    trm=(pl.col('src')!=99)
    return full.filter(trm&(pl.col('ym')!=ymv)), full.filter(trm&(pl.col('ym')==ymv))
def to_X(df,cols,cats=None):
    X=df.select(cols).to_pandas()
    for c in [c for c in cols if c in CAT]: X[c]=X[c].astype('category') if cats is None else X[c].astype('category').cat.set_categories(cats[c])
    return X
def hp(**kw): return dict(P,objective='regression',metric='rmse',num_threads=a.threads,**kw)
def _f32(X):
    """pandas (including category columns) -> float32 numpy; category columns use their codes, -1 -> NaN (same semantics as LightGBM's pandas path, but without a float64 copy). Lower memory."""
    A=np.empty(X.shape,dtype=np.float32)
    for j,c in enumerate(X.columns):
        if str(X[c].dtype)=='category': v=X[c].cat.codes.to_numpy().astype(np.float32); v[v<0]=np.nan; A[:,j]=v
        else: A[:,j]=X[c].to_numpy(dtype=np.float32,na_value=np.nan)
    return A
class _M:
    """Wrapper around the model returned by fit_reg: predict takes a DataFrame (compatible with the old call sites) and runs on float32 numpy internally; other attributes pass through to the Booster."""
    def __init__(s,m): s.m=m
    def predict(s,X,**kw): return s.m.predict(_f32(X) if hasattr(X,'columns') else X,**kw)
    def __getattr__(s,k): return getattr(s.m,k)
def fit_reg(Xtr,ytr,Xva=None,yva=None,params=None,rounds=4000,cats=None):
    params=params or hp(); cs=[c for c in Xtr.columns if c in CAT]; fn=list(Xtr.columns)
    if os.environ.get('LOWMEM','1')=='0':
        if Xva is not None:
            m=lgb.train(params,lgb.Dataset(Xtr,ytr,categorical_feature=cs),rounds,valid_sets=[lgb.Dataset(Xva,yva)],callbacks=[lgb.early_stopping(100,verbose=False)]); return m,m.predict(Xva,num_iteration=m.best_iteration),m.best_iteration
        m=lgb.train(params,lgb.Dataset(Xtr,ytr,categorical_feature=cs),rounds); return m,None,rounds
    dtr=lgb.Dataset(_f32(Xtr),ytr,feature_name=fn,categorical_feature=cs,free_raw_data=True)
    if Xva is not None:
        Av=_f32(Xva); m=lgb.train(params,dtr,rounds,valid_sets=[lgb.Dataset(Av,yva,reference=dtr)],callbacks=[lgb.early_stopping(100,verbose=False)]); return _M(m),m.predict(Av,num_iteration=m.best_iteration),m.best_iteration
    m=lgb.train(params,dtr,rounds); return _M(m),None,rounds
# LIRF class-B sub-model (recalibrated version from the full-fit script)
LCAT=['al','ADES_mvt','RUNWAY_mvt','STAND_mvt']; LNUM=['mvt_delay','sh','mh','dow2','fnum']; LSTAND=['mvt_inblk','inblk_sched','arr_taxi_in_prev','prev_same_al','prev_same_type','arr_fltnull_prev']+(['pi_sched'] if os.environ.get('PI')=='1' else [])
LIRFSTAND=os.environ.get('LIRFSTAND','0')=='1'
def lirf_frame(df):
    return df.with_columns(pl.col('FLIGHT_mvt').str.extract(r'(\d+)').cast(pl.Int32,strict=False).alias('fnum'),pl.col('SCHED_TIME_UTC_mvt').dt.hour().alias('sh'),pl.col('MVT_TIME_UTC_mvt').dt.hour().alias('mh'),pl.col('SCHED_TIME_UTC_mvt').dt.weekday().alias('dow2'))
def lirf_X(df,cols,cats=None):
    x=df.select(cols).to_pandas()
    for c in [c for c in cols if c in LCAT]: x[c]=x[c].astype('category') if cats is None else x[c].astype('category').cat.set_categories(cats[c])
    return x
BMASK=lambda df:df.select((pl.col('flt_null')==1)&(pl.col('ADEP_mvt')=='LIRF')&(pl.col('mvt_delay')>3600)).to_numpy().ravel()
LP=dict(objective='binary',learning_rate=0.05,num_leaves=15,min_data_in_leaf=10,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,verbose=-1,num_threads=4,cat_smooth=5)
def fit_lirf(tr,deep=False,iso=False,stand=None):
    """Returns predict(df_sub) -> p_final. deep: adds OOF flight-number and airline x destination defect rates; iso: isotonic calibration."""
    from sklearn.isotonic import IsotonicRegression
    sub=lirf_frame(tr.filter(pl.Series(BMASK(tr)))); y=sub['TAXITIME_SEC_mvt'].to_numpy().astype(float); ms=sub['mvt_delay'].to_numpy().astype(float); lab=(y/ms>0.5).astype(int)
    stand=LIRFSTAND if stand is None else stand; cols=LCAT+LNUM+(LSTAND if stand and all(c in sub.columns for c in LSTAND) else []); rng=np.random.default_rng(1); fold5=rng.integers(0,5,len(y))
    if deep:
        sub=sub.with_columns(pl.Series('lab',lab),pl.Series('f5',fold5)); tes=[]
        for k in range(5):
            o=sub.filter(pl.col('f5')!=k); i=sub.filter(pl.col('f5')==k)
            for keys,nm in [(['FLIGHT_mvt'],'te_fl'),(['al','ADES_mvt'],'te_ald'),(['al'],'te_al')]:
                g=o.group_by(keys).agg(pl.col('lab').mean().alias('_m'),pl.len().alias('_n')); i=i.join(g,on=keys,how='left').with_columns(((pl.col('_m').fill_null(0.5)*pl.col('_n').fill_null(0)+0.5*3)/(pl.col('_n').fill_null(0)+3)).alias(nm)).drop('_m','_n')
            tes.append(i)
        sub=pl.concat(tes).sort('MVT_TIME_UTC_mvt'); y=sub['TAXITIME_SEC_mvt'].to_numpy().astype(float); ms=sub['mvt_delay'].to_numpy().astype(float); lab=sub['lab'].to_numpy(); fold5=sub['f5'].to_numpy(); cols=cols+['te_fl','te_ald','te_al']
        full=sub  # at apply time use the encoding of the whole subset
    X=lirf_X(sub,cols); cats={c:X[c].cat.categories for c in LCAT}
    idx=np.random.default_rng(0).permutation(len(y)); v=idx[:len(idx)//5]; t=idx[len(idx)//5:]
    c=lgb.train(LP,lgb.Dataset(X.iloc[t],lab[t],categorical_feature=LCAT),2000,valid_sets=[lgb.Dataset(X.iloc[v],lab[v])],callbacks=[lgb.early_stopping(50,verbose=False)]); it=max(c.best_iteration,20)
    # Early stopping only fixes the number of rounds; the prediction model is refit on the full B subset with that many rounds (the old version predicted with the 80% model and lost about 20% of rows). env LIRF_REFIT=0 restores the old behaviour.
    if os.environ.get('LIRF_REFIT','1')=='1': c=lgb.train(LP,lgb.Dataset(X,lab,categorical_feature=LCAT),it)
    oof=np.zeros(len(y))
    for k in range(5):
        ck=lgb.train(LP,lgb.Dataset(X.iloc[fold5!=k],lab[fold5!=k],categorical_feature=LCAT),it); oof[fold5==k]=ck.predict(X.iloc[fold5==k])
    cal=None
    if iso: cal=IsotonicRegression(out_of_bounds='clip').fit(oof,lab); oof=cal.predict(oof)
    A=np.vstack([oof*ms,1-oof,np.ones(len(y))]).T; coef=np.linalg.lstsq(A,y,rcond=None)[0]
    if len(y)<500 or not (0.7<coef[0]<1.3): coef=np.array([1.0,float(np.mean(y[lab==0])),0.0]); log('  lirf recal fallback')
    from sklearn.metrics import roc_auc_score; auc=roc_auc_score(lab,oof); oof_rmse=rmse(A@coef,y)
    log(f'  lirf sub-model deep={deep} iso={iso} stand={stand}: n={len(y)} def={lab.mean():.2f} iter={it} AUC(oof)={auc:.3f} oof-subset RMSE {oof_rmse:.0f} coef a={coef[0]:.2f} b={coef[1]:.0f} c={coef[2]:.0f}')
    def predict(df):
        s=lirf_frame(df)
        if deep:
            for keys,nm in [(['FLIGHT_mvt'],'te_fl'),(['al','ADES_mvt'],'te_ald'),(['al'],'te_al')]:
                g=full.group_by(keys).agg(pl.col('lab').mean().alias('_m'),pl.len().alias('_n')); s=s.join(g,on=keys,how='left').with_columns(((pl.col('_m').fill_null(0.5)*pl.col('_n').fill_null(0)+0.5*3)/(pl.col('_n').fill_null(0)+3)).alias(nm)).drop('_m','_n')
        pr=c.predict(lirf_X(s,cols,cats),num_iteration=it)
        if cal is not None: pr=cal.predict(pr)
        m_=s['mvt_delay'].to_numpy().astype(float); return coef[0]*pr*m_+coef[1]*(1-pr)+coef[2]
    return predict,dict(auc=round(auc,3),oof_rmse=round(oof_rmse,0))
def apply_lirf(df,p,predict):
    mk=BMASK(df); p=p.copy()
    if mk.sum(): p[mk]=predict(df.filter(pl.Series(mk)))
    return p
def pipeline(tr,va,cols,params=None,seed=0,lirf_kw={}):
    """taxi model + delta model averaged + LIRF blend. Returns (p_final, p_taxi, p_delta, iters)."""
    params=dict(params or hp(),seed=seed)
    Xtr=to_X(tr,cols); cats={c:Xtr[c].cat.categories for c in CAT if c in cols}; Xva=to_X(va,cols,cats)
    ytr=tr['TAXITIME_SEC_mvt'].to_numpy().astype(float); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    m,p,it=fit_reg(Xtr,ytr,Xva,yva,params)
    mt=tr['delta'].is_not_null().to_numpy(); mv=va['delta'].is_not_null().to_numpy()
    md,pd_,itd=fit_reg(Xtr[mt],tr['delta'].to_numpy().astype(float)[mt],Xva[mv],va['delta'].to_numpy().astype(float)[mv],params)
    p2=p.copy(); p2[mv]=va['mvt_aobt3'].to_numpy().astype(float)[mv]+pd_
    lp,linfo=fit_lirf(tr,**lirf_kw); p=apply_lirf(va,p,lp); p2=apply_lirf(va,p2,lp)
    log(f'  taxi {rmse(p,yva):.1f} delta {rmse(p2,yva):.1f} avg {rmse((p+p2)/2,yva):.1f} iters {it}/{itd}')
    return (p+p2)/2,p,p2,dict(it=it,itd=itd,**linfo)

# ---------- Stages ----------
def stage_featset(name,exp=None):
    cols=FEATSETS[name]; exp=exp or name; g=GRID[a.grid]
    for mo in [int(x) for x in os.environ.get('FOLDS','1,7').split(',')]:  # FOLDS=1 / FOLDS=7 can run as two separate processes to avoid stacking the peaks of both folds
        tr,va=fold(mo); t0=time.time(); p,p1,p2,info=pipeline(tr,va,cols,hp(**g))
        np.save(os.path.join(a.work,'oof',f'{exp}_taxi_{mo}.npy'),p1); np.save(os.path.join(a.work,'oof',f'{exp}_delta_{mo}.npy'),p2)
        record(exp,mo,va,p,dict(info,sec=round(time.time()-t0),ncol=len(cols),g=g,lirfstand=LIRFSTAND,fs=name,grid=a.grid))
        np.save(os.path.join(a.work,'oof',f'{exp}_{mo}.npy'),p)
        del tr,va,p,p1,p2; import gc; gc.collect()  # the previous fold's objects were not freed and stacked onto the start of the next fold (the start of fold 7 hit the memory guard at 10.2 GB)
def exp_meta(exp):
    """Takes featset/grid/lirfstand/iteration count of an experiment from results.tsv (two-fold mean x 1.15)."""
    rows=[json.loads(l.split('\t')[5]) for l in open(RES) if l.split('\t')[0]==exp]
    assert rows,f'no rows for {exp}'; r=rows[-1]; it=int(np.mean([x.get('it',0) for x in rows])*1.15); itd=int(np.mean([x.get('itd',0) for x in rows])*1.15)
    ALIAS={'stand_ls':'stand','m_g1':'metar'}; fs=r.get('fs') or ALIAS.get(exp,exp)
    return dict(fs=fs,grid=r.get('grid',a.grid),lirfstand=r.get('lirfstand',False),it=it,itd=itd)
def stage_lomo(exp):
    """P0-11: 12-fold leave-one-month-out OOF (members = experiments already run on the two folds); writes oof12/<exp>_{taxi,delta,avg}.npy, aligned with the training rows (src!=99); fixed iteration count (two-fold mean x 1.15), no early stopping."""
    m=exp_meta(exp); cols=FEATSETS[m['fs']]; g=GRID[m['grid']]; os.makedirs(os.path.join(a.work,'oof12'),exist_ok=True)
    d=data().filter(pl.col('src')!=99); ym=d['ym'].to_numpy(); y=d['TAXITIME_SEC_mvt'].to_numpy().astype(float); ma=d['mvt_aobt3'].to_numpy().astype(float); dl=d['delta'].to_numpy().astype(float)
    Xall=to_X(d,cols); cats={c:Xall[c].cat.categories for c in CAT if c in cols}; pt=np.zeros(d.height); pdl=np.zeros(d.height)
    for mo in range(1,13):
        te=ym==202500+mo; tr=~te; t0=time.time()
        mt=tr&~np.isnan(dl); mv=te&~np.isnan(dl)
        pt[te]=fit_reg(Xall[tr],y[tr],params=hp(**g),rounds=m['it'])[0].predict(Xall[te])
        pd_=fit_reg(Xall[mt],dl[mt],params=hp(**g),rounds=m['itd'])[0].predict(Xall[mv]); p2=pt[te].copy(); p2[mv[te]]=ma[mv]+pd_; pdl[te]=p2
        lp,_=fit_lirf(d.filter(pl.Series(tr)),stand=m['lirfstand']); sub=d.filter(pl.Series(te)); pt[te]=apply_lirf(sub,pt[te],lp); pdl[te]=apply_lirf(sub,pdl[te],lp)
        log(f'  lomo {exp} month {mo}: taxi {rmse(pt[te],y[te]):.1f} delta {rmse(pdl[te],y[te]):.1f} avg {rmse((pt[te]+pdl[te])/2,y[te]):.1f} {time.time()-t0:.0f}s')
    for nm,p in (('taxi',pt),('delta',pdl),('avg',(pt+pdl)/2)):
        np.save(os.path.join(a.work,'oof12',f'{exp}_{nm}.npy'),p); c=components(d,p); log(f'  lomo {exp} {nm} all-year RMSE {c["rmse"]} nonA {c["nonA_rmse"]} dist {c["dist_rmse"]} cls {json.dumps(c["cls"])}')
        with open(os.path.join(a.work,'results12.tsv'),'a') as f: f.write(f"{exp}_{nm}\t{c['rmse']}\t{c['nonA_rmse']}\t{json.dumps(c)}\n")
def stage_ens12():
    """Caruana fit on the 12-fold OOF (all-year training rows), members = oof12/*.npy; weights go to ens12_weights.json; also reports nonA."""
    dd=os.path.join(a.work,'oof12'); names=sorted(f[:-4] for f in os.listdir(dd) if f.endswith('.npy'))
    d=data().filter(pl.col('src')!=99); y=d['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    nonA=d.select(~(((pl.col('flt_null')==1)&(pl.col('ADEP_mvt')=='LIRF')&(pl.col('mvt_delay')>3600)).not_()&(pl.col('TAXITIME_SEC_mvt')>10800))).to_numpy().ravel()
    M={n:np.load(os.path.join(dd,n+'.npy')) for n in names}; mask=nonA if os.environ.get('ENS_NONA','1')=='1' else np.ones(len(y),bool)
    def score(w): p=sum(w[n]*M[n] for n in w)/sum(w.values()); return rmse(p[mask],y[mask])
    single={n:round(score({n:1}),2) for n in names}; log('members(12-fold,'+('nonA' if mask is nonA else 'all')+'):',single)
    w={min(single,key=single.get):1}; cur=score(w)
    for step in range(60):
        best=None
        for n in names:
            w2=dict(w); w2[n]=w2.get(n,0)+1; sc=score(w2)
            if best is None or sc<best[0]: best=(sc,n)
        if best[0]>=cur-0.003: break
        cur=best[0]; w[best[1]]=w.get(best[1],0)+1
    tot=sum(w.values()); wn={k:round(v/tot,3) for k,v in w.items()}; p=sum(w[n]*M[n] for n in w)/tot; c=components(d,p)
    log(f'ens12 RMSE {c["rmse"]} nonA {c["nonA_rmse"]} dist {c["dist_rmse"]} weights {wn} cls {json.dumps(c["cls"])}')
    json.dump(wn,open(os.path.join(a.work,'ens12_weights.json'),'w'))
ANCH={'eobt':'mvt_eobt1','sched':'mvt_sched','lobt':'mvt_lobt','iobt':'mvt_iobt'}
def stage_anchors(base):
    """P0-15 anchor-reparametrised target members: target d_k = y - (MVT-anchor_k) (= anchor_k - BLOCK), trained only on rows with a non-empty anchor; prediction = (MVT-anchor_k) + d_hat,
    rows with an empty anchor fall back to the OOF of the FALLBACK member (default <base>_taxi); LIRF blend as before. Member name anc_<k>."""
    cols=FEATSETS[base]; g=GRID[a.grid]; fb=os.environ.get('FALLBACK',f'{base}_taxi')
    for mo in (1,7):
        tr,va=fold(mo); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float); ytr=tr['TAXITIME_SEC_mvt'].to_numpy().astype(float)
        Xtr=to_X(tr,cols); cats={c:Xtr[c].cat.categories for c in CAT if c in cols}; Xva=to_X(va,cols,cats); pfb=np.load(os.path.join(a.work,'oof',f'{fb}_{mo}.npy')); lp,_=fit_lirf(tr)
        for k,col in ANCH.items():
            t0=time.time(); atr=tr[col].to_numpy().astype(float); ava=va[col].to_numpy().astype(float); mt=~np.isnan(atr); mv=~np.isnan(ava)
            dtr=ytr[mt]-atr[mt]; dva=yva[mv]-ava[mv]
            m,pd_,it=fit_reg(Xtr[mt],dtr,Xva[mv],dva,hp(**g)); p=pfb.copy(); p[mv]=ava[mv]+pd_; p=apply_lirf(va,p,lp)
            log(f'  anc_{k} fold{mo}: rows {mt.sum():,}/{mv.sum():,} iter {it} sub RMSE {rmse(p[mv],yva[mv]):.1f} (fallback {rmse(pfb[mv],yva[mv]):.1f}) {time.time()-t0:.0f}s')
            np.save(os.path.join(a.work,'oof',f'anc_{k}_{mo}.npy'),p); record(f'anc_{k}',mo,va,p,dict(it=it,itd=it,fs=base,grid=a.grid,lirfstand=LIRFSTAND,anchor=col,cover=round(float(mv.mean()),3)))
def stage_lirf2(base):
    """P0-14 LIRF late-departure two-stage model (B and C merged): candidates = ADEP=LIRF and (unmatched with ms>1h, or matched with aobt_sched>20min);
    label = nearest anchor in {SCHED,EOBT_1,LOBT,IOBT,AOBT_3} with |BLOCK-anchor|<120s, else other; multi-class LGB -> expected value: sum_k p_k(MVT-anchor_k) + p_aobt*delta member + p_other*global member;
    probabilities of missing anchors go to the global member. Members lirf2_over (overrides candidate rows) and lirf2_avg (half-half with base)."""
    ANC=['SCHED_TIME_UTC_mvt','EOBT_1_flt','LOBT_flt','IOBT_flt','AOBT_3_flt']; K=len(ANC)+1
    cand=lambda d:d.select((pl.col('ADEP_mvt')=='LIRF')&(((pl.col('flt_null')==1)&(pl.col('mvt_delay')>3600))|((pl.col('flt_null')==0)&(pl.col('aobt_sched')>1200)))).to_numpy().ravel()
    def label(d):
        D=np.vstack([np.nan_to_num(np.abs(d.select(sec('BLOCK_TIME_UTC_mvt',c)).to_numpy().ravel().astype(float)),nan=1e9) for c in ANC]).T; return np.where(D.min(1)<120,D.argmin(1),K-1)
    cols=list(dict.fromkeys(FEATSETS[base]+['aobt_sched','mvt_sched','fnum','sh','mh','dow2'])); cs=[c for c in cols if c in CAT]
    for mo in (1,7):
        tr,va=fold(mo); pg=np.load(os.path.join(a.work,'oof',f'{base}_{mo}.npy')); pdm=np.load(os.path.join(a.work,'oof',f'{base}_delta_{mo}.npy')); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float)
        ct=cand(tr); cv=cand(va); trs=lirf_frame(tr.filter(pl.Series(ct))); vas=lirf_frame(va.filter(pl.Series(cv))); lab=label(trs); labv=label(vas)
        log(f'  lirf2 fold{mo}: cand train {ct.sum():,} val {cv.sum():,}; label dist train {np.round(np.bincount(lab,minlength=K)/len(lab),3)} val {np.round(np.bincount(labv,minlength=K)/len(labv),3)}')
        Xtr=to_X(trs,cols); cats={c:Xtr[c].cat.categories for c in CAT if c in cols}; Xva=to_X(vas,cols,cats)
        pm=dict(P,objective='multiclass',num_class=K,metric='multi_logloss',num_threads=a.threads,num_leaves=31,min_data_in_leaf=30,learning_rate=0.03)
        m=lgb.train(pm,lgb.Dataset(Xtr,lab,categorical_feature=cs),3000,valid_sets=[lgb.Dataset(Xva,labv)],callbacks=[lgb.early_stopping(100,verbose=False)])
        pr=m.predict(Xva,num_iteration=m.best_iteration); acc=float((pr.argmax(1)==labv).mean())
        mv=vas['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy().astype(float); pgs=pg[cv]; pds=pdm[cv]; e=pr[:,K-1]*pgs
        for k,c in enumerate(ANC):
            v=mv-vas[c].dt.epoch('s').to_numpy().astype(float); ok=~np.isnan(v)
            if c=='AOBT_3_flt': e=e+np.where(ok,pr[:,k]*pds,pr[:,k]*pgs)
            else: e=e+np.where(ok,pr[:,k]*v,pr[:,k]*pgs)
        log(f'   acc {acc:.3f} iter {m.best_iteration}; cand subset RMSE base {rmse(pgs,yva[cv]):.0f} two-stage {rmse(e,yva[cv]):.0f} avg {rmse((e+pgs)/2,yva[cv]):.0f}')
        imp=sorted(zip(m.feature_name(),m.feature_importance('gain')),key=lambda x:-x[1])[:10]; log('   gain:',[(n,int(v)) for n,v in imp])
        for nm,w in [('lirf2_over',1.0),('lirf2_avg',0.5)]:
            p=pg.copy(); p[cv]=w*e+(1-w)*pgs; np.save(os.path.join(a.work,'oof',f'{nm}_{mo}.npy'),p); record(nm,mo,va,p,dict(acc=round(acc,3),ncand=int(cv.sum()),it=m.best_iteration,fs=base))
def stage_report():
    """Two-fold summary of results.tsv: mean RMSE, weighted nonA, nonA_RMSE = sqrt(weighted nonA MSE), slices of disturbed days."""
    rows={}
    for l in open(RES):
        e,mo,r,r3,c,x=l.rstrip('\n').split('\t'); c=json.loads(c); rows.setdefault(e,{})[int(mo)]=c
    log(f"{'exp':22s} {'mean':>7s} {'nonA1':>7s} {'nonA7':>7s} {'nonA_RMSE':>9s} {'dist1':>7s} {'dist7':>7s} {'B7':>6s} {'C7':>6s} {'D1':>5s} {'D7':>5s}")
    out=[]
    for e,d in rows.items():
        if 1 not in d or 7 not in d or 'nonA_mse' not in d[1]: continue
        m=(d[1]['rmse']+d[7]['rmse'])/2; n1,n7=d[1]['nonA_n'],d[7]['nonA_n']; mse=(d[1]['nonA_mse']*n1+d[7]['nonA_mse']*n7)/(n1+n7); nr=np.sqrt(mse)
        out.append((nr,e,m,d)); 
    for nr,e,m,d in sorted(out):
        g=lambda k,mo:d[mo]['cls'].get(k,{}).get('rmse','-')
        log(f"{e:22s} {m:7.1f} {d[1]['nonA_rmse']:7.1f} {d[7]['nonA_rmse']:7.1f} {nr:9.1f} {str(d[1]['dist_rmse']):>7s} {str(d[7]['dist_rmse']):>7s} {str(g('B',7)):>6s} {str(g('C',7)):>6s} {str(g('D',1)):>5s} {str(g('D',7)):>5s}")
def best_featset():
    """Best of base/p15/p01a/p05/metar in results.tsv by two-fold mean."""
    rows=[l.split('\t') for l in open(RES)] if os.path.exists(RES) else []
    sc={}
    for r in rows:
        if r[0] in FEATSETS: sc.setdefault(r[0],[]).append(float(r[2]))
    sc={k:np.mean(v) for k,v in sc.items() if len(v)==2}
    b=min(sc,key=sc.get) if sc else 'base'; log('featset scores',{k:round(v,1) for k,v in sc.items()},'-> best',b); return b
PHYS=['ADEP_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','ADES_mvt','AIRCRAFT_OPERATOR_flt','WK_TBL_CAT_flt','al','hour','dow','mod','month','flt_null','mvt_delay','mvt_eobt1','mvt_lobt','mvt_iobt','lobt_eobt1','dep10','dep30','arr15','arr20','rwy_gap_prev','rwy_gap_next','rwy_dep10','rwy_dep20_fwd','sched30','arr_taxi_med','arr_taxi_med60','turn_land','day_n','stand_hour_n','tc','sknt','vsby','wx_frz','deice_risk','synth','ma_dev']
def stage_p01b(base):
    """Physical sub-model for synthetic rows and unmatched rows: physical features only (no mvt_aobt3 etc.), trained on the subset, overrides/blends the global prediction."""
    for mo in (1,7):
        tr,va=fold(mo); pg=np.load(os.path.join(a.work,'oof',f'{base}_{mo}.npy')); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float)
        sel=lambda d:d.select(((pl.col('synth')==1)|(pl.col('flt_null')==1))&(pl.col('TAXITIME_SEC_mvt').fill_null(0)<=10800)).to_numpy().ravel()
        st=sel(tr); sv=va.select((pl.col('synth')==1)|(pl.col('flt_null')==1)).to_numpy().ravel()&~BMASK(va)
        trs=tr.filter(pl.Series(st)); Xtr=to_X(trs,PHYS); cats={c:Xtr[c].cat.categories for c in CAT if c in PHYS}; Xva=to_X(va.filter(pl.Series(sv)),PHYS,cats)
        m,ps,it=fit_reg(Xtr,trs['TAXITIME_SEC_mvt'].to_numpy().astype(float),Xva,yva[sv],hp(num_leaves=63,min_data_in_leaf=50))
        log(f'  p01b fold{mo}: subset train {st.sum():,} val {sv.sum():,} iter {it}; subset RMSE global {rmse(pg[sv],yva[sv]):.1f} phys {rmse(ps,yva[sv]):.1f} avg {rmse((pg[sv]+ps)/2,yva[sv]):.1f}')
        for nm,w in [('p01b_over',1.0),('p01b_avg',0.5)]:
            p=pg.copy(); p[sv]=w*ps+(1-w)*pg[sv]; record(nm,mo,va,p,dict(it=it,nsub=int(sv.sum())))
        imp=sorted(zip(m.feature_name(),m.feature_importance('gain')),key=lambda x:-x[1])[:12]; log('   phys gain:',[(n,int(v/1e6)) for n,v in imp])
def stage_p03(base):
    """Class-C anchor two-stage model: candidate set (aobt_sched>20m or aobt_eobt>20m, with match), multi-class anchor -> expected value."""
    ANC=['SCHED_TIME_UTC_mvt','EOBT_1_flt','LOBT_flt','IOBT_flt']; K=len(ANC)+2  # +normal(AOBT) +other
    cand=lambda d:d.select((pl.col('flt_null')==0)&((pl.col('aobt_sched')>1200)|(pl.col('aobt_eobt')>1200))).to_numpy().ravel()
    def label(d):
        diffs=[np.abs(d.select(sec('BLOCK_TIME_UTC_mvt',c)).to_numpy().ravel().astype(float)) for c in ANC+['AOBT_3_flt']]
        D=np.vstack([np.nan_to_num(x,nan=1e9) for x in diffs]).T; k=D.argmin(1); lab=np.where(D.min(1)<120,k,K-1); return lab
    cols=[c for c in FEATSETS[best_featset()] if c not in ('al',)]+['al','aobt_sched','mvt_sched']
    cols=list(dict.fromkeys(cols))
    for mo in (1,7):
        tr,va=fold(mo); pg=np.load(os.path.join(a.work,'oof',f'{base}_{mo}.npy')); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float)
        ct=cand(tr); cv=cand(va); trs=tr.filter(pl.Series(ct)); vas=va.filter(pl.Series(cv)); lab=label(trs); labv=label(vas)
        log(f'  p03 fold{mo}: cand train {ct.sum():,} val {cv.sum():,}; label dist train {np.bincount(lab,minlength=K)/len(lab)}')
        Xtr=to_X(trs,cols); cats={c:Xtr[c].cat.categories for c in CAT if c in cols}; Xva=to_X(vas,cols,cats); cs=[c for c in cols if c in CAT]
        pm=dict(P,objective='multiclass',num_class=K,metric='multi_logloss',num_threads=a.threads,num_leaves=63)
        m=lgb.train(pm,lgb.Dataset(Xtr,lab,categorical_feature=cs),3000,valid_sets=[lgb.Dataset(Xva,labv)],callbacks=[lgb.early_stopping(100,verbose=False)])
        pr=m.predict(Xva,num_iteration=m.best_iteration); acc=float((pr.argmax(1)==labv).mean())
        mv=vas['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy().astype(float)
        anc=np.vstack([mv-np.nan_to_num(vas[c].dt.epoch('s').to_numpy().astype(float),nan=np.nan) for c in ANC]).T  # MVT-anchor = taxi if BLOCK==anchor
        pgs=pg[cv]; e=pr[:,K-2]*pgs+pr[:,K-1]*pgs
        for k in range(len(ANC)):
            v=anc[:,k]; ok=~np.isnan(v); e=e+np.where(ok,pr[:,k]*v,pr[:,k]*pgs)
        log(f'   acc {acc:.3f} iter {m.best_iteration}; cand subset RMSE global {rmse(pgs,yva[cv]):.1f} two-stage {rmse(e,yva[cv]):.1f} avg {rmse((e+pgs)/2,yva[cv]):.1f}')
        for nm,w in [('p03_over',1.0),('p03_avg',0.5)]:
            p=pg.copy(); p[cv]=w*e+(1-w)*pgs; record(nm,mo,va,p,dict(acc=round(acc,3),ncand=int(cv.sum())))
def stage_p02(base):
    for mo in (1,7):
        tr,va=fold(mo); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float); pg=np.load(os.path.join(a.work,'oof',f'{base}_{mo}.npy')); mk=BMASK(va)
        for nm,kw in [('p02_iso',dict(iso=True)),('p02_deep',dict(deep=True)),('p02_deep_iso',dict(deep=True,iso=True))]:
            lp,info=fit_lirf(tr,**kw); p=apply_lirf(va,pg,lp)
            log(f'  {nm} fold{mo}: B subset RMSE base {rmse(pg[mk],yva[mk]):.0f} -> {rmse(p[mk],yva[mk]):.0f} (n={mk.sum()})'); record(nm,mo,va,p,info)
GRID=[dict(),dict(num_leaves=255,min_data_in_leaf=50),dict(num_leaves=63),dict(learning_rate=0.02),dict(feature_fraction=0.6,min_data_in_leaf=300),dict(num_leaves=255,learning_rate=0.03,feature_fraction=0.7,lambda_l2=10)]
def stage_seeds(base):
    cols=FEATSETS[base]
    for gi,g in enumerate(GRID):
        for s in range(3 if gi==0 else 1):
            for mo in (1,7):
                tr,va=fold(mo); t0=time.time(); p,p1,p2,info=pipeline(tr,va,cols,hp(**g),seed=s)
                nm=f'g{gi}s{s}'; np.save(os.path.join(a.work,'oof',f'{nm}_taxi_{mo}.npy'),p1); np.save(os.path.join(a.work,'oof',f'{nm}_delta_{mo}.npy'),p2); record(nm,mo,va,p,dict(info,g=g,seed=s,sec=round(time.time()-t0)))
def stage_boost(base):
    """CatBoost / XGBoost members (taxi and delta targets), using the best feature set."""
    cols=FEATSETS[base]; cs=[c for c in cols if c in CAT]
    for mo in (1,7):
        tr,va=fold(mo); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float); ytr=tr['TAXITIME_SEC_mvt'].to_numpy().astype(float)
        mt=tr['delta'].is_not_null().to_numpy(); mv=va['delta'].is_not_null().to_numpy(); lp,_=fit_lirf(tr)
        try:
            from catboost import CatBoostRegressor, Pool
            Xtr=tr.select(cols).to_pandas(); Xva=va.select(cols).to_pandas()
            for c in cs: Xtr[c]=Xtr[c].fillna('NA').astype(str); Xva[c]=Xva[c].fillna('NA').astype(str)
            t0=time.time(); cb=CatBoostRegressor(iterations=3000,learning_rate=0.08,depth=8,loss_function='RMSE',thread_count=a.threads,verbose=0,od_type='Iter',od_wait=100,border_count=128)
            cb.fit(Pool(Xtr,ytr,cat_features=cs),eval_set=Pool(Xva,yva,cat_features=cs)); p1=cb.predict(Xva)
            cb2=CatBoostRegressor(iterations=3000,learning_rate=0.08,depth=8,loss_function='RMSE',thread_count=a.threads,verbose=0,od_type='Iter',od_wait=100,border_count=128)
            cb2.fit(Pool(Xtr[mt],tr['delta'].to_numpy().astype(float)[mt],cat_features=cs),eval_set=Pool(Xva[mv],va['delta'].to_numpy().astype(float)[mv],cat_features=cs))
            p2=p1.copy(); p2[mv]=va['mvt_aobt3'].to_numpy().astype(float)[mv]+cb2.predict(Xva[mv]); p1=apply_lirf(va,p1,lp); p2=apply_lirf(va,p2,lp)
            np.save(os.path.join(a.work,'oof',f'cat_taxi_{mo}.npy'),p1); np.save(os.path.join(a.work,'oof',f'cat_delta_{mo}.npy'),p2)
            record('cat',mo,va,(p1+p2)/2,dict(it=cb.get_best_iteration(),itd=cb2.get_best_iteration(),sec=round(time.time()-t0),fs=base,lirfstand=LIRFSTAND)); del Xtr,Xva
        except Exception as e: log('  catboost failed:',repr(e))
        if os.environ.get('NOXGB')=='1': continue
        try:
            import xgboost as xgb
            Xtr=to_X(tr,cols); cats={c:Xtr[c].cat.categories for c in cs}; Xva=to_X(va,cols,cats); t0=time.time()
            prm=dict(objective='reg:squarederror',eta=0.05,max_depth=9,min_child_weight=50,subsample=0.8,colsample_bytree=0.8,tree_method='hist',max_cat_to_onehot=1,max_bin=256,nthread=a.threads)
            dtr=xgb.DMatrix(Xtr,ytr,enable_categorical=True); dva=xgb.DMatrix(Xva,yva,enable_categorical=True)
            b=xgb.train(prm,dtr,4000,evals=[(dva,'va')],early_stopping_rounds=100,verbose_eval=False); p1=b.predict(dva,iteration_range=(0,b.best_iteration+1))
            dtr2=xgb.DMatrix(Xtr[mt],tr['delta'].to_numpy().astype(float)[mt],enable_categorical=True); dva2=xgb.DMatrix(Xva[mv],va['delta'].to_numpy().astype(float)[mv],enable_categorical=True)
            b2=xgb.train(prm,dtr2,4000,evals=[(dva2,'va')],early_stopping_rounds=100,verbose_eval=False)
            p2=p1.copy(); p2[mv]=va['mvt_aobt3'].to_numpy().astype(float)[mv]+b2.predict(dva2,iteration_range=(0,b2.best_iteration+1)); p1=apply_lirf(va,p1,lp); p2=apply_lirf(va,p2,lp)
            np.save(os.path.join(a.work,'oof',f'xgb_taxi_{mo}.npy'),p1); np.save(os.path.join(a.work,'oof',f'xgb_delta_{mo}.npy'),p2)
            record('xgb',mo,va,(p1+p2)/2,dict(it=b.best_iteration,itd=b2.best_iteration,sec=round(time.time()-t0)))
        except Exception as e: log('  xgboost failed:',repr(e))
def stage_ens():
    """Caruana hill climbing (with replacement, positive weights), both folds jointly; members = all *_{1,7}.npy under oof/ that exist for both folds."""
    d=os.path.join(a.work,'oof'); names=sorted(n for n in {f[:-6] for f in os.listdir(d) if f.endswith('_1.npy')}&{f[:-6] for f in os.listdir(d) if f.endswith('_7.npy')} if not n.startswith(('ens','p03prob')))
    vas={mo:fold(mo)[1] for mo in (1,7)}; ys={mo:vas[mo]['TAXITIME_SEC_mvt'].to_numpy().astype(float) for mo in (1,7)}
    M={n:{mo:np.load(os.path.join(d,f'{n}_{mo}.npy')) for mo in (1,7)} for n in names}
    # ENS_OBJ=mean (default, used for the submission): arithmetic mean of the two-fold full RMSE; ENS_OBJ=proxy (optional, not adopted): non-extreme RMSE pooled row-wise over the two folds (same definition as the report's nonA_RMSE).
    obj=os.environ.get('ENS_OBJ','mean')
    isA={mo:vas[mo].select((pl.col('TAXITIME_SEC_mvt')>10800)&~((pl.col('flt_null')==1)&(pl.col('ADEP_mvt')=='LIRF')&(pl.col('mvt_delay')>3600))).to_numpy().ravel() for mo in (1,7)}
    log(f'ens objective: {obj}'+('' if obj=='mean' else f' (nonA rows {sum(int((~isA[mo]).sum()) for mo in (1,7)):,})'))
    def score(w):
        tot=0; sse=0; n=0
        for mo in (1,7):
            p=sum(w[n_]*M[n_][mo] for n_ in w)/sum(w.values())
            if obj=='mean': tot+=rmse(p,ys[mo])
            else: e=(p-ys[mo])[~isA[mo]]; sse+=float((e*e).sum()); n+=len(e)
        return tot/2 if obj=='mean' else float(np.sqrt(sse/n))
    single={n:round(score({n:1}),2) for n in names}; log('members:',single)
    w={min(single,key=single.get):1}; cur=score(w)
    for step in range(40):
        best=None
        for n in names:
            w2=dict(w); w2[n]=w2.get(n,0)+1; s=score(w2)
            if best is None or s<best[0]: best=(s,n)
        if best[0]>=cur-0.005: break
        cur,_=best; w[best[1]]=w.get(best[1],0)+1
    tot=sum(w.values()); wn={k:round(v/tot,3) for k,v in w.items()}; log(f'ensemble two-fold mean RMSE {cur:.2f} weights {wn}')
    for mo in (1,7):
        p=sum(w[n]*M[n][mo] for n in w)/tot; va=fold(mo)[1]; record('ens',mo,va,p,dict(weights=wn))
    json.dump(wn,open(os.path.join(a.work,'ens_weights.json'),'w'))

def iters_for(name):
    rows=[json.loads(l.split('\t')[5]) for l in open(RES) if l.split('\t')[0]==name]
    it=int(np.mean([r['it'] for r in rows])*1.15); itd=int(np.mean([r['itd'] for r in rows])*1.15); return it,itd
def stage_full(base):
    """Full-year fit -> sub/upstanding-firefly_<tag>.parquet (no upload). Options via environment: NSEED (default 3), LIRF (none/iso/deep/deep_iso), P01B (over/avg/none), P03 (over/avg/none)."""
    nseed=int(os.environ.get('NSEED','3')); lk=os.environ.get('LIRF','none'); lirf_kw=dict(deep='deep' in lk,iso='iso' in lk)
    cols=FEATSETS[base]; it,itd=iters_for(base); log(f'full: featset {base} iters {it}/{itd} seeds {nseed} lirf {lirf_kw} p01b {os.environ.get("P01B","none")} p03 {os.environ.get("P03","none")}')
    d=data(); tr=d.filter(pl.col('src')!=99); rk=d.filter(pl.col('src')==99)
    Xtr=to_X(tr,cols); cats={c:Xtr[c].cat.categories for c in CAT if c in cols}; Xrk=to_X(rk,cols,cats); ytr=tr['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    mt=tr['delta'].is_not_null().to_numpy(); mv=rk['AOBT_3_flt'].is_not_null().to_numpy(); acc=np.zeros(rk.height)
    for s in range(nseed):
        m,_,_=fit_reg(Xtr,ytr,params=hp(seed=s),rounds=it); p=m.predict(Xrk)
        md,_,_=fit_reg(Xtr[mt],tr['delta'].to_numpy().astype(float)[mt],params=hp(seed=s),rounds=itd); p2=p.copy(); p2[mv]=rk['mvt_aobt3'].to_numpy().astype(float)[mv]+md.predict(Xrk[mv])
        acc+=(p+p2)/2; log(f'  seed {s} done')
    p=acc/nseed
    lp,info=fit_lirf(tr,**lirf_kw); p=apply_lirf(rk,p,lp); log('  lirf applied',info,'rows',int(BMASK(rk).sum()))
    w={'over':1.0,'avg':0.5}.get(os.environ.get('P01B','none'))
    if w:
        st=tr.select(((pl.col('synth')==1)|(pl.col('flt_null')==1))&(pl.col('TAXITIME_SEC_mvt')<=10800)).to_numpy().ravel(); sv=rk.select((pl.col('synth')==1)|(pl.col('flt_null')==1)).to_numpy().ravel()&~BMASK(rk)
        trs=tr.filter(pl.Series(st)); X1=to_X(trs,PHYS); c1={c:X1[c].cat.categories for c in CAT if c in PHYS}; itb=int(np.mean([json.loads(l.split('\t')[5])['it'] for l in open(RES) if l.startswith('p01b_over\t')])*1.15)
        m1,_,_=fit_reg(X1,trs['TAXITIME_SEC_mvt'].to_numpy().astype(float),params=hp(num_leaves=63,min_data_in_leaf=50),rounds=itb); ps=m1.predict(to_X(rk.filter(pl.Series(sv)),PHYS,c1)); p[sv]=w*ps+(1-w)*p[sv]; log(f'  p01b w={w} rows {sv.sum()} iters {itb}')
    sub=pl.read_parquet(os.path.join(a.raw,'submitting.parquet')).select('MVT_ID_mvt')
    out=sub.join(rk.select('MVT_ID_mvt',pl.Series('TAXITIME_SEC_mvt',p)),on='MVT_ID_mvt',how='left'); miss=out['TAXITIME_SEC_mvt'].null_count()
    if miss: out=out.with_columns(pl.col('TAXITIME_SEC_mvt').fill_null(float(np.median(p))))
    out=out.with_columns(pl.col('TAXITIME_SEC_mvt').clip(30,None).round(0).cast(pl.Int32)); fn=os.path.join(a.raw,'..','sub',f'upstanding-firefly_{a.tag}.parquet'); out.write_parquet(fn)
    log('wrote',fn,'rows',out.height,'missing',miss,out['TAXITIME_SEC_mvt'].describe())

def stage_apt(base):
    """Per-airport member (taxi + delta targets, LIRF blend)."""
    cols=[c for c in FEATSETS[base] if c!='ADEP_mvt']
    for mo in (1,7):
        tr,va=fold(mo); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float); p=np.zeros(va.height); p2=np.zeros(va.height); t0=time.time(); its=[]
        for ap_ in sorted(va['ADEP_mvt'].unique()):
            mt_=tr.select(pl.col('ADEP_mvt')==ap_).to_numpy().ravel(); mv_=va.select(pl.col('ADEP_mvt')==ap_).to_numpy().ravel(); tra=tr.filter(pl.Series(mt_)); vaa=va.filter(pl.Series(mv_))
            Xtr=to_X(tra,cols); cats={c:Xtr[c].cat.categories for c in CAT if c in cols}; Xva=to_X(vaa,cols,cats)
            m,pa,it=fit_reg(Xtr,tra['TAXITIME_SEC_mvt'].to_numpy().astype(float),Xva,yva[mv_],hp(num_leaves=63))
            mt=tra['delta'].is_not_null().to_numpy(); mv=vaa['delta'].is_not_null().to_numpy()
            md,pd_,itd=fit_reg(Xtr[mt],tra['delta'].to_numpy().astype(float)[mt],Xva[mv],vaa['delta'].to_numpy().astype(float)[mv],hp(num_leaves=63))
            pb=pa.copy(); pb[mv]=vaa['mvt_aobt3'].to_numpy().astype(float)[mv]+pd_; p[mv_]=pa; p2[mv_]=pb; its.append((ap_,it,itd))
        lp,_=fit_lirf(tr); p=apply_lirf(va,p,lp); p2=apply_lirf(va,p2,lp)
        np.save(os.path.join(a.work,'oof',f'apt_taxi_{mo}.npy'),p); np.save(os.path.join(a.work,'oof',f'apt_delta_{mo}.npy'),p2)
        np.save(os.path.join(a.work,'oof',f'apt_{mo}.npy'),(p+p2)/2)
        log(f'  apt fold{mo}: taxi {rmse(p,yva):.1f} delta {rmse(p2,yva):.1f} iters {its}'); record('apt',mo,va,(p+p2)/2,dict(sec=round(time.time()-t0),fs=base,lirfstand=LIRFSTAND,its=its))
def stage_logt(base):
    """log1p target + Duan smearing member (taxi target; not applicable to the delta branch)."""
    cols=FEATSETS[base]
    for mo in (1,7):
        tr,va=fold(mo); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float); ytr=tr['TAXITIME_SEC_mvt'].to_numpy().astype(float)
        Xtr=to_X(tr,cols); cats={c:Xtr[c].cat.categories for c in CAT if c in cols}; Xva=to_X(va,cols,cats)
        m,pl_,it=fit_reg(Xtr,np.log1p(np.clip(ytr,0,None)),Xva,np.log1p(np.clip(yva,0,None)),hp())
        res=np.log1p(np.clip(ytr,0,None))-m.predict(Xtr,num_iteration=it); sm=float(np.mean(np.exp(res))); p=np.expm1(pl_)*sm
        lp,_=fit_lirf(tr); p=apply_lirf(va,p,lp); log(f'  logt fold{mo}: smear {sm:.3f} iter {it}'); record('logt',mo,va,p,dict(it=it,smear=round(sm,3)))
def stage_p03b(base):
    """p03 shrinkage scan: e = pg + lambda*sum_k p_k(anchor_k - pg), only on rows whose anchor probability sum > tau."""
    ANC=['SCHED_TIME_UTC_mvt','EOBT_1_flt','LOBT_flt','IOBT_flt']; K=len(ANC)+2
    cand=lambda d:d.select((pl.col('flt_null')==0)&((pl.col('aobt_sched')>1200)|(pl.col('aobt_eobt')>1200))).to_numpy().ravel()
    def label(d):
        D=np.vstack([np.nan_to_num(np.abs(d.select(sec('BLOCK_TIME_UTC_mvt',c)).to_numpy().ravel().astype(float)),nan=1e9) for c in ANC+['AOBT_3_flt']]).T; return np.where(D.min(1)<120,D.argmin(1),K-1)
    cols=list(dict.fromkeys(FEATSETS[base]+['aobt_sched','mvt_sched']))
    for mo in (1,7):
        tr,va=fold(mo); pg=np.load(os.path.join(a.work,'oof',f'{base}_{mo}.npy')); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float)
        ct=cand(tr); cv=cand(va); trs=tr.filter(pl.Series(ct)); vas=va.filter(pl.Series(cv)); lab=label(trs); labv=label(vas)
        Xtr=to_X(trs,cols); cats={c:Xtr[c].cat.categories for c in CAT if c in cols}; Xva=to_X(vas,cols,cats); cs=[c for c in cols if c in CAT]
        m=lgb.train(dict(P,objective='multiclass',num_class=K,metric='multi_logloss',num_threads=a.threads,num_leaves=63),lgb.Dataset(Xtr,lab,categorical_feature=cs),3000,valid_sets=[lgb.Dataset(Xva,labv)],callbacks=[lgb.early_stopping(100,verbose=False)])
        pr=m.predict(Xva,num_iteration=m.best_iteration); np.save(os.path.join(a.work,'oof',f'p03prob_{mo}.npy'),pr)
        mv=vas['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy().astype(float); pgs=pg[cv]; adj=np.zeros(len(pgs)); psum=np.zeros(len(pgs))
        for k in range(len(ANC)):
            v=mv-vas[ANC[k]].dt.epoch('s').to_numpy().astype(float); ok=~np.isnan(v); adj+=np.where(ok,pr[:,k]*(v-pgs),0); psum+=np.where(ok,pr[:,k],0)
        for lam in (0.3,0.5,0.7,1.0):
            for tau in (0.0,0.3,0.5):
                e=pgs+np.where(psum>tau,lam*adj,0); p=pg.copy(); p[cv]=e; nm=f'p03b_l{lam}_t{tau}'; c=components(va,p)
                log(f'  {nm} fold{mo}: RMSE {c["rmse"]} C {c["cls"]["C"]["rmse"]} D {c["cls"]["D"]["rmse"]}')
                if (lam,tau) in ((0.5,0.3),(0.7,0.3),(0.7,0.5),(1.0,0.5)): record(nm,mo,va,p)

def stage_fullens(base):
    """Fit the members listed in the work directory's ens_weights.json on the full year and weight them -> sub/upstanding-firefly_<tag>.parquet (no upload).
    Supported members: experiment names (rows in results.tsv give fs/grid/iterations; suffix _taxi/_delta/none = average), g{i}s{s}, cat, xgb, anc_<k>, apt[_taxi|_delta], lirf2_over|lirf2_avg.
    The full-year fits are cached per (fs,grid) and shared; the LIRF sub-model follows the LIRFSTAND env."""
    w=json.load(open(os.path.join(a.work,'ens_weights.json')))
    global DF
    d=data(); tr=d.filter(pl.col('src')!=99); rk=d.filter(pl.col('src')==99); ytr=tr['TAXITIME_SEC_mvt'].to_numpy().astype(float); dtr=tr['delta'].to_numpy().astype(float)
    if os.environ.get('LOWMEM','1')=='1': DF=None; del d; import gc; gc.collect()  # fullens no longer uses the global store; free about 1.1 GB
    mt=tr['delta'].is_not_null().to_numpy(); mv=rk['AOBT_3_flt'].is_not_null().to_numpy(); ma=rk['mvt_aobt3'].to_numpy().astype(float)
    XC={}
    def X(fs):
        if fs not in XC:
            cc=FEATSETS[fs]; Xt=to_X(tr,cc); ct={c:Xt[c].cat.categories for c in CAT if c in cc}; XC[fs]=(Xt,to_X(rk,cc,ct))
        return XC[fs]
    FY={}
    def fy(fs,grid,it,itd):
        """Full-year taxi and delta predictions (delta only on mv rows), cached."""
        k=(fs,grid); ns=int(os.environ.get('NSEED','1')); fyp=os.path.join(a.work,f'fy_{fs}_g{grid}_it{it}_{itd}_ns{ns}_{a.tag}.npz')
        if k not in FY and os.environ.get('REUSE')=='1' and os.path.exists(fyp): z=np.load(fyp); FY[k]=(z['pt'],z['pd']); log(f'  fy{k} reused {os.path.basename(fyp)}')  # fy predictions are written to disk so that fullens can be split across processes
        if k not in FY:
            Xt,Xr=X(fs); pr=hp(**GRID[grid]); t0=time.time(); pt=np.zeros(rk.height); pd_=np.zeros(int(mv.sum()))
            for s_ in range(ns):
                pt+=fit_reg(Xt,ytr,params=dict(pr,seed=s_),rounds=it)[0].predict(Xr)/ns; pd_+=fit_reg(Xt[mt],dtr[mt],params=dict(pr,seed=s_),rounds=itd)[0].predict(Xr[mv])/ns
            FY[k]=(pt,pd_); np.savez(fyp,pt=pt,pd=pd_); log(f'  fy{k} iters {it}/{itd} seeds {ns} {time.time()-t0:.0f}s -> {os.path.basename(fyp)}')
        return FY[k]
    def combine(pt,pd_,kind):
        if kind=='taxi': return pt
        p2=pt.copy(); p2[mv]=ma[mv]+pd_
        return p2 if kind=='delta' else (pt+p2)/2
    lp,info=fit_lirf(tr); preds={}
    for name in w:
        kind=name.rsplit('_',1)[1] if name.endswith(('_taxi','_delta')) else 'avg'; stem=name.rsplit('_',1)[0] if kind!='avg' else name; t0=time.time()
        fp=os.path.join(a.work,f'full_{name}.npy')
        if os.environ.get('REUSE')=='1' and os.path.exists(fp): preds[name]=np.load(fp); log(f'  member {name} w={w[name]} reused'); continue
        sel=os.environ.get('MEMBERS')  # MEMBERS=a,b computes only these members in this process (others without a cache are skipped); together with REUSE=1 this allows separate processes and avoids stacking CatBoost and LGB members against the memory guard
        if sel and name not in sel.split(','): log(f'  member {name}: not in MEMBERS, skipped this process'); continue
        try:
            if stem.startswith('g') and 's' in stem and stem[1:].replace('s','').isdigit():
                gi,sd=map(int,stem[1:].split('s')); mm=exp_meta(stem); pr=hp(**GRID[gi],seed=sd); Xt,Xr=X(mm['fs'])
                pt=fit_reg(Xt,ytr,params=pr,rounds=mm['it'])[0].predict(Xr); pd_=fit_reg(Xt[mt],dtr[mt],params=pr,rounds=mm['itd'])[0].predict(Xr[mv]); p=combine(pt,pd_,kind)
            elif stem=='apt':
                mm=exp_meta('apt'); cc=[c for c in FEATSETS[mm['fs']] if c!='ADEP_mvt']; rows=[json.loads(l.split('\t')[5])['its'] for l in open(RES) if l.split('\t')[0]=='apt']
                its={ap_:(int(np.mean([r[i][1] for r in rows])*1.15),int(np.mean([r[i][2] for r in rows])*1.15)) for i,(ap_,_,_) in enumerate(rows[0])}
                pt=np.zeros(rk.height); pd_=np.zeros(int(mv.sum()))
                for ap_,(it,itd) in its.items():
                    mt_=tr.select(pl.col('ADEP_mvt')==ap_).to_numpy().ravel(); mr_=rk.select(pl.col('ADEP_mvt')==ap_).to_numpy().ravel(); tra=tr.filter(pl.Series(mt_)); rka=rk.filter(pl.Series(mr_))
                    Xt=to_X(tra,cc); ct={c:Xt[c].cat.categories for c in CAT if c in cc}; Xr=to_X(rka,cc,ct); ya=ytr[mt_]; da=dtr[mt_]; mta=~np.isnan(da); mra=mv[mr_]
                    pt[mr_]=fit_reg(Xt,ya,params=hp(num_leaves=63),rounds=it)[0].predict(Xr)
                    pda=fit_reg(Xt[mta],da[mta],params=hp(num_leaves=63),rounds=itd)[0].predict(Xr[mra]); tmp=np.zeros(int(mr_.sum())); tmp[mra]=pda; full=np.zeros(rk.height); full[mr_]=tmp; pd_[:]+=full[mv]
                p=combine(pt,pd_,kind)
            elif stem.startswith('lirf2'):
                mm=exp_meta(stem); fs=mm['fs']; ms=exp_meta(fs); pt,pd_=fy(fs,ms['grid'],ms['it'],ms['itd']); pg=apply_lirf(rk,combine(pt,pd_,'avg'),lp); pdm=apply_lirf(rk,combine(pt,pd_,'delta'),lp)
                ANC=['SCHED_TIME_UTC_mvt','EOBT_1_flt','LOBT_flt','IOBT_flt','AOBT_3_flt']; K=len(ANC)+1
                cand=lambda dd:dd.select((pl.col('ADEP_mvt')=='LIRF')&(((pl.col('flt_null')==1)&(pl.col('mvt_delay')>3600))|((pl.col('flt_null')==0)&(pl.col('aobt_sched')>1200)))).to_numpy().ravel()
                ct_=cand(tr); cr_=cand(rk); trs=lirf_frame(tr.filter(pl.Series(ct_))); rks=lirf_frame(rk.filter(pl.Series(cr_)))
                D_=np.vstack([np.nan_to_num(np.abs(trs.select(sec('BLOCK_TIME_UTC_mvt',c)).to_numpy().ravel().astype(float)),nan=1e9) for c in ANC]).T; lab=np.where(D_.min(1)<120,D_.argmin(1),K-1)
                cc=list(dict.fromkeys(FEATSETS[fs]+['aobt_sched','mvt_sched','fnum','sh','mh','dow2'])); cs=[c for c in cc if c in CAT]
                Xt=to_X(trs,cc); ct={c:Xt[c].cat.categories for c in CAT if c in cc}; Xr=to_X(rks,cc,ct)
                pm=dict(P,objective='multiclass',num_class=K,num_threads=a.threads,num_leaves=31,min_data_in_leaf=30,learning_rate=0.03)
                m2=lgb.train(pm,lgb.Dataset(Xt,lab,categorical_feature=cs),mm['it']); pr=m2.predict(Xr)
                mvs=rks['MVT_TIME_UTC_mvt'].dt.epoch('s').to_numpy().astype(float); pgs=pg[cr_]; pds=pdm[cr_]; e=pr[:,K-1]*pgs
                for k,c in enumerate(ANC):
                    v=mvs-rks[c].dt.epoch('s').to_numpy().astype(float); ok=~np.isnan(v); e=e+np.where(ok,pr[:,k]*(pds if c=='AOBT_3_flt' else v),pr[:,k]*pgs)
                ww=1.0 if stem=='lirf2_over' else 0.5; p=pg.copy(); p[cr_]=ww*e+(1-ww)*pgs; log(f'  lirf2 cand rows {int(cr_.sum())} label dist {np.round(np.bincount(lab,minlength=K)/len(lab),3)}')
            elif stem.startswith('anc_'):
                mm=exp_meta(stem); fs=mm['fs']; col=ANCH[stem[4:]]; fbn=os.environ.get('FALLBACK',f'{fs}_taxi'); fbm=exp_meta(fbn.rsplit('_',1)[0])
                pt=fy(fs,fbm['grid'],fbm['it'],fbm['itd'])[0].copy(); Xt,Xr=X(fs); atr=tr[col].to_numpy().astype(float); ark=rk[col].to_numpy().astype(float); mta=~np.isnan(atr); mva=~np.isnan(ark)
                pda=fit_reg(Xt[mta],ytr[mta]-atr[mta],params=hp(**GRID[mm['grid']]),rounds=mm['it'])[0].predict(Xr[mva]); pt[mva]=ark[mva]+pda; p=pt
            elif stem=='cat':
                from catboost import CatBoostRegressor, Pool; mm=exp_meta('cat'); cc=FEATSETS[mm['fs']]; cs=[c for c in cc if c in CAT]; import pandas as pd, gc
                mk=lambda n,s_=0:CatBoostRegressor(iterations=n,learning_rate=0.08,depth=8,loss_function='RMSE',thread_count=a.threads,verbose=0,border_count=128,random_seed=s_)
                cs_=int(os.environ.get('CATSEED','1')); part=os.environ.get('CATPART','both'); fpt=os.path.join(a.work,f'cat_pt_{a.tag}.npy'); fpd=os.path.join(a.work,f'cat_pd_{a.tag}.npy')
                def cframe(src_):
                    """Category columns use integer codes from a mapping shared by training and evaluation (CatBoost accepts int categories; an object-string copy once took a single process to 9.6 GB)."""
                    C=src_.select(cc).to_pandas(); R=rk.select(cc).to_pandas()
                    for c in cs: cats_=pd.Index(pd.concat([C[c],R[c]]).fillna('NA').unique()); C[c]=pd.Categorical(C[c].fillna('NA'),categories=cats_).codes.astype(np.int32); R[c]=pd.Categorical(R[c].fillna('NA'),categories=cats_).codes.astype(np.int32)
                    return C,R
                # CATPART=taxi / delta: one process each (both targets in one process once reached 10.9 GB and hit the memory guard); intermediate predictions go to disk; both = only assemble when both are on disk
                if part=='taxi' or (part=='both' and not os.path.exists(fpt)):
                    Ct,Cr=cframe(tr)
                    if os.environ.get('MEMBERS')=='cat': tr=None; gc.collect()
                    pt=sum(mk(mm['it'],s_).fit(Pool(Ct,ytr,cat_features=cs)).predict(Cr) for s_ in range(cs_))/cs_; np.save(fpt,pt); del Ct,Cr; gc.collect(); log(f'  cat taxi part -> {os.path.basename(fpt)}')
                    if part=='taxi': continue
                if part=='delta' or (part=='both' and not os.path.exists(fpd)):
                    Ct,Cr=cframe(tr.filter(pl.Series(mt))); Cr=Cr[mv]
                    if os.environ.get('MEMBERS')=='cat': tr=None; gc.collect()
                    pd_=sum(mk(mm['itd'],s_).fit(Pool(Ct,dtr[mt],cat_features=cs)).predict(Cr) for s_ in range(cs_))/cs_; np.save(fpd,pd_); del Ct,Cr; gc.collect(); log(f'  cat delta part -> {os.path.basename(fpd)}')
                    if part=='delta': continue
                pt=np.load(fpt); pd_=np.load(fpd) if kind!='taxi' else None; p=combine(pt,pd_,kind)
            elif any(l.split('\t')[0]==stem for l in open(RES)):
                mm=exp_meta(stem); pt,pd_=fy(mm['fs'],mm['grid'],mm['it'],mm['itd']); p=combine(pt,pd_,kind)
            else: log(f'  member {name}: unsupported, skipped'); continue
        except Exception as e:
            import traceback; log(f'  member {name} failed: {e!r}'); traceback.print_exc(); continue
        if not stem.startswith('lirf2'): p=apply_lirf(rk,p,lp)
        preds[name]=p; np.save(os.path.join(a.work,f'full_{name}.npy'),p); log(f'  member {name} w={w[name]} done {time.time()-t0:.0f}s mean {np.mean(p):.0f}')
    PM=np.vstack([preds[n] for n in preds]); W=np.array([w[n] for n in preds])[:,None]; M=~np.isnan(PM); PM=np.nan_to_num(PM)
    p=(PM*W*M).sum(0)/np.maximum((W*M).sum(0),1e-9)
    sub=pl.read_parquet(os.path.join(a.raw,'submitting.parquet')).select('MVT_ID_mvt')
    out=sub.join(rk.select('MVT_ID_mvt',pl.Series('TAXITIME_SEC_mvt',p)),on='MVT_ID_mvt',how='left'); miss=out['TAXITIME_SEC_mvt'].null_count()
    if miss: out=out.with_columns(pl.col('TAXITIME_SEC_mvt').fill_null(float(np.median(p))))
    out=out.with_columns(pl.col('TAXITIME_SEC_mvt').clip(30,None).round(0).cast(pl.Int32)); fn=os.path.join(a.raw,'..','sub',f'upstanding-firefly_{a.tag}.parquet'); out.write_parquet(fn)
    np.save(os.path.join(a.work,f'pred_{a.tag}.npy'),p); log('wrote',fn,'rows',out.height,'missing',miss,'members',list(preds),out['TAXITIME_SEC_mvt'].describe())
    # row-wise difference to the previous version + per-airport means (guardrail check)
    prev=os.environ.get('PREV')
    if prev and os.path.exists(prev):
        q=pl.read_parquet(prev).select('MVT_ID_mvt',pl.col('TAXITIME_SEC_mvt').alias('prev')).join(out,on='MVT_ID_mvt'); x=q['prev'].to_numpy().astype(float); y_=q['TAXITIME_SEC_mvt'].to_numpy().astype(float)
        log(f'  vs {os.path.basename(prev)}: corr {np.corrcoef(x,y_)[0,1]:.4f} rowdiff RMSE {rmse(x,y_):.1f} mean {y_.mean():.1f} vs {x.mean():.1f} >3h rows {(y_>10800).sum()} vs {(x>10800).sum()}')
        ap=rk.select('MVT_ID_mvt','ADEP_mvt','ym').join(q,on='MVT_ID_mvt').group_by('ADEP_mvt','ym').agg(pl.col('TAXITIME_SEC_mvt').mean().round(0).alias('new'),pl.col('prev').mean().round(0).alias('prev'),pl.len()).sort('ADEP_mvt','ym'); log(ap)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('raw'); ap.add_argument('work'); ap.add_argument('stages',nargs='+'); ap.add_argument('--threads',type=int,default=6); ap.add_argument('--tag',default='ensa'); ap.add_argument('--grid',type=int,default=1)
    args=ap.parse_args(); configure(args.raw,args.work,args.threads,args.tag,args.grid)
    os.makedirs(os.path.join(a.work,'oof'),exist_ok=True)
    for st in args.stages:
        t0=time.time(); log(f'\n===== stage {st} {time.strftime("%H:%M:%S")}')
        if st=='feat': build_feat()
        elif st in FEATSETS: stage_featset(st)
        elif '@' in st and st.split('@')[0] in FEATSETS: stage_featset(*st.split('@'))
        elif st=='report': stage_report()
        elif st.startswith('lomo:'): stage_lomo(st[5:])
        elif st=='ens12': stage_ens12()
        elif st=='anchors': stage_anchors(os.environ.get('FEATSET') or best_featset())
        elif st=='lirf2': stage_lirf2(os.environ.get('FEATSET') or best_featset())
        elif st=='p01b': stage_p01b(best_featset())
        elif st=='p03': stage_p03(best_featset())
        elif st=='p02': stage_p02(best_featset())
        elif st=='seeds': stage_seeds(best_featset())
        elif st=='boost': stage_boost(os.environ.get('FEATSET') or best_featset())
        elif st=='ens': stage_ens()
        elif st=='fullens': stage_fullens(os.environ.get('FEATSET') or best_featset())
        elif st=='apt': stage_apt(os.environ.get('FEATSET') or best_featset())
        elif st=='logt': stage_logt(best_featset())
        elif st=='p03b': stage_p03b(best_featset())
        elif st=='full': stage_full(os.environ.get('FEATSET') or best_featset())
        else: raise ValueError(f'unknown stage {st}')
        log(f'===== stage {st} done {time.time()-t0:.0f}s')
    with open(os.path.join(a.work,f'done_{"_".join(args.stages)}'),'w') as stream:
        stream.write(time.strftime('%F %T'))

if __name__=='__main__':
    main()
