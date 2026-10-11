"""Experiment round: new features (overtaking, turnaround, stand x runway target encoding) + two-stage blended model vs a single L2 model.
Two folds (validation November/December, trained on the first two months each). All features use only columns available in the DEP/ARR rows of ranking. Memory < 1.5 GB.
"""
import sys, os, time
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR','load_as_storage')
sys.path.insert(0, os.path.dirname(__file__))
import polars as pl, numpy as np, lightgbm as lgb
from features_base import feats as base_feats, CAT, BASE, CONG, MONTHS, to_X, rmse
raw = sys.argv[1] if len(sys.argv)>1 else 'raw'
sec=lambda a,b:(pl.col(a)-pl.col(b)).dt.total_seconds()
NEW = ['overtake','overtake_rwy','turn_aobt','turn_land','same_type','te_stand_rwy','te_stand','te_rwy','delta_te']

def feats2(df_raw, canonical=False):
    dep = base_feats(df_raw, canonical=canonical)
    # overtaking: rank by MVT minus rank by AOBT at the same airport (i64); same-runway version
    dep = dep.sort(['ADEP_mvt','MVT_TIME_UTC_mvt']+(['MVT_ID_mvt'] if canonical else [])).with_columns(
        pl.col('MVT_TIME_UTC_mvt').rank('ordinal').over('ADEP_mvt').cast(pl.Int64).alias('rk_m'),
        pl.col('AOBT_3_flt').rank('ordinal').over('ADEP_mvt').cast(pl.Int64).alias('rk_a'),
        pl.col('MVT_TIME_UTC_mvt').rank('ordinal').over('ADEP_mvt','RUNWAY_mvt').cast(pl.Int64).alias('rk_mr'),
        pl.col('AOBT_3_flt').rank('ordinal').over('ADEP_mvt','RUNWAY_mvt').cast(pl.Int64).alias('rk_ar'),
    ).with_columns((pl.col('rk_m')-pl.col('rk_a')).alias('overtake'), (pl.col('rk_mr')-pl.col('rk_ar')).alias('overtake_rwy'))
    # turnaround: previous arrival at the same airport and stand (BLOCK of an ARR row = in-block, complete in ranking)
    arr = df_raw.filter(pl.col('PHASE_mvt')=='ARR').select(pl.col('ADES_mvt').alias('ADEP_mvt'),'STAND_mvt',pl.col('BLOCK_TIME_UTC_mvt').alias('in_block'),pl.col('MVT_TIME_UTC_mvt').alias('land'),pl.col('AIRCRAFT_TYPE_mvt').alias('arr_type')).drop_nulls(['STAND_mvt','in_block']).sort('in_block')
    dep = dep.drop_nulls(['STAND_mvt']).sort(['MVT_TIME_UTC_mvt']+(['MVT_ID_mvt'] if canonical else []))
    dep = dep.join_asof(arr, left_on='MVT_TIME_UTC_mvt', right_on='in_block', by=['ADEP_mvt','STAND_mvt'], strategy='backward')
    dep = dep.with_columns(sec('AOBT_3_flt','in_block').alias('turn_aobt'), sec('AOBT_3_flt','land').alias('turn_land'),
                           (pl.col('arr_type')==pl.col('AIRCRAFT_TYPE_mvt')).cast(pl.Int8).alias('same_type'))
    return dep

def add_te(tr, va, k=20):
    """target encoding: median taxitime and mean delta of the training set by (airport, stand, runway), smoothed towards the airport level."""
    tr = tr.with_columns(sec('AOBT_3_flt','MVT_TIME_UTC_mvt').alias('_neg_ma'))
    ap = tr.group_by('ADEP_mvt').agg(pl.col('TAXITIME_SEC_mvt').median().alias('ap_med'))
    out=[]
    for df in (tr, va):
        df = df.join(ap, on='ADEP_mvt', how='left')
        for keys,name in [(['ADEP_mvt','STAND_mvt','RUNWAY_mvt'],'te_stand_rwy'),(['ADEP_mvt','STAND_mvt'],'te_stand'),(['ADEP_mvt','RUNWAY_mvt'],'te_rwy')]:
            g = tr.group_by(keys).agg(pl.col('TAXITIME_SEC_mvt').median().alias('g_med'), pl.len().alias('g_n'))
            df = df.join(g, on=keys, how='left').with_columns(
                ((pl.col('g_med').fill_null(pl.col('ap_med'))*pl.col('g_n').fill_null(0) + pl.col('ap_med')*k)/(pl.col('g_n').fill_null(0)+k)).alias(name)).drop('g_med','g_n')
        # airport x stand mean of delta (AOBT-BLOCK): tendency towards anomalous rows
        gd = tr.with_columns(sec('AOBT_3_flt','BLOCK_TIME_UTC_mvt').alias('_d')).group_by('ADEP_mvt','STAND_mvt').agg(pl.col('_d').mean().alias('d_mu'), pl.len().alias('d_n'))
        df = df.join(gd, on=['ADEP_mvt','STAND_mvt'], how='left').with_columns(((pl.col('d_mu').fill_null(0)*pl.col('d_n').fill_null(0))/(pl.col('d_n').fill_null(0)+k)).alias('delta_te')).drop('d_mu','d_n','ap_med')
        out.append(df)
    return out

def load(m): return feats2(pl.read_parquet(f'{raw}/training_{MONTHS[m]}.parquet'))

P = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
def train_reg(Xtr,ytr,Xva,yva):
    cats=[c for c in CAT if c in Xtr]
    m=lgb.train(dict(P,objective='regression',metric='rmse'), lgb.Dataset(Xtr,ytr,categorical_feature=cats), 3000, valid_sets=[lgb.Dataset(Xva,yva)], callbacks=[lgb.early_stopping(100,verbose=False)])
    return m, m.predict(Xva, num_iteration=m.best_iteration)
def train_clf(Xtr,ytr,Xva,yva):
    cats=[c for c in CAT if c in Xtr]
    m=lgb.train(dict(P,objective='binary',metric='binary_logloss',num_leaves=63), lgb.Dataset(Xtr,ytr,categorical_feature=cats), 3000, valid_sets=[lgb.Dataset(Xva,yva)], callbacks=[lgb.early_stopping(100,verbose=False)])
    return m, m.predict(Xva, num_iteration=m.best_iteration)

if __name__=='__main__':
    res=[]
    for val in (11,12):
        t0=time.time()
        tr=pl.concat([load(val-2),load(val-1)],how='vertical_relaxed'); va=load(val)
        tr,va=add_te(tr,va)
        ytr=tr['TAXITIME_SEC_mvt'].to_numpy().astype(float); yva=va['TAXITIME_SEC_mvt'].to_numpy().astype(float)
        atr=(sec('AOBT_3_flt','BLOCK_TIME_UTC_mvt')>1200); anom_tr=tr.select(atr.fill_null(False)).to_numpy().ravel(); anom_va=va.select(atr.fill_null(False)).to_numpy().ravel()
        print(f'\n=== fold val={val}: train {tr.height:,} val {va.height:,} feat {time.time()-t0:.0f}s  anomaly rate tr {anom_tr.mean()*100:.2f}% va {anom_va.mean()*100:.2f}%')
        sets=[('prev(base+cong)',CAT+BASE+CONG),('+overtake/turn',CAT+BASE+CONG+NEW[:5]),('+TE',CAT+BASE+CONG+NEW)]
        for name,cols in sets:
            Xtr=to_X(tr,cols); cats={c:Xtr[c].cat.categories for c in CAT}; Xva=to_X(va,cols,cats)
            t0=time.time(); m,p=train_reg(Xtr,ytr,Xva,yva); r=rmse(p,yva)
            print(f'{name:18s} single L2      RMSE {r:6.1f}  RMSE(<=1h) {float(np.sqrt(np.mean(((p-yva)**2)[yva<=3600]))):6.1f}  iter {m.best_iteration} {time.time()-t0:.0f}s'); res.append((val,name,'single',round(r,1)))
            if name=='+TE':
                imp=sorted(zip(m.feature_name(),m.feature_importance('gain')),key=lambda x:-x[1]); print('  gain:',[(n,int(v/1e6)) for n,v in imp[:16]])
                pa=va.with_columns(pl.Series('p',p)).group_by('ADEP_mvt').agg(((pl.col('p')-pl.col('TAXITIME_SEC_mvt'))**2).mean().sqrt().round(0).alias('rmse')).sort('rmse'); print('  per-airport:',dict(zip(pa['ADEP_mvt'],pa['rmse'])))
                # two-stage: P(anomalous row) x anomalous-row regression + (1-P) x regular regression
                t0=time.time()
                c,pa_=train_clf(Xtr,anom_tr.astype(float),Xva,anom_va.astype(float))
                from sklearn.metrics import roc_auc_score; auc=roc_auc_score(anom_va,pa_)
                mn,pn=train_reg(Xtr[~anom_tr],ytr[~anom_tr],Xva,yva)
                ma_,pan=train_reg(Xtr[anom_tr],ytr[anom_tr],Xva,yva)
                mix=pa_*pan+(1-pa_)*pn
                print(f'{name:18s} two-stage mix  RMSE {rmse(mix,yva):6.1f}  clf AUC {auc:.3f}  normal-only RMSE {rmse(pn,yva):6.1f}  {time.time()-t0:.0f}s'); res.append((val,name,'two-stage',round(rmse(mix,yva),1),round(auc,3)))
                # blend + average with the single model
                print(f'{name:18s} avg(single,mix) RMSE {rmse((p+mix)/2,yva):6.1f}')
                res.append((val,name,'avg',round(rmse((p+mix)/2,yva),1)))
    print('\n=== summary'); [print(r) for r in res]
