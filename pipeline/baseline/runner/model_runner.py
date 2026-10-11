"""model_runner: runs baseline_models.py / schedule_match_classifier.py without modifying them, and saves every trained model to disk (first stage of the baseline rerun).

What it does (everything is patched at run time; the sources and default behaviour of the baseline scripts are unchanged; the subcommand 'basemodels' = baseline_models.py):
  1. lightgbm.train -> after training, the Booster is saved unchanged with save_model into $RERUN_MODEL_DIR (including the pandas category mapping), and one JSON line is appended to the manifest.
     Covers fit_reg and fit_lirf (including the 5-fold OOF small models) of baseline_models.py, the lirf2 multi-class model, and the 12 folds + full-year discriminators of schedule_match_classifier.py.
  2. baseline_models.fit_lirf -> additionally saves the recalibration coefficient coef, round count, columns and category tables from the closure as lirf_<seq>.json (the booster itself is saved by 1).
  3. catboost.CatBoostRegressor.fit -> saves a .cbm; also caches by "training-data fingerprint + all hyper-parameters": if the same RERUN_CB_CACHE directory already holds a model with the same fingerprint,
     it is loaded instead of retrained (CatBoost seed 0 of ensemble B and the single seed of ensemble A are identical on the same feature store, which saves about 1.4 hours).
  4. baseline_models.exp_meta -> only records the current member name, used to name the model files.
Usage:
  python model_runner.py basemodels <raw> <work> <stage...> [--threads 6] [--tag ensa_rerun]     # equivalent to python baseline_models.py ...
  python model_runner.py pi_sched <work> [--threads 6]                               # equivalent to python schedule_match_classifier.py ...
Environment variables: RERUN_MODEL_DIR (required, model output directory), RERUN_CTX (file-name prefix, e.g. ensa_rerun_lgb), RERUN_CB_CACHE (optional, CatBoost cache directory),
  RERUN_SAVE_LGB=0 turns off saving of LGB models (default on). All other environment variables (FEAT_SPLIT, LIRF_REFIT, PI, NSEED, ...) are passed through unchanged to baseline_models.py.
"""
import hashlib
import json
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)
os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')

MODEL_DIR = os.environ.get('RERUN_MODEL_DIR')
CTX = os.environ.get('RERUN_CTX', 'run')
CB_CACHE = os.environ.get('RERUN_CB_CACHE')
SAVE_LGB = os.environ.get('RERUN_SAVE_LGB', '1') == '1'
STATE = {'seq': 0, 'member': '-'}


def _log(*x):
    print('[modelrun]', *x, flush=True)


def _manifest(rec):
    rec = dict(rec, ctx=CTX, member=STATE['member'], t=time.strftime('%Y-%m-%dT%H:%M:%S'))
    with open(os.path.join(MODEL_DIR, 'manifest.jsonl'), 'a') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')


def _next(stem):
    STATE['seq'] += 1
    safe = ''.join(c if c.isalnum() or c in '._-' else '_' for c in f"{CTX}_{STATE['member']}_{stem}")
    return f"{STATE['seq']:04d}_{safe}"


def patch_lightgbm():
    import lightgbm as lgb
    orig = lgb.train

    def train(params, train_set, num_boost_round=100, *args, **kw):
        t0 = time.time()
        from_f32 = type(getattr(train_set, 'data', None)).__name__ == 'ndarray'  # fit_reg path with LOWMEM=1 in baseline_models.py
        b = orig(params, train_set, num_boost_round, *args, **kw)
        if SAVE_LGB:
            obj = str(params.get('objective', 'na'))
            name = _next(f"lgb_{obj}_s{params.get('seed', 'na')}") + '.txt'
            b.save_model(os.path.join(MODEL_DIR, name))
            if from_f32 and STATE.get('last_cats') is not None:  # the LOWMEM=1 path feeds float32 codes, so the category table is not in the model file and is saved as a sidecar json
                json.dump(dict(cols=STATE['last_cols'], cats=STATE['last_cats']), open(os.path.join(MODEL_DIR, name[:-4] + '.cats.json'), 'w'), ensure_ascii=False)
                STATE['last_cats'] = None
            _manifest(dict(kind='lgb', file=name, objective=obj, seed=params.get('seed'), num_trees=b.num_trees(),
                           best_iteration=b.best_iteration, rounds=num_boost_round, num_feature=b.num_feature(),
                           params={k: v for k, v in params.items() if k != 'num_threads'}, fit_sec=round(time.time() - t0, 1)))
        return b
    lgb.train = train


def _frame_fp(data, label):
    """Training-data fingerprint: column names + shape + sum of row hashes + label bytes. Used only to decide CatBoost cache hits."""
    import numpy as np
    import pandas as pd
    h = hashlib.sha256()
    h.update(repr((list(getattr(data, 'columns', [])), getattr(data, 'shape', None))).encode())
    if isinstance(data, pd.DataFrame):
        v = pd.util.hash_pandas_object(data, index=False).to_numpy()
        h.update(np.ascontiguousarray(v).tobytes())
    else:
        h.update(np.ascontiguousarray(np.asarray(data)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(label, dtype=np.float64)).tobytes())
    return h.hexdigest()


def patch_catboost():
    try:
        import catboost
    except ImportError:
        return
    Pool = catboost.Pool
    orig_init = Pool.__init__

    def pool_init(self, data, label=None, *a, **kw):
        fp = None
        if CB_CACHE and label is not None:
            try:
                fp = _frame_fp(data, label)
            except Exception as e:  # if the fingerprint fails, the cache is skipped; only speed is affected
                _log('pool fingerprint failed', repr(e))
        orig_init(self, data, label, *a, **kw)
        self._rerun_fp = fp
    Pool.__init__ = pool_init

    R = catboost.CatBoostRegressor
    orig_fit = R.fit

    def fit(self, X, y=None, *a, **kw):
        prm = {k: v for k, v in self.get_params().items() if k not in ('thread_count', 'verbose')}
        fp = getattr(X, '_rerun_fp', None)
        key = hashlib.sha256((str(fp) + json.dumps(prm, sort_keys=True, default=str)).encode()).hexdigest()[:24] if fp else None
        if key and CB_CACHE:
            cp = os.path.join(CB_CACHE, f'cb_{key}.cbm')
            if os.path.exists(cp):
                self.load_model(cp)
                _manifest(dict(kind='catboost', file=None, seq=_next('cb_cached'), cache_hit=cp, key=key, params=prm))
                _log('catboost cache hit', cp)
                return self
        t0 = time.time()
        orig_fit(self, X, y, *a, **kw)
        name = _next(f"cb_s{prm.get('random_seed')}_it{prm.get('iterations')}") + '.cbm'
        self.save_model(os.path.join(MODEL_DIR, name))
        if key and CB_CACHE:
            os.makedirs(CB_CACHE, exist_ok=True)
            tmp = os.path.join(CB_CACHE, f'cb_{key}.cbm.tmp')
            self.save_model(tmp)
            os.replace(tmp, os.path.join(CB_CACHE, f'cb_{key}.cbm'))
        _manifest(dict(kind='catboost', file=name, key=key, params=prm, fit_sec=round(time.time() - t0, 1)))
        return self
    R.fit = fit


def patch_basemodels(bmod):
    orig_f32 = bmod._f32

    def _f32(X):
        STATE['last_cols'] = list(X.columns)
        STATE['last_cats'] = {c: [str(v) for v in X[c].cat.categories] for c in X.columns if str(X[c].dtype) == 'category'}
        return orig_f32(X)
    bmod._f32 = _f32
    orig_meta = bmod.exp_meta

    def exp_meta(exp):
        STATE['member'] = exp
        return orig_meta(exp)
    bmod.exp_meta = exp_meta
    orig_lirf = bmod.fit_lirf

    def fit_lirf(*a, **kw):
        predict, info = orig_lirf(*a, **kw)
        try:
            cl = {}
            for k, cell in zip(predict.__code__.co_freevars, predict.__closure__):
                try:
                    cl[k] = cell.cell_contents
                except ValueError:  # without deep, full is not assigned in the closure
                    cl[k] = None
            rec = dict(coef=[float(x) for x in cl['coef']], it=int(cl['it']), cols=list(cl['cols']), deep=bool(cl['deep']),
                       iso=cl.get('cal') is not None, cats={k: [str(x) for x in v] for k, v in cl['cats'].items()}, info=info,
                       env={k: os.environ.get(k) for k in ('PI', 'LIRFSTAND', 'LIRF_REFIT')})
            name = _next('lirf') + '.json'
            json.dump(rec, open(os.path.join(MODEL_DIR, name), 'w'), ensure_ascii=False)
            _manifest(dict(kind='lirf_calib', file=name, note='booster = the previous record with kind=lgb objective=binary (not the 5-fold OOF one)'))
        except Exception as e:
            _log('lirf closure dump failed', repr(e))
        return predict, info
    bmod.fit_lirf = fit_lirf


def selftest():
    """Self-check of a few seconds: an LGB model with category columns can be read back from disk with identical predictions; a second CatBoost fit with the same data and parameters hits the cache with identical predictions."""
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor, Pool
    rng = np.random.default_rng(0); n = 2000
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': pd.Categorical(rng.choice(['x', 'y', 'z'], n))}); y = X['a'].to_numpy() * 3 + (X['b'] == 'y') * 2 + rng.normal(size=n)
    b = lgb.train(dict(objective='regression', verbose=-1, num_threads=2, seed=0), lgb.Dataset(X, y, categorical_feature=['b']), 20)
    rec = [json.loads(l) for l in open(os.path.join(MODEL_DIR, 'manifest.jsonl'))]
    b2 = lgb.Booster(model_file=os.path.join(MODEL_DIR, rec[-1]['file']))
    assert np.allclose(b.predict(X), b2.predict(X)), 'lgb predictions differ after save and reload'
    C = X.assign(b=X['b'].cat.codes.astype(np.int32))
    m1 = CatBoostRegressor(iterations=20, depth=3, verbose=0, random_seed=0, thread_count=2).fit(Pool(C, y, cat_features=['b']))
    m2 = CatBoostRegressor(iterations=20, depth=3, verbose=0, random_seed=0, thread_count=2).fit(Pool(C, y, cat_features=['b']))
    rec = [json.loads(l) for l in open(os.path.join(MODEL_DIR, 'manifest.jsonl'))]
    assert rec[-1].get('cache_hit'), 'catboost cache miss'
    assert np.allclose(m1.predict(C), m2.predict(C)), 'catboost predictions differ after cache reload'
    _log('SELFTEST OK', len(rec), 'manifest rows')


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('basemodels', 'pi_sched', 'selftest'):
        sys.exit('usage: model_runner.py basemodels|pi_sched|selftest <args...>')
    assert MODEL_DIR, 'RERUN_MODEL_DIR is not set'
    os.makedirs(MODEL_DIR, exist_ok=True)
    patch_lightgbm()
    patch_catboost()
    which, rest = sys.argv[1], sys.argv[2:]
    if which == 'selftest':
        assert CB_CACHE, 'selftest needs RERUN_CB_CACHE'
        selftest()
        return
    _log(which, rest, 'models ->', MODEL_DIR, 'cb_cache ->', CB_CACHE,
         {k: os.environ.get(k) for k in ('FEAT_SPLIT', 'LIRF_REFIT', 'PI', 'PI_NESTED', 'LIRFSTAND', 'NSEED', 'CATSEED', 'LOWMEM', 'ENS_OBJ', 'REUSE', 'MEMBERS', 'CATPART')})
    if which == 'pi_sched':
        sys.argv = [os.path.join(SRC, 'schedule_match_classifier.py')] + rest
        STATE['member'] = 'pi_sched'
        runpy.run_path(os.path.join(SRC, 'schedule_match_classifier.py'), run_name='__main__')
        return
    import baseline_models as bmod
    patch_basemodels(bmod)
    sys.argv = [os.path.join(SRC, 'baseline_models.py')] + rest
    bmod.main()


if __name__ == '__main__':
    main()
