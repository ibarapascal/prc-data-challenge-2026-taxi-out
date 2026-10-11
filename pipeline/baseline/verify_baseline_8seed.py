"""verify_baseline_8seed: check the relation between the 2026-side 8-seed baseline (ensemble B with CATSEED=8; tags ensavg_8seed/ensb_8seed) and the 3-seed baseline (tags ensavg_3seed/ensb_3seed; "RUN3SEED" below).

Checks (any hard check failing -> exit 1):
1. CatBoost cache: in the manifest of the new work directory, both cat targets (2846 / 3436 rounds) hit the cache for seeds 0-2, and the cache keys equal those of the same seeds in the 3-seed manifest (= same training-data fingerprint + hyper-parameters = the original models);
   seeds 3-7 are each trained once.
2. Member agreement: the ensemble-B members other than full_cat_delta (srp_g5_delta, anc_iobt, apt_delta) are element-wise equal to the 3-seed run.
3. Propagation: pred_ensb_8seed - pred_ensb_3seed ~ w_cat*(full_cat_delta_new - full_cat_delta_RUN3SEED) (w_cat = .286/1.001), max residual < 1e-6.
Only reports: per-month row-wise differences of ensb_8seed vs ensb_3seed and ensavg_8seed vs ensavg_3seed (changed rows, mean difference, RMS, P95 of |difference|, max).
Usage: python verify_baseline_8seed.py --run3seed <3-seed work dir> --w <new work dir> --out <json>
"""
import argparse
import json
import os

import numpy as np
import polars as pl
import pyarrow.parquet as pq


def cat_records(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get('kind') != 'catboost' or not str(r.get('ctx', '')).endswith(('cat_taxi', 'cat_delta')):
            continue
        p = r['params']; k = (p['iterations'], p['random_seed'])
        out.setdefault(k, []).append(dict(hit=r.get('cache_hit') is not None, key=r.get('key'), file=r.get('file'), ctx=r.get('ctx')))
    return out


def sub(p):
    t = pq.read_table(p)
    return t.column(0).to_numpy().astype(np.float64), t.column(1).to_numpy().astype(np.float64)


def stats(d):
    ad = np.abs(d)
    return dict(n=int(len(d)), changed=int((d != 0).sum()), mean=round(float(d.mean()), 4), rms=round(float(np.sqrt((d ** 2).mean())), 4),
                p95_abs=round(float(np.quantile(ad, .95)), 3), max_abs=round(float(ad.max()), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run3seed', required=True); ap.add_argument('--w', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rep = dict(ok={}, run3seed=a.run3seed, w=a.w)
    old = cat_records(os.path.join(a.run3seed, 'models', 'ensb_3seed', 'manifest.jsonl'))
    new = cat_records(os.path.join(a.w, 'models', 'ensb_8seed', 'manifest.jsonl'))
    cache = {}
    ok_cache = True
    for it in (2846, 3436):
        for s in range(8):
            n = new.get((it, s), []); o = old.get((it, s), [])
            cache[f'{it}_s{s}'] = dict(new=n, old_key=[x['key'] for x in o])
            if s < 3:
                ok_cache &= len(n) == 1 and n[0]['hit'] and len(o) == 1 and n[0]['key'] == o[0]['key']
            else:
                ok_cache &= len(n) == 1 and not n[0]['hit'] and n[0]['file'] is not None
    rep['cache'] = cache; rep['ok']['cache_seeds012_hit_eq_run3seed_and_345_67_trained'] = bool(ok_cache)

    ok_mem = True; mem = {}
    for m in ('srp_g5_delta', 'anc_iobt', 'apt_delta'):
        x = np.load(os.path.join(a.w, 'ensb', f'full_{m}.npy')); y = np.load(os.path.join(a.run3seed, 'ensb', f'full_{m}.npy'))
        eq = bool(x.shape == y.shape and np.array_equal(x, y, equal_nan=True)); mem[m] = eq; ok_mem &= eq
    rep['members_equal'] = mem; rep['ok']['other_members_equal'] = bool(ok_mem)

    cn = np.load(os.path.join(a.w, 'ensb', 'full_cat_delta.npy')); co = np.load(os.path.join(a.run3seed, 'ensb', 'full_cat_delta.npy'))
    pn = np.load(os.path.join(a.w, 'ensb', 'pred_ensb_8seed.npy')); po = np.load(os.path.join(a.run3seed, 'ensb', 'pred_ensb_3seed.npy'))
    w = 0.286 / 1.001
    resid = (pn - po) - w * (cn - co)
    rep['cat_member_diff'] = stats(cn - co)
    rep['propagation'] = dict(w_cat=w, max_resid=float(np.nanmax(np.abs(resid))), rows_cat_equal=int((cn == co).sum()))
    rep['ok']['propagation'] = bool(np.nanmax(np.abs(resid)) < 1e-6)

    rk = (pl.read_parquet(os.path.join(a.w, 'raw', 'ranking.parquet'), columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt'])
          .filter(pl.col('PHASE_mvt') == 'DEP').select(pl.col('MVT_ID_mvt').cast(pl.Float64), pl.col('MVT_TIME_UTC_mvt').dt.month().alias('mo')))
    for nm, fn, fo in (('ensb', 'upstanding-firefly_ensb_8seed.parquet', 'upstanding-firefly_ensb_3seed.parquet'), ('ensavg', 'upstanding-firefly_ensavg_8seed.parquet', 'upstanding-firefly_ensavg_3seed.parquet')):
        idn, yn = sub(os.path.join(a.w, 'sub', fn)); ido, yo = sub(os.path.join(a.run3seed, 'sub', fo))
        assert np.array_equal(idn, ido), f'{nm}: row order differs'
        mo = pl.DataFrame({'MVT_ID_mvt': idn}).join(rk, on='MVT_ID_mvt', how='left')['mo'].to_numpy()
        d = yn - yo
        rep[f'{nm}_new_minus_run3seed'] = dict(all=stats(d), by_month={str(int(m)): stats(d[mo == m]) for m in np.unique(mo)},
                                         corr=round(float(np.corrcoef(yn, yo)[0, 1]), 7), mean_new=round(float(yn.mean()), 3), mean_run3seed=round(float(yo.mean()), 3))
    rep['ok_all'] = all(rep['ok'].values())
    json.dump(rep, open(a.out + '.tmp', 'w'), ensure_ascii=False, indent=1); os.replace(a.out + '.tmp', a.out)
    print(json.dumps(dict(ok=rep['ok'], cat=rep['cat_member_diff'], prop=rep['propagation'], ensavg=rep['ensavg_new_minus_run3seed']), ensure_ascii=False))
    if not rep['ok_all']:
        raise SystemExit('ensavg_8seed verification failed')


if __name__ == '__main__':
    main()
