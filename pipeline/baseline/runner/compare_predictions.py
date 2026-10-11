"""compare_predictions: row-by-row comparison of rerun outputs with earlier outputs. Read-only, no training; memory < 2 GB (feature stores are read in column chunks).

Subcommands:
  subs     --pairs name=new.parquet:ref.parquet[,...] [--keys ranking.parquet] --out x.json
           Prediction files aligned by MVT_ID: row counts, whether the ID sets agree, n_diff, max_abs, RMS, mean difference, correlation; with --keys also reports airport x month.
  members  --new-feat F --new-dir D --ref-feat F0 --ref-map name=old.npy[,...] [--new-map name=new.npy,...] --out x.json
           Member-level full-year predictions (full_<name>.npy, float, row order = order of src==99 rows in the respective feat.parquet) compared after aligning by MVT_ID.
  feat     --new F --ref F0 --out x.json [--chunk 12]
           Two feature stores aligned by MVT_ID and compared column by column (missing values count as equal); reports unequal rows per column and the maximum absolute difference of numeric columns.
"""
import argparse
import json

import numpy as np
import polars as pl


def stats(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    fin = np.isfinite(d)
    out = dict(n=int(len(d)), n_nonfinite=int((~fin).sum()))
    d = d[fin]; a2 = a[fin].astype(np.float64); b2 = b[fin].astype(np.float64)
    out.update(n_diff=int((d != 0).sum()), max_abs=float(np.abs(d).max()) if len(d) else 0.0,
               rms=float(np.sqrt((d ** 2).mean())) if len(d) else 0.0, mean_diff=float(d.mean()) if len(d) else 0.0,
               corr=float(np.corrcoef(a2, b2)[0, 1]) if len(d) > 1 and a2.std() > 0 and b2.std() > 0 else None,
               mean_new=float(a2.mean()) if len(d) else None, mean_ref=float(b2.mean()) if len(d) else None)
    return out


def align(ids_new, ids_ref):
    """Returns the index perm into ref such that ids_ref[perm]==ids_new; requires identical sets without duplicates."""
    assert len(np.unique(ids_new)) == len(ids_new) and len(np.unique(ids_ref)) == len(ids_ref), 'duplicate MVT_ID'
    assert len(ids_new) == len(ids_ref), ('different row counts', len(ids_new), len(ids_ref))
    o = np.argsort(ids_ref); pos = np.searchsorted(ids_ref[o], ids_new)
    pos = np.clip(pos, 0, len(o) - 1); perm = o[pos]
    assert (ids_ref[perm] == ids_new).all(), 'MVT_ID sets differ'
    return perm


def cmd_subs(a):
    rep = {}
    keys = pl.read_parquet(a.keys, columns=['MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt']).filter(pl.col('PHASE_mvt') == 'DEP') if a.keys else None
    for item in a.pairs.split(','):
        name, pr = item.split('=', 1); pn, pf = pr.split(':', 1)
        n = pl.read_parquet(pn); r = pl.read_parquet(pf)
        j = n.join(r, on='MVT_ID_mvt', how='full', suffix='_ref', coalesce=True)
        both = j.drop_nulls(['TAXITIME_SEC_mvt', 'TAXITIME_SEC_mvt_ref'])
        s = stats(both['TAXITIME_SEC_mvt'].to_numpy(), both['TAXITIME_SEC_mvt_ref'].to_numpy())
        s.update(rows_new=n.height, rows_ref=r.height, only_new=int(j['TAXITIME_SEC_mvt_ref'].null_count()), only_ref=int(j['TAXITIME_SEC_mvt'].null_count()),
                 same_row_order=bool(n.height == r.height and (n['MVT_ID_mvt'] == r['MVT_ID_mvt']).all()))
        if keys is not None:
            k = both.join(keys, on='MVT_ID_mvt', how='left').with_columns((pl.col('TAXITIME_SEC_mvt').cast(pl.Float64) - pl.col('TAXITIME_SEC_mvt_ref')).alias('d'),
                                                                          pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m').alias('ym'))
            g = k.group_by('ADEP_mvt', 'ym').agg(pl.len().alias('n'), pl.col('d').mean().alias('mean_diff'), (pl.col('d') ** 2).mean().sqrt().alias('rms'),
                                                 pl.col('d').abs().max().alias('max_abs')).sort('ADEP_mvt', 'ym')
            s['by_apt_ym'] = {f"{r_['ADEP_mvt']}-{r_['ym']}": {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r_.items() if kk not in ('ADEP_mvt', 'ym')} for r_ in g.iter_rows(named=True)}
        rep[name] = s
        print(name, {k: s[k] for k in ('rows_new', 'rows_ref', 'only_new', 'only_ref', 'n_diff', 'max_abs', 'rms', 'mean_diff', 'corr')}, flush=True)
    json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)


def rank_ids(feat):
    return pl.read_parquet(feat, columns=['MVT_ID_mvt', 'src']).filter(pl.col('src') == 99)['MVT_ID_mvt'].to_numpy()


def cmd_members(a):
    idn = rank_ids(a.new_feat); idr = rank_ids(a.ref_feat); perm = align(idn, idr)
    newmap = dict(x.split('=', 1) for x in a.new_map.split(',')) if a.new_map else {}
    rep = {}
    for item in a.ref_map.split(','):
        name, old = item.split('=', 1)
        newp = newmap.get(name, f'{a.new_dir}/full_{name}.npy')
        try:
            pn = np.load(newp); po = np.load(old)
        except FileNotFoundError as e:
            rep[name] = dict(error=str(e)); continue
        assert len(pn) == len(idn) and len(po) == len(idr), (name, len(pn), len(po))
        rep[name] = dict(stats(pn, po[perm]), new=newp, ref=old)
        print(name, {k: rep[name][k] for k in ('n_diff', 'max_abs', 'rms', 'mean_diff', 'corr')}, flush=True)
    json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)


def cmd_feat(a):
    sn = pl.read_parquet_schema(a.new); sr = pl.read_parquet_schema(a.ref)
    idn = pl.read_parquet(a.new, columns=['MVT_ID_mvt'])['MVT_ID_mvt'].to_numpy(); idr = pl.read_parquet(a.ref, columns=['MVT_ID_mvt'])['MVT_ID_mvt'].to_numpy()
    rep = dict(rows_new=len(idn), rows_ref=len(idr), only_new_cols=[c for c in sn if c not in sr], only_ref_cols=[c for c in sr if c not in sn], cols={})
    perm = align(idn, idr); cols = [c for c in sn if c in sr and c != 'MVT_ID_mvt']
    for i in range(0, len(cols), a.chunk):
        ch = cols[i:i + a.chunk]
        n = pl.read_parquet(a.new, columns=ch); r = pl.read_parquet(a.ref, columns=ch)[perm]
        for c in ch:
            x, y = n[c], r[c]
            if x.dtype != y.dtype:
                try:
                    y = y.cast(x.dtype)
                except Exception:
                    rep['cols'][c] = dict(dtype_new=str(x.dtype), dtype_ref=str(r[c].dtype), comparable=False); continue
            neq = int((~x.eq_missing(y)).sum())
            d = dict(n_neq=neq, dtype=str(x.dtype))
            if x.dtype.is_numeric() and neq:
                dd = (x.cast(pl.Float64) - y.cast(pl.Float64)).abs()
                d['max_abs'] = float(dd.max()) if dd.max() is not None else None
                d['n_null_mismatch'] = int((x.is_null() != y.is_null()).sum())
            rep['cols'][c] = d
        del n, r
    rep['n_cols_neq'] = sum(1 for v in rep['cols'].values() if v.get('n_neq'))
    print('feat cols compared', len(rep['cols']), 'cols with any diff', rep['n_cols_neq'], flush=True)
    json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('subs'); s.add_argument('--pairs', required=True); s.add_argument('--keys'); s.add_argument('--out', required=True)
    s = sp.add_parser('members'); s.add_argument('--new-feat', required=True); s.add_argument('--new-dir', required=True); s.add_argument('--ref-feat', required=True)
    s.add_argument('--ref-map', required=True); s.add_argument('--new-map', default=''); s.add_argument('--out', required=True)
    s = sp.add_parser('feat'); s.add_argument('--new', required=True); s.add_argument('--ref', required=True); s.add_argument('--out', required=True); s.add_argument('--chunk', type=int, default=12)
    a = ap.parse_args()
    {'subs': cmd_subs, 'members': cmd_members, 'feat': cmd_feat}[a.cmd](a)


if __name__ == '__main__':
    main()
