"""feat_align (final phase): reorder the training rows of the new feature store to the MVT_ID order of the old feat.parquet.

Why: the 2025-side baseline arrays used by tabular_term.py and train_resid (OOF files, 12-fold OOF, residual-CV OOF, ...) are aligned by "position of the training row in feat"; the training-row order of the feature store rebuilt on final_ranking need not equal the old one (row order within the same instant is undefined in non-canonical mode). This script only reorders and changes no value:
  - Assertions: the sets of training-row (src != 99) MVT_IDs of the old and new store are equal and without duplicates; all ranking rows (src == 99) of the new store are at the end;
    by default the old columns must be a subset of the new columns (downstream reads by the old column names; --allow-missing-cols relaxes this).
  - Output: training rows = old order, ranking rows = original order of the new store appended at the end, column order = column order of the new store; written to <out>.tmp first, then renamed.
  - Report (--report): row counts, column differences, whether src of the training rows equals the old one row by row, number of unequal rows per common column on the training rows (diagnoses whether FEAT_SPLIT 0->1 changed the training rows).
  - --verify-equal F: compares the output with F column by column in column chunks (eq_missing) and exits 0 only if all are equal -- used as a regression check (with --new and --old both the old store, the output must equal the old store).
Usage: python feat_align.py --new <new feat.parquet> --old <old feat.parquet> --out <aligned feat.parquet> --report x.json
Dependencies: polars, numpy. Memory: the new store is read whole (about 3-4 GB) + the old store in column chunks; the identity check is of the same order.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

os.environ.setdefault('POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR', 'load_as_storage')
ID = 'MVT_ID_mvt'


def log(*x):
    print(time.strftime('%H:%M:%S'), *x, flush=True)


def colwise_neq(a_path, b_path, cols, rows_a=None, rows_b=None, chunk=16):
    """Number of unequal rows per column for the given columns of two parquet files a and b (row indices can be taken for each first); read in column chunks to limit memory."""
    out = {}
    for i in range(0, len(cols), chunk):
        ch = cols[i:i + chunk]
        x = pl.read_parquet(a_path, columns=ch)
        y = pl.read_parquet(b_path, columns=ch)
        if rows_a is not None:
            x = x[rows_a]
        if rows_b is not None:
            y = y[rows_b]
        for c in ch:
            u, v = x[c], y[c]
            if u.dtype != v.dtype:
                try:
                    v = v.cast(u.dtype)
                except Exception:
                    out[c] = dict(comparable=False, dtype_a=str(u.dtype), dtype_b=str(y[c].dtype)); continue
            same = u.eq_missing(v)
            if u.dtype.is_float():  # NaN equals NaN (does not depend on the NaN comparison semantics of the polars version)
                same = same | (u.is_nan() & v.is_nan()).fill_null(False)
            out[c] = int((~same).sum())
        del x, y
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True); ap.add_argument('--old', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--report'); ap.add_argument('--allow-missing-cols', action='store_true'); ap.add_argument('--verify-equal')
    ap.add_argument('--no-colcheck', action='store_true', help='skip the column-by-column comparison of the training rows (saves time)')
    a = ap.parse_args()
    rep = dict(new=a.new, old=a.old, out=a.out)
    sn, so = pl.read_parquet_schema(a.new), pl.read_parquet_schema(a.old)
    rep['only_new_cols'] = [c for c in sn if c not in so]; rep['only_old_cols'] = [c for c in so if c not in sn]
    assert a.allow_missing_cols or not rep['only_old_cols'], ('the old store has columns that the new store lacks (downstream may read them)', rep['only_old_cols'])

    o = pl.read_parquet(a.old, columns=[ID, 'src'])
    otr = o.filter(pl.col('src') != 99)
    oid = otr[ID].to_numpy()
    rep['old_rows'], rep['old_train'] = o.height, otr.height
    assert len(np.unique(oid)) == len(oid), 'duplicate MVT_ID in the training rows of the old store'

    n = pl.read_parquet(a.new)
    src = n['src'].to_numpy()
    ntr = int((src != 99).sum())
    assert (src[ntr:] == 99).all() and (src[:ntr] != 99).all(), 'ranking rows (src==99) of the new store are not all at the end'
    nid = n[ID].to_numpy()[:ntr]
    rep['new_rows'], rep['new_train'], rep['new_ranking'] = n.height, ntr, n.height - ntr
    assert len(np.unique(nid)) == ntr, 'duplicate MVT_ID in the training rows of the new store'
    assert ntr == len(oid), ('different number of training rows', ntr, len(oid))
    srt = np.argsort(nid); pos = srt[np.searchsorted(nid[srt], oid).clip(0, ntr - 1)]
    assert (nid[pos] == oid).all(), 'the MVT_ID sets of the old and new training rows differ'
    rep['train_already_same_order'] = bool((pos == np.arange(ntr)).all())
    rep['train_src_eq_old'] = bool((src[pos] == otr['src'].to_numpy()).all())
    idx = np.concatenate([pos, np.arange(ntr, n.height)])
    out = n[idx]
    del n
    tmp = a.out + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.write_parquet(tmp)
    del out
    os.replace(tmp, a.out)
    log('wrote', a.out, 'rows', len(idx), 'train same order already' if rep['train_already_same_order'] else 'train reordered')

    if not a.no_colcheck:  # training rows column by column: first ntr rows of the output (already in old order) vs training rows of the old store
        common = [c for c in sn if c in so and c != ID]
        otr_rows = np.flatnonzero(o['src'].to_numpy() != 99)
        neq = colwise_neq(a.out, a.old, common, rows_a=np.arange(ntr), rows_b=otr_rows)
        rep['train_cols_neq'] = {c: v for c, v in neq.items() if v}
        rep['train_cols_compared'] = len(common)
        log('train cols compared', len(common), 'with diff', len(rep['train_cols_neq']))
    ok = True
    if a.verify_equal:
        se = pl.read_parquet_schema(a.verify_equal)
        so_ = pl.read_parquet_schema(a.out)
        same_cols = list(se) == list(so_)
        rows_eq = pl.scan_parquet(a.verify_equal).select(pl.len()).collect().item() == len(idx)
        neq = colwise_neq(a.out, a.verify_equal, list(so_)) if same_cols and rows_eq else {}
        bad = {c: v for c, v in neq.items() if v}
        ok = same_cols and rows_eq and not bad
        rep['verify_equal'] = dict(ref=a.verify_equal, same_cols=same_cols, rows_eq=rows_eq, cols_neq=bad, equal=ok)
        log('verify_equal', ok)
    if a.report:
        with open(a.report, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != 'train_cols_neq'}, ensure_ascii=False, default=str), flush=True)
    sys.exit(0 if ok else 3)


if __name__ == '__main__':
    main()
