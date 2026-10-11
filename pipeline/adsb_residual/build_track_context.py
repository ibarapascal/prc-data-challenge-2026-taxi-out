"""Per-row table of the track-context features (tk_* trajectory features before the start of the ground segment; label-free).

Subcommands (run from the working directory):
  main --rows gap/track_rows.parquet --feats feats_tma --out v22/tk_main.parquet
      Takes the tk_* columns from track_rows (the per-row table of 124 days: 62 training days of 2025-01/07 plus 62 days
      of 2026-01/07, computed from the ab_hex / ab_last_ground of feats_tm). Checks that ab_hex / ab_last_ground of
      feats_tma (the anchored features used by the simpler residual variant) agree with track_rows (anchoring changes only
      the segment start, not the matching or the take-off time; tk_* depends only on these two columns plus the track).
      Rows that disagree are recomputed from feats_tma with track_context_rows.track_frame and replaced. Also reports
      the agreement of tk_first with ab_obs_first on unanchored rows (the number of rows differing by > 1 s should be ~0).
  pool --feats feats_tmafy --months 11,12 --out v22/tk_pool.parquet
      For the winter-pool months (RESID_WINTER = 11+12), computes tk_* from the ab_hex / ab_last_ground of feats_tmafy
      with the same track_frame.
Dependencies: track_context_rows.py, train_resid.py, build_feats.py, type_chk.py and airports.json in the same directory.
TC_CODE_DIR is forced to this directory so that track_context_rows imports this directory's train_resid and build_feats.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
os.environ['TC_CODE_DIR'] = str(HERE)
sys.path.insert(0, str(HERE))
import track_context_rows as H  # noqa: E402

assert Path(H.T.__file__).resolve().parent == HERE, (H.T.__file__, HERE)
assert Path(H.BF.__file__).resolve().parent == HERE, (H.BF.__file__, HERE)
TK_NUM = [c for c in H.T.TK_ALL if c != 'tk_stop_code']
SEG = ['MVT_ID_mvt', 'date', 'ADEP_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt', 'ab_hex', 'ab_last_ground']


def stand_tables():
    return H.BF.load_stand_tables(H.STANDS, str(Path(H.STANDS).with_name('xplane_runways.csv')))


def recompute(seg):
    """seg: SEG columns (ab_hex / ab_last_ground non-null) -> per-row tk_* table (column names as in track_rows)."""
    tk = H.track_frame(seg, stand_tables())
    return tk.rename({c: f'tk_{c}' for c in tk.columns if c not in ('MVT_ID_mvt', 'tk_ok')})


def read_feats(d, pattern, cols):
    fs = sorted(Path(d).glob(pattern))
    out = []
    for f in fs:
        have = pl.read_parquet_schema(f)
        out.append(pl.read_parquet(f, columns=[c for c in cols if c in have]))
    return pl.concat(out, how='diagonal_relaxed').unique('MVT_ID_mvt', keep='first', maintain_order=True), len(fs)


def cmd_main(a):
    t0 = time.time()
    R = pl.read_parquet(a.rows)
    tkc = [c for c in R.columns if c.startswith('tk_')]
    assert set(TK_NUM + ['tk_stop']) <= set(tkc), sorted(set(TK_NUM + ['tk_stop']) - set(tkc))
    R = R.select(['MVT_ID_mvt', 'yr'] + [c for c in SEG if c != 'MVT_ID_mvt'] + ['gate'] + tkc)
    assert R['MVT_ID_mvt'].n_unique() == R.height, 'track_rows has duplicate MVT_ID'
    F, nf = read_feats(a.feats, '20*.parquet', ['MVT_ID_mvt', 'ab_hex', 'ab_last_ground', 'ab_obs_first', 'ab_anchor'])
    F = F.rename({c: f'fa_{c}' for c in F.columns if c != 'MVT_ID_mvt'})
    J = R.join(F, on='MVT_ID_mvt', how='left')
    hx = pl.col('ab_hex').cast(pl.String)
    fhx = pl.col('fa_ab_hex').cast(pl.String)
    same_hex = (hx.is_null() & fhx.is_null()) | (hx == fhx).fill_null(False)
    lg, flg = pl.col('ab_last_ground').cast(pl.Float64), pl.col('fa_ab_last_ground').cast(pl.Float64)
    same_lg = (lg.is_null() & flg.is_null()) | ((lg - flg).abs() <= 0.01).fill_null(False)
    J = J.with_columns((~(same_hex & same_lg)).alias('mismatch'))
    info = dict(rows=R.height, feats_files=nf, rows_in_feats=int(J['fa_ab_hex'].is_not_null().sum()),
                mismatch=int(J['mismatch'].sum()), mismatch_by_yr={str(k): int(v) for k, v in
                                                                   J.filter(pl.col('mismatch')).group_by('yr').len().iter_rows()})
    print('main check', info, f'{time.time() - t0:.0f}s', flush=True)
    out = J.select(['MVT_ID_mvt', 'yr', 'date', 'ADEP_mvt'] + tkc)
    if info['mismatch']:
        bad = J.filter(pl.col('mismatch') & pl.col('fa_ab_hex').is_not_null() & pl.col('fa_ab_last_ground').is_not_null())
        seg = bad.select([c for c in SEG if not c.startswith('ab_')] + [pl.col('fa_ab_hex').alias('ab_hex'),
                                                                         pl.col('fa_ab_last_ground').alias('ab_last_ground')])
        fix = recompute(seg) if seg.height else pl.DataFrame({'MVT_ID_mvt': []})
        # Disagreeing rows: set all tk columns to null first, then overwrite with the values recomputed from feats_tma (rows that cannot be recomputed stay null)
        mm = J['mismatch'].to_numpy()
        out = out.with_columns([pl.when(pl.Series(mm)).then(None).otherwise(pl.col(c)).alias(c) for c in tkc])
        if fix.height:
            keep = [c for c in tkc if c in fix.columns]
            out = out.with_row_index('_r').join(fix.select(['MVT_ID_mvt'] + keep).rename({c: f'_n_{c}' for c in keep}),
                                                on='MVT_ID_mvt', how='left').sort('_r')
            out = out.with_columns([pl.coalesce(pl.col(f'_n_{c}'), pl.col(c)).alias(c) for c in keep]).drop(
                ['_r'] + [f'_n_{c}' for c in keep])
        info['recomputed'] = int(fix.height)
    # Consistency: on unanchored rows tk_first should equal ab_obs_first of feats_tma (no row differs by > 1 s)
    chk = J.filter((pl.col('tk_ok') == 1) & (pl.col('fa_ab_anchor').fill_null(0) != 1) & ~pl.col('mismatch'))
    info['first_vs_tma_gt1s_unanchored'] = int(((chk['tk_first'] - chk['fa_ab_obs_first']).abs() > 1).sum())
    info['n_check'] = chk.height
    cov = out.join(R.select('MVT_ID_mvt', 'gate'), on='MVT_ID_mvt').filter(pl.col('gate')).with_columns(
        pl.concat_str([pl.col('ADEP_mvt'), pl.col('date').str.slice(0, 7)], separator='|').alias('k'))
    info['gated_tk_stop_cov'] = {k: round(float(v), 3) for k, v in
                                 cov.group_by('k').agg(pl.col('tk_stop').is_not_null().mean()).sort('k').iter_rows()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    json.dump(info, open(a.out.replace('.parquet', '.json'), 'w'), indent=1, ensure_ascii=False)
    print('wrote', a.out, out.height, info, f'{time.time() - t0:.0f}s', flush=True)


def cmd_pool(a):
    t0 = time.time()
    months = [m for m in a.months.split(',') if m]
    full = pl.read_parquet('n2/feat.parquet', columns=['MVT_ID_mvt', 'ADEP_mvt', 'STAND_mvt', 'src', 'MVT_TIME_UTC_mvt']).filter(
        pl.col('src') != 99).drop('src').unique('MVT_ID_mvt', keep='first', maintain_order=True)
    F, nf = read_feats(a.feats, '2025-*.parquet', ['MVT_ID_mvt', 'ab_hex', 'ab_last_ground', 'ab_obs_first', 'ab_anchor'])
    rows = F.join(full, on='MVT_ID_mvt', how='inner').with_columns(
        pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d').alias('date'))
    rows = rows.filter(pl.col('date').str.slice(5, 2).is_in(months))
    seg = rows.filter(pl.col('ab_hex').is_not_null() & pl.col('ab_last_ground').is_not_null()).select(SEG)
    print('pool rows', rows.height, 'with segment', seg.height, 'feats files', nf, f'{time.time() - t0:.0f}s', flush=True)
    tk = recompute(seg)
    out = rows.select('MVT_ID_mvt', 'date', 'ADEP_mvt', 'ab_obs_first', 'ab_anchor' if 'ab_anchor' in rows.columns else pl.lit(None).alias(
        'ab_anchor')).join(tk, on='MVT_ID_mvt', how='left')
    chk = out.filter((pl.col('tk_ok') == 1) & (pl.col('ab_anchor').fill_null(0) != 1))
    info = dict(months=months, rows=out.height, seg_rows=seg.height, track_ok=int((out['tk_ok'] == 1).sum()),
                first_vs_feats_gt1s_unanchored=int(((chk['tk_first'] - chk['ab_obs_first']).abs() > 1).sum()), n_check=chk.height,
                stop_share={str(k): int(v) for k, v in out.group_by('tk_stop').len().iter_rows()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.drop('ab_obs_first', 'ab_anchor').write_parquet(a.out)
    json.dump(info, open(a.out.replace('.parquet', '.json'), 'w'), indent=1, ensure_ascii=False)
    print('wrote', a.out, info, f'{time.time() - t0:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['main', 'pool'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', default='gap/track_rows.parquet')
    ap.add_argument('--feats', default='feats_tma')
    ap.add_argument('--months', default='11,12')
    a = ap.parse_args()
    print('code', HERE, 'train_resid', H.T.__file__, 'build_feats', H.BF.__file__, flush=True)
    {'main': cmd_main, 'pool': cmd_pool}[a.cmd](a)


if __name__ == '__main__':
    main()
