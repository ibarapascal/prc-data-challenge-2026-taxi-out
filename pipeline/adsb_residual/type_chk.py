"""Aircraft-type mismatch flag for time-matched events: compares the aircraft type of the paired ADS-B event with
AIRCRAFT_TYPE_mvt row by row and flags a mismatch by aircraft-type family.

Usage (from the working directory): python type_chk.py --feats feats feats_fy --out typechk.parquet
Output: MVT_ID_mvt, cs_match, ds_type, actype, fam_ds, fam_ab, tmis (True only for pure time matches with
cs_match in {0, 2}, both types present and the families different).
Type families: derivatives of one series are merged (A320 family, B737 family, E-Jet, CRJ, A330, B777, B787, A350,
ATR, Dash 8, A220, ...); all other types use their first 3 characters.
The ADS-B type is the actype seen most often for that hex on the cropped day (only the hex/actype columns are read, day by day).
"""
import argparse
import re
from pathlib import Path

import polars as pl

FAM = [(r'^(A318|A319|A320|A321|A19N|A20N|A21N)$', 'A32F'), (r'^(B73\d|B3[789]M|B37M)$', 'B73F'),
       (r'^(E17\d|E19\d|E75\w|E29\d)$', 'EJET'), (r'^CRJ\w$', 'CRJF'), (r'^A33\w$', 'A33F'), (r'^B77\w$', 'B77F'),
       (r'^B78\w$', 'B78F'), (r'^A35\w$', 'A35F'), (r'^AT[4-7]\w$', 'ATRF'), (r'^DH8\w$', 'DH8F'), (r'^BCS\d$', 'A22F'),
       (r'^B76\w$', 'B76F'), (r'^B74\w$', 'B74F'), (r'^A34\w$', 'A34F'), (r'^A38\w$', 'A38F'), (r'^E1[34]\w$', 'ERJF')]


def fam(t):
    if not t:
        return None
    for pat, f in FAM:
        if re.match(pat, t):
            return f
    return t[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', nargs='+', default=['feats', 'feats_fy'])
    ap.add_argument('--raw', default='raw')
    ap.add_argument('--out', default='typechk.parquet')
    a = ap.parse_args()
    fs = [f for d in a.feats for f in sorted(Path(d).glob('*.parquet'))]
    ab = pl.concat([pl.read_parquet(f, columns=['MVT_ID_mvt', 'ab_hex', 'ab_cs_match', 'date']) for f in fs],
                   how='vertical_relaxed').drop_nulls('ab_hex').unique('MVT_ID_mvt').with_columns(pl.col('ab_hex').cast(pl.String))
    rf = sorted(Path(a.raw).glob('training_2025-*.parquet')) + [Path(a.raw) / 'ranking.parquet']
    ds = pl.concat([pl.read_parquet(f, columns=['MVT_ID_mvt', 'AIRCRAFT_TYPE_mvt']) for f in rf], how='vertical_relaxed') \
           .unique('MVT_ID_mvt').join(ab.select('MVT_ID_mvt'), on='MVT_ID_mvt', how='semi') \
           .with_columns(pl.col('AIRCRAFT_TYPE_mvt').cast(pl.String).str.strip_chars().str.to_uppercase().alias('ds_type')).drop('AIRCRAFT_TYPE_mvt')
    ht = []
    for dt in sorted(ab['date'].unique().to_list()):
        f = Path('crop') / dt[:4] / f'{dt}.parquet'
        if not f.exists():
            continue
        p = pl.read_parquet(f, columns=['hex', 'actype']).drop_nulls('actype').with_columns(
            pl.col('hex').cast(pl.String), pl.col('actype').str.strip_chars().str.to_uppercase())
        ht.append(p.group_by('hex', 'actype').len().sort('len', descending=True).unique('hex', keep='first')
                   .select('hex', 'actype').with_columns(pl.lit(dt).alias('date')))
    ht = pl.concat(ht)
    v = ab.join(ds, on='MVT_ID_mvt', how='left').join(ht, left_on=['ab_hex', 'date'], right_on=['hex', 'date'], how='left') \
          .with_columns(pl.col('ds_type').map_elements(fam, return_dtype=pl.String).alias('fam_ds'),
                        pl.col('actype').map_elements(fam, return_dtype=pl.String).alias('fam_ab'))
    v = v.with_columns((pl.col('ab_cs_match').is_in([0, 2]) & pl.col('fam_ds').is_not_null() & pl.col('fam_ab').is_not_null()
                        & (pl.col('fam_ds') != pl.col('fam_ab'))).alias('tmis'))
    v.select('MVT_ID_mvt', 'date', 'ab_cs_match', 'ds_type', 'actype', 'fam_ds', 'fam_ab', 'tmis').write_parquet(a.out)
    s = v.with_columns(pl.col('date').str.slice(0, 7).alias('ym')).group_by('ym').agg(
        pl.len().alias('matched'), pl.col('ab_cs_match').is_in([0, 2]).sum().alias('time'), pl.col('tmis').sum().alias('tmis')).sort('ym')
    print(s, flush=True)


if __name__ == '__main__':
    main()
