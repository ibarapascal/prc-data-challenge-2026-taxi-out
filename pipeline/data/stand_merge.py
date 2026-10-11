"""Stand-table merge: X-Plane Scenery Gateway apt.dat stands + OpenStreetMap stands -> matched to the STAND values of the
challenge data under a normalised stand name; writes the merged stand coordinate table, the apt.dat runway table and coverage reports.

Usage:
  python stand_merge.py --xplane-dir <dir> --osm-dir <dir> --raw <dir> [--airports airports.json] [--storedir <dir>]
    --xplane-dir  directory with <ICAO>/apt.dat for the ten airports (written by fetch_xplane_apt.sh); all outputs are written here
    --osm-dir     directory with the Overpass API JSON files <ICAO>.json (see README, section 1)
    --raw         directory with the organiser files training_*.parquet and ranking.parquet (DEP rows give the STAND values)
    --airports    airports.json of this repository (default: the copy next to this script); only used for the report column
                  in_airports_json (stand name found by the upper-case exact match on the airports.json stand list)
    --storedir    optional: baseline feature-store directory with feat.parquet and the two-fold out-of-fold files
                  oof/ens_v5b_{1,7}.npy and oof/ens_v6_{1,7}.npy; when given, also writes oof_by_cov.csv (report only)
Outputs (in --xplane-dir): stand_coords_merged.parquet and .csv (read by build_feats.py --stands and build_tabular_features.py),
  xplane_runways.csv (runway ends from apt.dat row code 100), xplane_stands.csv (all apt.dat stands, row code 1300), coverage.csv,
  summary.json, and oof_by_cov.csv with --storedir.
Normalisation canon(): upper case; drop bracketed parts; drop a leading word GATE/RAMP/STAND/PARKING/POSITION/POS/APRON;
  drop spaces, hyphens and underscores; for the form letters + digits + letters, strip leading zeros of the digits
  (H01 -> H1, Ramp 01 -> 1). OSM ref values are first split into aliases at / + , ;.
  Match level 1 = identical canon key; level 2 = the data value without its letter suffix equals a source key (G11L -> G11),
  or the source has sub-stands of the same base number (202 -> 202A/202B), whose centroid is used.
  Several points for one key: points of the same source and the same priority within 150 m are averaged, otherwise the key is
  ambiguous and dropped (OSM parking_position takes priority over gate). apt.dat matches take priority over OSM at the same level.
Dependencies: numpy, polars.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import polars as pl

APTS = ['EGLL', 'LEMD', 'LIRF', 'LTFM', 'LSZH', 'LFPG', 'LEBL', 'EDDF', 'EHAM', 'EDDM']
SPREAD = 150.0

ap_ = argparse.ArgumentParser()
ap_.add_argument('--xplane-dir', required=True)
ap_.add_argument('--osm-dir', required=True)
ap_.add_argument('--raw', required=True)
ap_.add_argument('--airports', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'airports.json'))
ap_.add_argument('--storedir', default=None)
A = ap_.parse_args()
XP = A.xplane_dir


def hav(a, b):
    r = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    return 2 * r * math.asin(math.sqrt(math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def canon(s):
    if s is None:
        return None
    s = re.sub(r'\(.*?\)', '', str(s).upper().strip())
    s = re.sub(r'^(GATE|RAMP|STAND|PARKING|POSITION|POS|APRON)\b\s*', '', s)
    s = re.sub(r'[\s\-_]', '', s)
    m = re.fullmatch(r'([A-Z]*)(\d+)([A-Z]*)', s)
    if m:
        s = m.group(1) + str(int(m.group(2))) + m.group(3)
    return s or None


def base(s):
    m = re.fullmatch(r'([A-Z]*\d+)([A-Z]+)', s or '')
    return m.group(1) if m else None


def resolve(cands):
    """cands: [(prio, (lat, lon))] -> centroid, or None (ambiguous)."""
    best = min(p for p, _ in cands)
    pts = [c for p, c in cands if p == best]
    if max((hav(a, b) for a in pts for b in pts), default=0) >= SPREAD:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def build_index(raw):
    """raw: key -> [(prio, (lat, lon))] -> (resolved dict, ambiguous set, base -> keys)"""
    res, amb = {}, set()
    for k, c in raw.items():
        v = resolve(c)
        (res.__setitem__(k, v) if v else amb.add(k))
    bk = defaultdict(list)
    for k in res:
        b = base(k)
        if b:
            bk[b].append(k)
    return res, amb, bk


def lookup(idx, key):
    res, amb, bk = idx
    if key is None:
        return None, None
    if key in res:
        return res[key], 1
    b = base(key)
    if b and b in res:
        return res[b], 2
    kids = bk.get(key) or []
    if kids:
        pts = [res[k] for k in kids]
        if max(hav(a, c) for a in pts for c in pts) < SPREAD * 2:
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)), 2
    return None, None


# ---- X-Plane apt.dat ----
xp_rows, rw_rows, xp_idx = [], [], {}
for ap in APTS:
    raw = defaultdict(list)
    for line in open(os.path.join(XP, ap, 'apt.dat'), encoding='utf-8', errors='replace'):
        t = line.split()
        if not t:
            continue
        if t[0] == '1300' and len(t) >= 7:
            name = ' '.join(t[6:])
            lat, lon = float(t[1]), float(t[2])
            xp_rows.append(dict(apt=ap, name=name, key=canon(name), lat=lat, lon=lon, heading=float(t[3]), type=t[4], ac=t[5]))
            raw[canon(name)].append((0, (lat, lon)))
        elif t[0] == '100' and len(t) >= 20:
            for i in (8, 17):
                rw_rows.append(dict(apt=ap, rwy=t[i], lat=float(t[i + 1]), lon=float(t[i + 2]), displaced_m=float(t[i + 3])))
    xp_idx[ap] = build_index(raw)

# ---- OSM (raw Overpass output, indexed with the same rule) ----
osm_idx = {}
for ap in APTS:
    d = json.load(open(os.path.join(A.osm_dir, ap + '.json')))
    nodes = {e['id']: (e['lat'], e['lon']) for e in d['elements'] if e['type'] == 'node'}
    raw = defaultdict(list)
    for e in d['elements']:
        tg = e.get('tags', {})
        if tg.get('aeroway') not in ('parking_position', 'gate'):
            continue
        ref = tg.get('ref') or tg.get('name')
        if not ref:
            continue
        pts = [nodes[e['id']]] if e['type'] == 'node' else [nodes[n] for n in e.get('nodes', []) if n in nodes]
        if not pts:
            continue
        c = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        prio = 0 if tg['aeroway'] == 'parking_position' else 1
        for part in re.split(r'[/+,;]', ref):
            k = canon(part)
            if k:
                raw[k].append((prio, c))
    osm_idx[ap] = build_index(raw)

cur = json.load(open(A.airports))
cur_keys = {ap: {k.upper() for k in cur[ap]['stands']} for ap in APTS}

# ---- STAND values of the challenge data (DEP rows) ----
parts = []
for f in sorted(glob.glob(os.path.join(A.raw, 'training_*.parquet'))) + [os.path.join(A.raw, 'ranking.parquet')]:
    parts.append(pl.scan_parquet(f).filter(pl.col('PHASE_mvt') == 'DEP').group_by('ADEP_mvt', 'STAND_mvt').len()
                 .with_columns(pl.lit('rank' if 'ranking' in f else 'train').alias('src')).collect())
cnt = pl.concat(parts).group_by('src', 'ADEP_mvt', 'STAND_mvt').agg(pl.col('len').sum())

rows = []
for ap, st in cnt.select('ADEP_mvt', 'STAND_mvt').unique().iter_rows():
    k = canon(st) if st not in (None, 'NA', 'UNKNOWN') else None
    xc, xl = lookup(xp_idx[ap], k)
    oc, ol = lookup(osm_idx[ap], k)
    if xl == 1:
        src, c, lv = 'xplane', xc, 1
    elif ol == 1:
        src, c, lv = 'osm', oc, 1
    elif xl == 2:
        src, c, lv = 'xplane', xc, 2
    elif ol == 2:
        src, c, lv = 'osm', oc, 2
    else:
        src, c, lv = None, None, None
    rows.append(dict(apt=ap, stand=st, stand_key=k, lat=c[0] if c else None, lon=c[1] if c else None, source=src, match_level=lv,
                     xp_level=xl, osm_level=ol, xp_lat=xc[0] if xc else None, xp_lon=xc[1] if xc else None,
                     osm_lat=oc[0] if oc else None, osm_lon=oc[1] if oc else None,
                     d_xp_osm_m=round(hav(xc, oc), 1) if xc and oc else None,
                     in_airports_json=(st or '').upper() in cur_keys[ap] if st else False))
m = pl.DataFrame(rows, infer_schema_length=None)
m = m.join(cnt.pivot(on='src', index=['ADEP_mvt', 'STAND_mvt'], values='len').rename({'ADEP_mvt': 'apt', 'STAND_mvt': 'stand'}),
           on=['apt', 'stand'], how='left', nulls_equal=True).with_columns(pl.col('train').fill_null(0), pl.col('rank').fill_null(0))
m.sort('apt', 'stand').write_parquet(os.path.join(XP, 'stand_coords_merged.parquet'))
m.sort('apt', 'stand').write_csv(os.path.join(XP, 'stand_coords_merged.csv'))
pl.DataFrame(xp_rows).write_csv(os.path.join(XP, 'xplane_stands.csv'))
pl.DataFrame(rw_rows).write_csv(os.path.join(XP, 'xplane_runways.csv'))

cov = []
for (ap,), g in m.group_by('apt'):
    r = {'apt': ap}
    for s in ('train', 'rank'):
        n = g[s].sum()
        f = lambda e: round(float(g.filter(e)[s].sum()) / n * 100, 1)
        r[s + '_rows'] = int(n)
        r[s + '_null'] = f(pl.col('stand_key').is_null())
        r[s + '_airports_json'] = f(pl.col('in_airports_json'))
        r[s + '_osm'] = f(pl.col('osm_level').is_not_null())
        r[s + '_xp'] = f(pl.col('xp_level').is_not_null())
        r[s + '_xpL1'] = f(pl.col('xp_level') == 1)
        r[s + '_union'] = f(pl.col('match_level').is_not_null())
    both = g.filter(pl.col('d_xp_osm_m').is_not_null())
    w = both['train'].to_numpy()
    dd = both['d_xp_osm_m'].to_numpy()
    r['n_both'] = both.height
    r['d_med_m'] = round(float(np.median(dd)), 1) if len(dd) else None
    r['pct_rows_d_gt100m'] = round(float(w[dd > 100].sum() / w.sum() * 100), 1) if len(dd) else None
    r['xp_ambig'] = len(xp_idx[ap][1])
    r['osm_ambig'] = len(osm_idx[ap][1])
    cov.append(r)
cov = pl.DataFrame(cov).sort('apt')
cov.write_csv(os.path.join(XP, 'coverage.csv'))
with pl.Config(tbl_rows=20, tbl_cols=40, tbl_width_chars=400):
    print(cov)

summary = dict(coverage=cov.to_dicts(), spread_m=SPREAD)
# ---- Optional report: error of the averaged-baseline two-fold OOF (January / July 2025 held out) by stand-coverage class ----
if A.storedir:
    d = pl.read_parquet(os.path.join(A.storedir, 'feat.parquet'), columns=['src', 'ym', 'ADEP_mvt', 'STAND_mvt', 'TAXITIME_SEC_mvt'])
    out = []
    for mo in (1, 7):
        v = d.filter((pl.col('src') != 99) & (pl.col('ym') == 202500 + mo))
        b = (np.load(os.path.join(A.storedir, 'oof', 'ens_v5b_%d.npy' % mo)) + np.load(os.path.join(A.storedir, 'oof', 'ens_v6_%d.npy' % mo))) / 2
        assert len(b) == v.height
        out.append(v.with_columns(pl.Series('base', b), pl.lit(mo).alias('mo')))
    v = pl.concat(out).rename({'ADEP_mvt': 'apt', 'STAND_mvt': 'stand'})
    v = v.join(m.select('apt', 'stand', 'in_airports_json', 'match_level', 'osm_level'), on=['apt', 'stand'], how='left', nulls_equal=True)
    v = v.with_columns(pl.when(pl.col('in_airports_json')).then(pl.lit('a_airports_json')).when(pl.col('match_level').is_not_null())
                       .then(pl.lit('b_new')).otherwise(pl.lit('c_none')).alias('cat'),
                       (pl.col('TAXITIME_SEC_mvt') - pl.col('base')).alias('r')).filter(pl.col('TAXITIME_SEC_mvt') <= 10800)
    agg = v.group_by('apt', 'cat').agg(pl.len().alias('n'), (pl.col('r') ** 2).mean().sqrt().round(1).alias('rmse'),
                                        pl.col('r').mean().round(1).alias('bias')).sort('apt', 'cat')
    agg.write_csv(os.path.join(XP, 'oof_by_cov.csv'))
    tot = v.group_by('cat').agg(pl.len().alias('n'), (pl.col('r') ** 2).mean().sqrt().round(1).alias('rmse'),
                                 ((pl.col('r') ** 2).sum() / float((v['r'] ** 2).sum()) * 100).round(1).alias('sse_pct')).sort('cat')
    with pl.Config(tbl_rows=40):
        print(tot)
        print(agg)
    summary['oof_total'] = tot.to_dicts()
json.dump(summary, open(os.path.join(XP, 'summary.json'), 'w'), indent=1, default=str)
print('DONE')
