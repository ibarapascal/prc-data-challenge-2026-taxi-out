"""Build ADS-B features for every DEP row from the cropped tracks (per day); labels are not used.

Usage: python build_feats.py --crop <crop root> --raw <raw dir> --out <feats dir> [--dates 2025-01-05 ...|--all]
Input: crop/<YYYY>/<date>.parquet (that day plus the previous day, for ground segments crossing midnight); raw training-month parquet or ranking.parquet.
Output: feats/<date>.parquet, key MVT_ID_mvt, columns as in FEATS. Unmatched rows are written too (features null) to allow coverage statistics.

Method:
1. Take-off event: same hex, same airport, an airborne point within 120 s after a ground point; or the last point of a
   ground segment has gs >= 50 kt (points lost during the take-off roll).
1b. Fallback: low-altitude climb segment (<= field elevation + 2500 ft, climb >= 300 ft); the lift-off time is
   extrapolated from the first-point altitude and climb rate. Used only for rows not matched by a ground event;
   its ground segment is traced back from the last ground point of that hex before lift-off (within 1 h),
   ab_last_ground = MVT - time of the last ground point.
2. Matching: first by call sign (ADS-B cs == CALLSIGN_flt), nearest within +-180 s; the rest by time, nearest unused
   event within +-90 s; rejected if both call signs are present and differ. One event is matched to one row only.
3. Ground segment: trace back from the take-off point along the ground points of the same hex, consecutive gap < 600 s,
   at most 3 h, stop at a landing (air -> ground).
4. Features: obs_first / obs_move / obs_push (MVT minus the first ground point / the first point with gs >= 3 / the
   first sustained movement after >= 60 s standing still at the segment start = push-back),
   dwell_seen (whether the aircraft is seen standing still at the stand at the segment start; an earlier "last long
   stop" definition picked up de-icing pads and runway-holding queues and was dropped), straight-line distance from the
   first point to the take-off point, observed path length, in-segment stop time, first-point gs, distance from the
   first point to the registered stand (when coordinates exist), maximum sampling gap, number of points;
   plus airport-level congestion: number of hexes moving on the ground at the same airport in the 10 minutes before MVT (ADS-B visible only).
5. Stand-table switch (default off: output unchanged): --stands <stand_coords_merged.parquet> (OSM union X-Plane apt.dat,
   produced by the stand-table merge script). When given, the stand coordinates of ab_dist_stand are looked up in the
   merged table by (airport, raw STAND), falling back to stand_key under the same normalisation canon() (the old rule
   did an upper-case exact match only and covered about half of the test stands); also adds ab_dist_stand_rwy = straight-line
   distance from the stand to the end of the active take-off runway (--runways defaults to xplane_runways.csv in the same
   directory: runway-end coordinates from apt.dat row code 100, without displaced thresholds; independent of ADS-B
   matching, computed for all DEP rows).
   Runway alignment: RUNWAY_mvt is upper-cased and stripped of spaces; if it is a number with an optional L/C/R the number is
   zero-padded to two digits (7C -> 07C) and must equal the apt.dat end designator exactly; NA/H/HEL etc. -> null.
6. Aircraft-type constrained matching (default off: output unchanged): with --typefilter the second (nearest-time) round
   additionally requires the ADS-B type family (mode of the actype of that hex on the cropped day, normalised by type_chk.fam)
   to equal the family of AIRCRAFT_TYPE_mvt (no restriction if either is missing); candidates that disagree are skipped and
   the next one is tried; the first (call-sign) round is unchanged. On 2025 data about 14% of the pure time-matched ground
   matches have a mismatching type; for these rows the median of |y - obs_push| is 350 s.
7. Segment-start anchoring (default off: output unchanged): with --anchor (together with --stands), if the traced-back
   ground segment contains a sub-segment "within 150 m of the registered stand, standing still (gs < 1) for >= 300 s with
   points after it", and movement with gs >= 3 occurs before that sub-segment (arrival taxi / towing mixed in), the segment
   start is moved to the start of the last such sub-segment; adds the column ab_anchor (1 = trimmed).
   obs_first / move / push etc. are recomputed from the new segment start.
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import polars as pl

from type_chk import fam

GROUND = -1000
TYPEF = False
ANCHOR = False
FEATS = ['ab_hex', 'ab_dlift', 'ab_cs_match', 'ab_obs_first', 'ab_obs_move', 'ab_dwell_seen',
         'ab_dist_first_lift', 'ab_path_len', 'ab_stop_after_move', 'ab_gs_first', 'ab_dist_stand',
         'ab_max_gap', 'ab_npts', 'ab_first_lat', 'ab_first_lon', 'ab_taxi_now10', 'ab_hold_last',
         'ab_obs_push', 'ab_etype', 'ab_last_ground']


def hav(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def norm(s):
    return s.strip().upper() if isinstance(s, str) and s.strip() else None


def canon(s):
    """Stand-name normalisation, the same rule as canon() of the stand-table merge script."""
    if s is None:
        return None
    s = re.sub(r'\(.*?\)', '', str(s).upper().strip())
    s = re.sub(r'^(GATE|RAMP|STAND|PARKING|POSITION|POS|APRON)\b\s*', '', s)
    s = re.sub(r'[\s\-_]', '', s)
    m = re.fullmatch(r'([A-Z]*)(\d+)([A-Z]*)', s)
    if m:
        s = m.group(1) + str(int(m.group(2))) + m.group(3)
    return s or None


def rwy_key(s):
    m = re.fullmatch(r'(\d{1,2})([LCR]?)', str(s).upper().replace(' ', '')) if s is not None else None
    return f'{int(m.group(1)):02d}{m.group(2)}' if m else None


def load_stand_tables(stands, runways):
    """-> (st[apt][raw STAND] -> (lat,lon), sk[apt][canon key] -> (lat,lon), rw[apt][runway end designator] -> (lat,lon))."""
    m = pl.read_parquet(stands).filter(pl.col('lat').is_not_null())
    st, sk = {}, {}
    for apt, stand, key, la, lo in m.select('apt', 'stand', 'stand_key', 'lat', 'lon').iter_rows():
        st.setdefault(apt, {})[stand] = (la, lo)
        if key:
            sk.setdefault(apt, {}).setdefault(key, (la, lo))
    rw = {}
    for apt, r, la, lo in pl.read_csv(runways, schema_overrides={'rwy': pl.String}).select('apt', 'rwy', 'lat', 'lon').iter_rows():
        if rwy_key(r):
            rw.setdefault(apt, {})[rwy_key(r)] = (la, lo)
    return st, sk, rw


def load_points(crop, date):
    d = np.datetime64(date)
    fs = [Path(crop) / str(x)[:4] / f'{x}.parquet' for x in (d - 1, d)]
    fs = [f for f in fs if f.exists()]
    if not (Path(crop) / date[:4] / f'{date}.parquet').exists():
        return None
    cols = ['hex', 'apt', 't', 'lat', 'lon', 'alt', 'gs', 'cs'] + (['actype'] if TYPEF else [])
    return pl.concat([pl.read_parquet(f, columns=cols)
                      for f in fs]).unique(['hex', 't'], keep='first').sort(['hex', 't'])


def lift_events(p):
    q = p.with_columns([pl.col(c).shift(-1).alias(c + '_n') for c in ('hex', 't', 'alt', 'apt', 'cs', 'gs')])
    same = (pl.col('hex') == pl.col('hex_n')) & (pl.col('apt') == pl.col('apt_n'))
    a = q.filter(same & (pl.col('alt') == GROUND) & (pl.col('alt_n') > GROUND) &
                 (pl.col('t_n') - pl.col('t') < 120))
    a = a.select('hex', 'apt', t_lift=(pl.col('t') + pl.col('t_n')) / 2, t_lastg='t',
                 cs=pl.coalesce('cs_n', 'cs'))
    # Last ground point is in a fast take-off roll and no ground point at the same airport follows within 30 minutes (airborne points were lost)
    b = q.filter((pl.col('alt') == GROUND) & (pl.col('gs') >= 50) &
                 (~same | (pl.col('t_n') - pl.col('t') >= 120)))
    b = b.select('hex', 'apt', t_lift=pl.col('t') + 10, t_lastg='t', cs='cs')
    ev = pl.concat([a, b]).sort('hex', 't_lastg')
    # Of events repeated within 5 minutes for the same hex and airport, keep only the first
    ev = ev.filter((pl.col('t_lift') - pl.col('t_lift').shift(1).over('hex', 'apt')).fill_null(1e9) > 300)
    return ev.with_columns(pl.col('cs').map_elements(norm, return_dtype=pl.String))


ELEV = {'EDDF': 364, 'EDDM': 1487, 'EGLL': 83, 'EHAM': -11, 'LEBL': 12, 'LEMD': 1998,
        'LFPG': 392, 'LIRF': 13, 'LSZH': 1416, 'LTFM': 325}


def climb_events(p, names):
    """Low-altitude climb segment: consecutive airborne points of the same hex and airport (gap < 120 s) with altitude <= elevation + 2500 ft and a climb of >= 300 ft within the segment;
    the lift-off time is extrapolated from the first-point altitude and the mean climb rate. t_lastg = last ground point of the same hex before that time (NaN if none)."""
    elev = pl.DataFrame({'apt': list(range(len(names))), 'elev': [ELEV[n] for n in names]}).with_columns(pl.col('apt').cast(pl.Int8))
    a = (p.join(elev, on='apt').filter((pl.col('alt') > GROUND) & (pl.col('alt') <= pl.col('elev') + 2500))
         .with_columns(((pl.col('t') - pl.col('t').shift(1).over('hex', 'apt')).fill_null(1e9) > 120).cum_sum().over('hex', 'apt').alias('run')))
    r = (a.group_by('hex', 'apt', 'run').agg(pl.col('t').first().alias('t0'), pl.col('t').last().alias('t1'),
                                             pl.col('alt').first().alias('a0'), pl.col('alt').last().alias('a1'),
                                             pl.col('elev').first(), pl.col('cs').drop_nulls().last().alias('cs'))
         .filter((pl.col('a1') - pl.col('a0') >= 300) & (pl.col('t1') > pl.col('t0'))))
    rate = ((pl.col('a1') - pl.col('a0')) / (pl.col('t1') - pl.col('t0'))).clip(10, 80)
    r = r.with_columns((pl.col('t0') - ((pl.col('a0') - pl.col('elev')).clip(0, None) / rate)).alias('t_lift'))
    g = p.filter(pl.col('alt') == GROUND).select('hex', 'apt', pl.col('t').alias('t_lastg')).sort('t_lastg')
    r = r.sort('t_lift').join_asof(g, left_on='t_lift', right_on='t_lastg', by=['hex', 'apt'], strategy='backward',
                                   tolerance=3600)
    return r.select('hex', 'apt', 't_lift', 't_lastg', 'cs').with_columns(pl.col('cs').map_elements(norm, return_dtype=pl.String))


def match(dep, ev):
    """dep: MVT_ID, apt, mvt, cs -> dict MVT_ID -> (event row index, dlift, cs_match)."""
    evs = ev.with_row_index('eid')
    used, res = set(), {}
    by_apt = {k[0]: v for k, v in evs.partition_by('apt', as_dict=True).items()}
    # Round 1: call sign
    for apt, e in by_apt.items():
        et, ecs, eid = e['t_lift'].to_numpy(), e['cs'].to_list(), e['eid'].to_numpy()
        idx = {}
        for i, c in enumerate(ecs):
            if c:
                idx.setdefault(c, []).append(i)
        for r in dep.filter((pl.col('apt') == apt) & pl.col('cs').is_not_null()).iter_rows(named=True):
            cand = [i for i in idx.get(r['cs'], []) if abs(et[i] - r['mvt']) <= 180 and eid[i] not in used]
            if cand:
                i = min(cand, key=lambda i: abs(et[i] - r['mvt']))
                used.add(eid[i])
                res[r['MVT_ID_mvt']] = (int(eid[i]), float(et[i] - r['mvt']), 1)
    # Round 2: nearest in time
    pairs = []
    for apt, e in by_apt.items():
        et, ecs, eid = e['t_lift'].to_numpy(), e['cs'].to_list(), e['eid'].to_numpy()
        efam = e['fam'].to_list() if TYPEF else [None] * e.height
        order = np.argsort(et)
        ets = et[order]
        for r in dep.filter(pl.col('apt') == apt).iter_rows(named=True):
            if r['MVT_ID_mvt'] in res:
                continue
            lo, hi = np.searchsorted(ets, r['mvt'] - 90), np.searchsorted(ets, r['mvt'] + 90)
            for j in order[lo:hi]:
                if eid[j] in used or (ecs[j] and r['cs'] and ecs[j] != r['cs']):
                    continue
                if TYPEF and efam[j] and r['fam_ds'] and efam[j] != r['fam_ds']:
                    continue
                pairs.append((abs(et[j] - r['mvt']), r['MVT_ID_mvt'], int(eid[j]), float(et[j] - r['mvt'])))
    for d, mid, e, dl in sorted(pairs):
        if mid in res or e in used:
            continue
        used.add(e)
        res[mid] = (e, dl, 0)
    return res


def seg_feats(t, lat, lon, gs, alt, i_end, stand_ll, lift_ll):
    """t etc. are all points of the same hex and airport (sorted); i_end = index of the last ground point before take-off."""
    j = i_end
    while j > 0:
        if t[j] - t[j - 1] >= 600 or t[i_end] - t[j - 1] > 10800:
            break
        if alt[j - 1] != GROUND:  # previous point is airborne: landing, cut here
            break
        j -= 1
    anchored = 0
    if ANCHOR and stand_ll and i_end - j >= 2:  # --anchor: move the segment start to the last sub-segment within 150 m of the registered stand and standing still for >= 300 s (cut only if movement precedes it)
        gg = np.nan_to_num(gs[j:i_end + 1], nan=0.0)
        tt = t[j:i_end + 1]
        near = (gg < 1.0) & (hav(lat[j:i_end + 1], lon[j:i_end + 1], stand_ll[0], stand_ll[1]) <= 150)
        best, k, m = None, 0, len(tt)
        while k < m:
            if near[k]:
                s0 = k
                while k + 1 < m and near[k + 1]:
                    k += 1
                if tt[k] - tt[s0] >= 300 and k < m - 1:
                    best = s0
            k += 1
        if best and (gg[:best] >= 3).any():
            j, anchored = j + best, 1
    st, sla, slo, sg = t[j:i_end + 1], lat[j:i_end + 1], lon[j:i_end + 1], np.nan_to_num(gs[j:i_end + 1], nan=0.0)
    n = len(st)
    # Push-back time: after >= 60 s standing still (gs < 1) at the segment start, the first time two consecutive points have gs >= 1 (only defined if the aircraft is seen at the stand)
    still = sg < 1.0
    t_push, dwell = np.nan, 0
    if n > 2 and still[0]:
        k = 0
        while k < n and still[k]:
            k += 1
        if k < n - 1 and st[k - 1] - st[0] >= 60 and sg[k + 1] >= 1:
            t_push, dwell = st[k], 1
    mv = np.where(sg >= 3)[0]
    t_move = st[mv[0]] if len(mv) else st[0]
    after = st >= t_move
    dts = np.diff(st)
    stop_after = float(np.sum(dts[(sg[:-1] < 1) & after[:-1]])) if n > 1 else 0.0
    seg = hav(sla[:-1], slo[:-1], sla[1:], slo[1:]) if n > 1 else np.array([0.0])
    path = float(np.sum(seg[after[:-1]])) if n > 1 else 0.0
    return dict(t_first=st[0], t_move=t_move, t_push=t_push, dwell=dwell, path=path, stop_after=stop_after,
                gs_first=float(sg[0]), max_gap=float(dts.max()) if n > 1 else 0.0, npts=n,
                first_lat=sla[0], first_lon=slo[0],
                dist_first_lift=float(hav(sla[0], slo[0], lift_ll[0], lift_ll[1])),
                dist_stand=float(hav(sla[0], slo[0], stand_ll[0], stand_ll[1])) if stand_ll else None,
                hold_last=float(np.sum(dts[(sg[:-1] < 1) & (st[:-1] >= st[-1] - 600)])) if n > 1 else 0.0, anchor=anchored)


def taxi_now(p, apt_i, times):
    """Number of hexes moving on the ground at this airport with gs >= 3 in the 10 minutes before each time in `times`."""
    g = p.filter((pl.col('apt') == apt_i) & (pl.col('alt') == GROUND) & (pl.col('gs') >= 3)).select('hex', 't')
    if g.height == 0:
        return np.zeros(len(times))
    # De-duplicate per hex and minute, then count per minute bucket
    mins = (g['t'].to_numpy() // 60).astype(np.int64)
    hexes = g['hex'].to_numpy()
    s = pl.DataFrame({'m': mins, 'h': hexes}).unique().group_by('m').len()
    cnt = dict(zip(s['m'].to_list(), s['len'].to_list()))
    out = []
    for tt in times:
        m0 = int(tt // 60)
        out.append(max(cnt.get(m, 0) for m in range(m0 - 10, m0 + 1)))
    return np.array(out, dtype=float)


def add_fam(ev, p):
    """--typefilter: attach to each event the type family of the mode actype of its hex on that day (including the previous day); no change when off."""
    if not TYPEF:
        return ev
    m = (p.drop_nulls('actype').with_columns(pl.col('actype').str.strip_chars().str.to_uppercase())
         .group_by('hex', 'actype').len().sort('len', descending=True).unique('hex', keep='first')
         .select('hex', pl.col('actype').map_elements(fam, return_dtype=pl.String).alias('fam')))
    return ev.join(m, on='hex', how='left')


def build_day(date, crop, deps, apts, names, stab=None):
    p = load_points(crop, date)
    if p is None:
        return None
    dep = deps.filter(pl.col('date') == date)
    ev1 = add_fam(lift_events(p), p).with_columns(pl.lit(1, pl.Int8).alias('etype'))
    res = match(dep, ev1)
    ev2 = add_fam(climb_events(p, names), p).with_columns(pl.lit(2, pl.Int8).alias('etype'))
    # Drop climb-segment events whose hex is already covered by a ground take-off event (same airport, +-300 s)
    ev2 = ev2.join(ev1.select('hex', 'apt', pl.col('t_lift').alias('tl1')), on=['hex', 'apt'], how='left') \
             .filter(pl.col('tl1').is_null() | ((pl.col('t_lift') - pl.col('tl1')).abs() > 300)) \
             .drop('tl1').unique(['hex', 'apt', 't_lift'])
    res2 = match(dep.filter(~pl.col('MVT_ID_mvt').is_in(list(res))), ev2)
    ev = pl.concat([ev1, ev2.select(ev1.columns)])
    off = ev1.height
    res.update({k: (v[0] + off, v[1], v[2] + 2) for k, v in res2.items()})
    evs = ev.with_row_index('eid')
    groups = p.partition_by(['hex', 'apt'], as_dict=True)
    arrs = {}
    rows = []
    stands = {n: {k.upper(): v for k, v in apts[n]['stands'].items()} for n in names}
    for r in dep.iter_rows(named=True):
        out = {'MVT_ID_mvt': r['MVT_ID_mvt']}
        if stab:
            an = names[r['apt']]
            sll = stab[0].get(an, {}).get(r['STAND_mvt']) or stab[1].get(an, {}).get(canon(r['STAND_mvt']))
            rll = stab[2].get(an, {}).get(rwy_key(r['RUNWAY_mvt']))
            out['ab_dist_stand_rwy'] = float(hav(sll[0], sll[1], rll[0], rll[1])) if sll and rll else None
        m = res.get(r['MVT_ID_mvt'])
        if m:
            e = evs.row(m[0], named=True)
            key = (e['hex'], e['apt'])
            if key not in arrs:
                g = groups[key]
                arrs[key] = tuple(g[c].to_numpy() for c in ('t', 'lat', 'lon', 'gs', 'alt'))
            t, lat, lon, gs, alt = arrs[key]
            out.update(ab_hex=e['hex'], ab_dlift=m[1], ab_cs_match=m[2], ab_etype=e['etype'])
            if e['t_lastg'] is None or np.isnan(e['t_lastg']):
                rows.append(out)
                continue
            i_end = int(np.searchsorted(t, e['t_lastg'], side='right') - 1)
            li = min(i_end + 1, len(t) - 1)
            if not stab:
                sll = stands[names[r['apt']]].get((r['STAND_mvt'] or '').upper()) if r['STAND_mvt'] else None
            f = seg_feats(t, lat, lon, gs, alt, i_end, sll, (lat[li], lon[li]))
            out.update(ab_last_ground=r['mvt'] - t[i_end],
                       ab_obs_first=r['mvt'] - f['t_first'], ab_obs_move=r['mvt'] - f['t_move'],
                       ab_obs_push=r['mvt'] - f['t_push'],
                       ab_dwell_seen=f['dwell'], ab_dist_first_lift=f['dist_first_lift'],
                       ab_path_len=f['path'], ab_stop_after_move=f['stop_after'], ab_gs_first=f['gs_first'],
                       ab_dist_stand=f['dist_stand'], ab_max_gap=f['max_gap'], ab_npts=f['npts'],
                       ab_first_lat=f['first_lat'], ab_first_lon=f['first_lon'], ab_hold_last=f['hold_last'])
            if ANCHOR:
                out['ab_anchor'] = f['anchor']
        rows.append(out)
    df = pl.DataFrame(rows, infer_schema_length=None)
    feats = FEATS + (['ab_dist_stand_rwy'] if stab else []) + (['ab_anchor'] if ANCHOR else [])
    for c in feats:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    tn = np.full(dep.height, np.nan)
    for ai in range(len(names)):
        msk = (dep['apt'] == ai).to_numpy()
        if msk.any():
            tn[msk] = taxi_now(p, ai, dep['mvt'].to_numpy()[msk])
    df = df.with_columns(pl.Series('ab_taxi_now10', tn))
    return df.select(['MVT_ID_mvt'] + feats).with_columns(pl.lit(date).alias('date'))


def load_deps(raw, dates, names, rwy=False):
    """With TYPEF, also read AIRCRAFT_TYPE_mvt -> fam_ds."""
    frames = []
    months = sorted({d[:7] for d in dates})
    for mo in months:
        if mo.startswith('2026'):
            f = Path(raw) / 'ranking.parquet'
        else:
            f = next(Path(raw).glob(f'training_{mo}-01_*.parquet'))
        cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'CALLSIGN_flt', 'STAND_mvt', 'MVT_TIME_UTC_mvt'] + (['RUNWAY_mvt'] if rwy else []) \
            + (['AIRCRAFT_TYPE_mvt'] if TYPEF else [])
        d = (pl.read_parquet(f, columns=cols).filter(pl.col('PHASE_mvt') == 'DEP')
             .with_columns(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m-%d').alias('date'),
                           (pl.col('MVT_TIME_UTC_mvt').dt.epoch('ms') / 1000).alias('mvt'),
                           pl.col('CALLSIGN_flt').map_elements(norm, return_dtype=pl.String).alias('cs'),
                           pl.col('ADEP_mvt').replace_strict(names, list(range(len(names))), default=None,
                                                             return_dtype=pl.Int8).alias('apt'))
             .filter(pl.col('date').is_in(dates)))
        frames.append(d)
    d = pl.concat(frames).unique('MVT_ID_mvt', keep='first')
    if TYPEF:
        d = d.with_columns(pl.col('AIRCRAFT_TYPE_mvt').cast(pl.String).str.strip_chars().str.to_uppercase()
                           .map_elements(fam, return_dtype=pl.String).alias('fam_ds'))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--airports', default=str(Path(__file__).with_name('airports.json')))
    ap.add_argument('--dates', nargs='*')
    ap.add_argument('--redo', action='store_true')
    ap.add_argument('--stands', help='stand_coords_merged.parquet; enables the merged stand table and ab_dist_stand_rwy (default off: output unchanged)')
    ap.add_argument('--runways', help='xplane_runways.csv (default: same directory as --stands)')
    ap.add_argument('--typefilter', action='store_true', help='require matching aircraft-type families in the second (time) matching round (default off: output unchanged)')
    ap.add_argument('--anchor', action='store_true', help='anchor the ground-segment start to the stationary period at the stand (needs --stands; default off: output unchanged)')
    a = ap.parse_args()
    global TYPEF, ANCHOR
    TYPEF = a.typefilter
    ANCHOR = a.anchor
    stab = load_stand_tables(a.stands, a.runways or str(Path(a.stands).with_name('xplane_runways.csv'))) if a.stands else None
    apts = json.load(open(a.airports))
    names = sorted(apts)
    dates = a.dates or sorted(f.stem for f in Path(a.crop).glob('20[0-9][0-9]/20*.parquet'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dates = [d for d in dates if a.redo or not (out / f'{d}.parquet').exists()]
    if not dates:
        return
    deps = load_deps(a.raw, dates, names, rwy=stab is not None)
    for d in dates:
        df = build_day(d, a.crop, deps, apts, names, stab)
        if df is None:
            continue
        df.write_parquet(out / f'{d}.parquet')
        cov = df.filter(pl.col('ab_hex').is_not_null()).height
        print(d, 'dep', df.height, 'matched', cov, 'obs_move_med',
              df['ab_obs_move'].median(), flush=True)


if __name__ == '__main__':
    main()
