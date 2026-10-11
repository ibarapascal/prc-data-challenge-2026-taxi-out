"""patch_partial_day: downstream handling of the partially missing ADS-B archive of 2026-06-06 (only about 61 % of the data; 14 of the 24 UTC hours have points).

Background (behaviour of build_feats):
- Flights taking off in a period without data cannot be matched to a take-off event -> ab_obs_first etc. are null -> gate=False in train_resid
  -> the prediction is the baseline without ADS-B (retrained baseline + tabular term), i.e. they are already treated as missing. But ab_taxi_now10
  (number of hex codes moving on the airport ground in the 10 minutes before MVT) is computed for every row and is a "true 0" in a period without data.
- Contamination can only land on **gated rows** (rows matched to an event): the ground-segment look-back is cut at gaps of >= 600 s, the context before
  the segment start (tk_*) and the 10-minute window of taxi_now10 can all touch a missing hour, so the ADS-B features of these rows take "no data" for "no movement".
- Note on the submitted run: the fallback file was built on the 3-seed baseline, while the other rows of the submission use the 8-seed baseline
  (README, known limitations).
Rule (fixed before seeing results):
1. Hourly coverage: count points of the crops of 06-05, 06-06 and 06-07 per (airport, UTC hour). The cell (a, h) of 06-06 is missing = 0 points on 06-06 in that
   cell and (all airports have 0 points in that hour of 06-06, or the same cell has > 0 points on both 06-05 and 06-07). The former is a gap of the whole
   published layer, the latter means the airport usually has data but not on that day.
2. Affected rows: ranking DEP rows with MVT in [06-06 00:00Z, --until) (default 06-07 03:00Z; early rows of 06-07 look back into 06-06 points; the submitted
   run used margin 64800 = the 18 h look-back of the track-context gap family and until 06-08 00:00Z) that are gated rows in the full version
   (ab_obs_first of feats_tma not null) and whose window [MVT - max(obs_first_tm, obs_first_tma) - margin, MVT] (margin default 1800 s: 600 s gap cut-off
   + track context + the 600 s of taxi_now10) intersects any missing hour of their airport.
3. The prediction of an affected row is replaced by the value of the same ID in the fallback file (it has no ADS-B in June, i.e. the baseline without ADS-B;
   the same basis as the non-gated rows of the full version).
4. Evidence (report): share of non-gated rows of 06-06 where the full version equals the fallback file; number of DEP rows in missing hours and how many
   of them have taxi_now10 = 0; gated rows per hour.
Usage:
  python patch_partial_day.py --pred resid_pred/upstanding-firefly_final_full_raw.parquet --base submit/upstanding-firefly_final_baseline_20261008.parquet \
    --crop crop --feats-tm feats_tm --feats-tma feats_tma --ranking raw/ranking.parquet --airports code/airports.json \
    --out resid_pred/upstanding-firefly_final_full.parquet --report logs/partial_day_patch.json
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

IDC = 'MVT_ID_mvt'
D = '2026-06-06'


def epoch(s):
    return datetime.strptime(s, '%Y-%m-%dT%H:%M').replace(tzinfo=timezone.utc).timestamp()


def hourly(crop, date, names):
    p = pl.read_parquet(f'{crop}/{date[:4]}/{date}.parquet', columns=['apt', 't', 'alt'])
    if p['apt'].dtype != pl.String:
        p = p.with_columns(pl.col('apt').cast(pl.Int64).replace_strict(list(range(len(names))), names, return_dtype=pl.String))
    t0 = epoch(date + 'T00:00')
    p = p.with_columns(((pl.col('t') - t0) // 3600).cast(pl.Int64).alias('h')).filter(pl.col('h').is_between(0, 23))
    c = p.group_by('apt', 'h').agg(pl.len().alias('n'), (pl.col('alt') == -1000).sum().alias('ng'))
    tab = {a: [0] * 24 for a in names}
    gnd = {a: [0] * 24 for a in names}
    for a, h, n, ng in c.iter_rows():
        if a in tab:
            tab[a][h] = int(n)
            gnd[a][h] = int(ng)
    return tab, gnd


def sub(p):
    t = pq.read_table(p)
    yc = [c for c in t.column_names if c != IDC][0]
    return t, yc


def main():
    ap = argparse.ArgumentParser()
    for k in ('pred', 'base', 'crop', 'feats_tm', 'feats_tma', 'ranking', 'airports', 'out', 'report'):
        ap.add_argument('--' + k.replace('_', '-'), dest=k, required=True)
    ap.add_argument('--margin', type=float, default=1800.0)
    ap.add_argument('--until', default='2026-06-07T03:00')  # exclusive upper bound of MVT of affected rows; the submitted run (margin 64800) used 2026-06-08T00:00
    a = ap.parse_args()
    names = sorted(json.load(open(a.airports)))
    rep = dict(rule=__doc__.split('Rule (fixed before seeing results):')[1].split('Usage:')[0].strip(), margin=a.margin, until=a.until)

    # 1. Hourly coverage
    cov = {d: hourly(a.crop, d, names) for d in ('2026-06-05', D, '2026-06-07')}
    c05, c06, c07 = cov['2026-06-05'][0], cov[D][0], cov['2026-06-07'][0]
    glob06 = [sum(c06[x][h] for x in names) for h in range(24)]
    rep['global_hours_with_data_0606'] = [h for h in range(24) if glob06[h] > 0]
    rep['global_points_by_hour_0606'] = glob06
    missing = {x: [h for h in range(24) if c06[x][h] == 0 and (glob06[h] == 0 or (c05[x][h] > 0 and c07[x][h] > 0))] for x in names}
    rep['missing_hours'] = missing
    rep['hours_with_data'] = {x: {d: sum(1 for h in range(24) if cov[d][0][x][h] > 0) for d in cov} for x in names}
    rep['points_by_hour'] = {d: {x: cov[d][0][x] for x in names} for d in cov}
    rep['ground_points_by_hour_0606'] = cov[D][1]

    # 2. Affected rows
    t0, t1 = epoch(D + 'T00:00'), epoch(a.until)
    assert t1 <= epoch('2026-06-08T00:00'), 'feats are read for 06-06 / 06-07 only'
    k = (pl.read_parquet(a.ranking, columns=[IDC, 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
         .with_columns(pl.col(IDC).cast(pl.Float64), (pl.col('MVT_TIME_UTC_mvt').dt.epoch('ms') / 1000).alias('mvt'))
         .filter((pl.col('mvt') >= t0) & (pl.col('mvt') < t1)).select(IDC, 'ADEP_mvt', 'mvt'))

    def feats(dirn, tag):
        fs = [pl.read_parquet(f'{dirn}/{d}.parquet', columns=[IDC, 'ab_obs_first', 'ab_taxi_now10']) for d in (D, '2026-06-07')]
        return pl.concat(fs).with_columns(pl.col(IDC).cast(pl.Float64)).unique(IDC).rename(
            {'ab_obs_first': f'of_{tag}', 'ab_taxi_now10': f'tn_{tag}'})
    k = k.join(feats(a.feats_tm, 'tm'), on=IDC, how='left').join(feats(a.feats_tma, 'tma'), on=IDC, how='left')
    k = k.with_columns(pl.col('of_tma').is_not_null().alias('gate'),
                       pl.max_horizontal(pl.col('of_tm').fill_null(0), pl.col('of_tma').fill_null(0), pl.lit(0.0)).alias('seg'),
                       ((pl.col('mvt') - t0) // 3600).cast(pl.Int64).alias('h'))
    ws = (k['mvt'] - k['seg'] - a.margin).to_numpy()
    we = k['mvt'].to_numpy()
    apt = k['ADEP_mvt'].to_numpy()
    hit = np.zeros(k.height, bool)
    for i in range(k.height):
        for h in missing.get(apt[i], []):
            s, e = t0 + h * 3600, t0 + (h + 1) * 3600
            if ws[i] < e and we[i] > s:
                hit[i] = True
                break
    k = k.with_columns(pl.Series('hit', hit))
    aff = k.filter(pl.col('gate') & pl.col('hit'))
    d06 = k.filter(pl.col('h') < 24)
    in_miss = np.array([h in missing.get(x, []) for x, h in zip(d06['ADEP_mvt'].to_list(), d06['h'].to_list())])
    rep['rows'] = dict(window_rows=k.height, dep_0606=d06.height, gated_0606=int(d06['gate'].sum()),
                       dep_0606_in_missing_hour=int(in_miss.sum()),
                       gated_0606_in_missing_hour=int((in_miss & d06['gate'].to_numpy()).sum()),
                       taxi_now10_zero_in_missing_hour=int(((d06['tn_tma'].fill_null(-1).to_numpy() == 0) & in_miss).sum()),
                       gated_window=int(k['gate'].sum()), affected=aff.height,
                       affected_0606=int((aff['mvt'] < t0 + 86400).sum()), affected_0607=int((aff['mvt'] >= t0 + 86400).sum()),
                       affected_by_apt={x: int(n) for x, n in aff.group_by('ADEP_mvt').len().sort('ADEP_mvt').iter_rows()})
    # Coverage self-check: end of the last missing hour + margin + longest segment in the window < until shows that the upper bound does not cut off possibly affected rows
    last_end = max((t0 + (h + 1) * 3600 for hs in missing.values() for h in hs), default=t0)
    seg_max = float(k.filter(pl.col('gate'))['seg'].max() or 0.0)
    a07 = aff.filter(pl.col('mvt') >= t0 + 86400)
    utc = lambda s: datetime.fromtimestamp(s, timezone.utc).strftime('%Y-%m-%dT%H:%M')
    rep['coverage'] = dict(last_missing_end_utc=utc(last_end), seg_max_s=seg_max, latest_possible_hit_utc=utc(last_end + a.margin + seg_max),
                           until_covers=bool(last_end + a.margin + seg_max < t1),
                           affected_mvt_max_utc=utc(float(aff['mvt'].max())) if aff.height else None,
                           affected_0607_by_hour=[int((((a07['mvt'] - t0 - 86400) // 3600) == h).sum()) for h in range(24)],
                           affected_0607_by_apt={x: int(n) for x, n in a07.group_by('ADEP_mvt').len().sort('ADEP_mvt').iter_rows()},
                           gated_0607_in_window=int(k.filter(pl.col('gate') & (pl.col('mvt') >= t0 + 86400)).height))
    rep['gated_by_hour_0606'] = {x: [int(((d06['ADEP_mvt'] == x) & (d06['h'] == h) & d06['gate']).sum()) for h in range(24)] for x in names}

    # 3. Replacement + evidence
    pt, pyc = sub(a.pred)
    bt, byc = sub(a.base)
    pid = pt.column(IDC).to_numpy().astype(np.float64)
    pv = pt.column(pyc).to_numpy(zero_copy_only=False).astype(np.float64)
    bmap = pl.DataFrame({IDC: bt.column(IDC).to_numpy().astype(np.float64), 'b': bt.column(byc).to_numpy(zero_copy_only=False).astype(np.float64)})
    f = pl.DataFrame({IDC: pid, 'p': pv, 'i': np.arange(len(pid))}).join(bmap, on=IDC, how='left')
    assert f['b'].null_count() == 0
    w = k.join(f, on=IDC, how='left')
    ng = w.filter(~pl.col('gate'))
    rep['evidence'] = dict(nongated_eq_base=int((ng['p'] == ng['b']).sum()), nongated=ng.height,
                           gated_eq_base=int(w.filter(pl.col('gate')).select((pl.col('p') == pl.col('b')).sum()).item()), gated=int(w['gate'].sum()))
    af = f.filter(pl.col(IDC).is_in(aff[IDC].to_list()))
    d = (af['b'] - af['p']).to_numpy()
    rep['patch'] = dict(rows=af.height, changed=int((d != 0).sum()), mean_base_minus_full=round(float(d.mean()), 3) if len(d) else None,
                        mean_abs=round(float(np.abs(d).mean()), 3) if len(d) else None, max_abs=round(float(np.abs(d).max()), 1) if len(d) else None)
    afd = af.join(aff.select(IDC, 'mvt'), on=IDC, how='left').with_columns((pl.col('b') - pl.col('p')).alias('d'),
                                                                            (pl.col('mvt') >= t0 + 86400).alias('d07'))
    rep['patch_by_day'] = {('2026-06-07' if dd else D): dict(rows=g.height, changed=int((g['d'] != 0).sum()), mean=round(float(g['d'].mean()), 3),
                                                             mean_abs=round(float(g['d'].abs().mean()), 3), max_abs=round(float(g['d'].abs().max()), 1))
                           for (dd,), g in afd.group_by('d07')}
    nv = pv.copy()
    idx = af['i'].to_numpy()
    nv[idx] = af['b'].to_numpy()
    cols = [pt.column(c) if c != pyc else pa.array(nv).cast(pt.schema.field(pyc).type) for c in pt.column_names]
    out = pa.Table.from_arrays(cols, schema=pt.schema)
    tmp = a.out + '.tmp'
    pq.write_table(out, tmp)
    os.replace(tmp, a.out)
    rep['out'] = a.out
    json.dump(rep, open(a.report + '.tmp', 'w'), ensure_ascii=False, indent=1)
    os.replace(a.report + '.tmp', a.report)
    print(json.dumps(dict(rows=rep['rows'], evidence=rep['evidence'], patch=rep['patch'], patch_by_day=rep['patch_by_day'], coverage=rep['coverage'], global_hours=rep['global_hours_with_data_0606']), ensure_ascii=False))


if __name__ == '__main__':
    main()
