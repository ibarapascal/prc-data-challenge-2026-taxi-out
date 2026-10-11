"""drift_gate: input-distribution drift gate between 2025 and 2026 (no training, no labels; the pre-written drift-gate rule).

For each airport x month, compare the distributions of the 17 track-context (TK) columns and the 19 ADS-B (AB) columns on
gated rows (ab_obs_first not null) between the same month of 2025 and 2026 (missing-rate difference, PSI, KS, quantile shift).
EDDF-01 is additionally split at 2026-01-29 into an early and a late part. Column-level triggers and the cell verdict use the
thresholds fixed before use. Also computed: mean / RMS differences between the candidate prediction files (keys variant_simple = simpler
residual variant, variant_merged / variant_merged_sc = merged full variants, variant_norevert = merged prediction before reverts) on 2026 gated rows; the amount
by which the merged file changes when cells judged "revert" are set back to the simpler variant (variant_simple); and the CV-based cost
range of that revert. Side evidence (not part of the verdict): TK distributions of the winter pool (Nov/Dec 2025) against
2025-01 and 2026-01 for seven airports. Read-only, runs in seconds on pyarrow + numpy + scipy.
Usage: python3 drift_gate.py --dir <dir with tk_main.parquet, tk_pool.parquet, sc_table_all_full.parquet, feats_tma/>
        --sub <dir with the variant_simple / variant_merged / variant_merged_sc / variant_norevert parquet files, variant_full.res.npy, ranking.parquet> --out <report.json>
Final-phase options (defaults = original behaviour):
  --months 01,07 (2026 test months); --ref-map 02:01,06:06 (test month -> 2025 reference month, default = same month);
  --cv-map 02:01,06:07 (which CV month's CV_SHIFT to use, default = same month);
  --names variant_simple=resid_simple,variant_merged=resid_merged,variant_merged_sc=resid_merged_sc,variant_norevert=<fallback baseline>,res=resid_full (candidate file names, default = old names).
  The reference month's 2025 rows must be in tk_main and feats_tma (2025-06 is not, so every cell of that month gets the
  verdict 'insufficient_sample', not judged; unvalidated).
"""
import argparse
import glob
import json
import os

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import ks_2samp

TK_ALL = ['tk_gs1', 'tk_gs5', 'tk_gs8', 'tk_disp30', 'tk_rev', 'tk_turn', 'tk_trk_agree', 'tk_first_x', 'tk_push_x', 'tk_sleave',
          'tk_sdist_first_x', 'tk_stop_gap_s', 'tk_stop_gap_m', 'tk_arr', 'tk_ep_n', 'tk_ep_first', 'tk_stop_code']
TK_STOPS = ['gap', 'max3h', 'air', 'miss', 'start', 'win']
AB = ['ab_dlift', 'ab_cs_match', 'ab_obs_first', 'ab_obs_move', 'ab_dwell_seen', 'ab_dist_first_lift', 'ab_path_len',
      'ab_stop_after_move', 'ab_gs_first', 'ab_dist_stand', 'ab_max_gap', 'ab_npts', 'ab_first_lat', 'ab_first_lon',
      'ab_taxi_now10', 'ab_hold_last', 'ab_obs_push', 'ab_etype', 'ab_last_ground']
CV_SHIFT = {  # per-cell mean shift per gated row caused by the TK features in the 2025 CV (full model)
    'EDDF-01': -0.45, 'EDDF-07': 0.28, 'EDDM-01': -0.09, 'EDDM-07': -0.02, 'EGLL-01': -0.33, 'EGLL-07': -4.83, 'EHAM-01': 0.2,
    'EHAM-07': -0.94, 'LEBL-01': 0.18, 'LEBL-07': 0.07, 'LEMD-07': 0.78, 'LFPG-07': -9.91, 'LIRF-07': -0.93, 'LSZH-01': 1.23,
    'LSZH-07': 0.15}
PSI_T, MISS_T, REL, MIN_N, K_T, S_T, S_REL = 0.25, 0.10, 3.0, 200, 3, 3.0, 5.0


def col(t, c):
    if c == 'tk_stop_code':
        s = t.column('tk_stop').to_numpy(zero_copy_only=False)
        m = {k: i for i, k in enumerate(TK_STOPS)}
        return np.array([m.get(x, np.nan) if x is not None else np.nan for x in s], float)
    return t.column(c).to_numpy(zero_copy_only=False).astype(float)


def psi(ref, obj):
    """Deciles (or the categories, for columns with few values) plus a missing bin; bin shares floored at 1e-4."""
    rm, om = np.isnan(ref), np.isnan(obj)
    r, o = ref[~rm], obj[~om]
    u = np.unique(r)
    if len(u) <= 12:
        vals = np.union1d(u, np.unique(o))
        rb = np.array([(r == v).sum() for v in vals] + [rm.sum()], float)
        ob = np.array([(o == v).sum() for v in vals] + [om.sum()], float)
    else:
        e = np.unique(np.quantile(r, np.linspace(0, 1, 11)[1:-1]))
        rb = np.append(np.bincount(np.searchsorted(e, r, side='right'), minlength=len(e) + 1), rm.sum()).astype(float)
        ob = np.append(np.bincount(np.searchsorted(e, o, side='right'), minlength=len(e) + 1), om.sum()).astype(float)
    p, q = np.maximum(rb / rb.sum(), 1e-4), np.maximum(ob / ob.sum(), 1e-4)
    return float(((q - p) * np.log(q / p)).sum())


def stats(ref, obj):
    rm, om = np.isnan(ref), np.isnan(obj)
    d = dict(n_ref=int(len(ref)), n_obj=int(len(obj)), miss_ref=round(float(rm.mean()), 4), miss_obj=round(float(om.mean()), 4),
             dmiss=round(float(om.mean() - rm.mean()), 4), psi=round(psi(ref, obj), 4))
    r, o = ref[~rm], obj[~om]
    if len(r) > 20 and len(o) > 20:
        iqr = np.subtract(*np.quantile(r, [.75, .25])) or 1.0
        qr, qo = np.quantile(r, [.1, .5, .9]), np.quantile(o, [.1, .5, .9])
        d.update(ks=round(float(ks_2samp(r, o).statistic), 4), q_ref=[round(float(x), 2) for x in qr],
                 q_obj=[round(float(x), 2) for x in qo], dq_iqr=[round(float(x), 3) for x in (qo - qr) / iqr])
    return d


def load_feats(d):
    ids, date, cols = [], [], {c: [] for c in AB}
    for f in sorted(glob.glob(os.path.join(d, 'feats_tma', '*.parquet'))):
        t = pq.read_table(f)
        ids.append(t.column('MVT_ID_mvt').to_numpy())
        for c in AB:
            cols[c].append(t.column(c).to_numpy(zero_copy_only=False).astype(float) if c in t.column_names else np.full(t.num_rows, np.nan))
    ids = np.concatenate(ids)
    _, first = np.unique(ids, return_index=True)
    return ids[first], {c: np.concatenate(v)[first] for c, v in cols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--sub', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--months', default='01,07')
    ap.add_argument('--ref-map', default='')
    ap.add_argument('--cv-map', default='')
    ap.add_argument('--names', default='')
    a = ap.parse_args()
    MONTHS = a.months.split(',')
    RMAP = dict([(m, m) for m in MONTHS], **dict(x.split(':') for x in a.ref_map.split(',') if x))
    CMAP = dict([(m, m) for m in MONTHS], **dict(x.split(':') for x in a.cv_map.split(',') if x))
    NM = dict(dict(zip(('variant_simple', 'variant_merged', 'variant_merged_sc', 'variant_norevert', 'res'), ('variant_simple', 'variant_merged', 'variant_merged_sc', 'variant_norevert', 'variant_full'))),
              **dict(x.split('=') for x in a.names.split(',') if x))
    t = pq.read_table(os.path.join(a.dir, 'tk_main.parquet'),
                      columns=['MVT_ID_mvt', 'yr', 'date', 'ADEP_mvt', 'tk_stop'] + [c for c in TK_ALL if c != 'tk_stop_code'])
    mid = t.column('MVT_ID_mvt').to_numpy()
    yr = t.column('yr').to_numpy()
    date = t.column('date').to_numpy(zero_copy_only=False).astype(str)
    apt = t.column('ADEP_mvt').to_numpy(zero_copy_only=False).astype(str)
    mo = np.array([s[5:7] for s in date])
    day = np.array([int(s[8:10]) for s in date])
    TK = {c: col(t, c) for c in TK_ALL}
    fid, F = load_feats(a.dir)
    pos = np.searchsorted(fid, mid)
    hit = (pos < len(fid)) & (fid[np.minimum(pos, len(fid) - 1)] == mid)
    ABv = {c: np.where(hit, F[c][np.minimum(pos, len(fid) - 1)], np.nan) for c in AB}
    gate = ~np.isnan(ABv['ab_obs_first'])
    out = dict(rules=dict(PSI_T=PSI_T, MISS_T=MISS_T, REL=REL, MIN_N=MIN_N, K_T=K_T, S_T=S_T, S_REL=S_REL),
               rows=int(len(mid)), rows_2026=int((yr == 2026).sum()), feats_hit=int(hit.sum()), gated=int(gate.sum()))

    # Cells: airport x month, plus the EDDF-01 early / late parts
    cells = {}
    for ap_ in sorted(set(apt)):
        for m in MONTHS:
            ref = (apt == ap_) & (mo == RMAP[m]) & (yr == 2025)
            obj = (apt == ap_) & (mo == m) & (yr == 2026)
            cells[f'{ap_}-{m}'] = (ref, obj)
    ref, obj = cells.get('EDDF-01', (None, None))
    if ref is not None: cells['EDDF-01pre'] = (ref, obj & (day <= 28))
    if ref is not None: cells['EDDF-01post'] = (ref, obj & (day >= 29))
    V = dict(TK, **ABv)
    res, gr = {}, {}
    for k, (ref, obj) in cells.items():
        gr[k] = dict(n25=int(ref.sum()), g25=round(float(gate[ref].mean()), 4) if ref.any() else None,
                     n26=int(obj.sum()), g26=round(float(gate[obj].mean()), 4) if obj.any() else None,
                     ng25=int((ref & gate).sum()), ng26=int((obj & gate).sum()))
        if gr[k]['g25'] and gr[k]['g26'] is not None:
            gr[k]['ratio'] = round(gr[k]['g26'] / gr[k]['g25'], 3)
        if (ref & gate).sum() < MIN_N or (obj & gate).sum() < MIN_N:
            res[k] = None
            continue
        res[k] = {c: stats(V[c][ref & gate], V[c][obj & gate]) for c in TK_ALL + AB}
    out['gate_rate'] = gr

    # Null: split the same 2025 month into days <=15 and >=16
    null = {}
    for k, (ref, _) in cells.items():
        if k.endswith(('pre', 'post')):
            continue
        h1, h2 = ref & gate & (day <= 15), ref & gate & (day >= 16)
        if h1.sum() >= MIN_N and h2.sum() >= MIN_N:
            null[k] = {c: round(psi(V[c][h1], V[c][h2]), 4) for c in TK_ALL + AB}
    nv = np.array([v for d in null.values() for v in d.values()])
    out['null'] = dict(cells=len(null), n=int(len(nv)), frac_psi_ge_T=round(float((nv >= PSI_T).mean()), 4),
                       max=round(float(nv.max()), 4), p95=round(float(np.quantile(nv, .95)), 4),
                       top=sorted(((v, k, c) for k, d in null.items() for c, v in d.items()), reverse=True)[:8])

    # Column-level triggers (relative condition: median over the other full-month cells of the same column and month)
    full = [k for k in res if res[k] is not None and not k.endswith(('pre', 'post'))]
    trig, glob_cols = {}, {}
    for k in res:
        if res[k] is None:
            continue
        m = k[5:7]
        base = k[:7]
        peers = [p for p in full if p[5:7] == m and p != base]
        tk_t, ab_t = [], []
        for c in TK_ALL + AB:
            s = res[k][c]
            med_p = float(np.median([res[p][c]['psi'] for p in peers]))
            med_m = float(np.median([abs(res[p][c]['dmiss']) for p in peers]))
            f1 = s['psi'] >= PSI_T and s['psi'] >= REL * med_p
            f2 = abs(s['dmiss']) >= MISS_T and abs(s['dmiss']) >= REL * med_m
            s['med_psi_peers'], s['med_dmiss_peers'] = round(med_p, 4), round(med_m, 4)
            s['trig'] = bool(f1 or f2)
            if s['trig']:
                (tk_t if c in TK_ALL else ab_t).append(c)
        trig[k] = dict(K=len(tk_t), tk=tk_t, K_AB=len(ab_t), ab=ab_t)
    for c in TK_ALL + AB:
        n_hi = sum(res[k][c]['psi'] >= PSI_T for k in full)
        if n_hi * 2 >= len(full):
            glob_cols[c] = n_hi
    out['global_drift_cols'] = dict(n_cells=len(full), cols=glob_cols)

    # Candidate differences (2026 rows, in submission row order); gated = variant_full.res.npy != 0
    def sub(nm):
        s = pq.read_table(os.path.join(a.sub, f'upstanding-firefly_{NM[nm]}.parquet'))
        return s.column(0).to_numpy(), s.column(1).to_numpy().astype(float)
    sid, psimple = sub('variant_simple')
    P = {'variant_simple': psimple}
    for nm in ('variant_merged', 'variant_merged_sc', 'variant_norevert'):
        i, p = sub(nm)
        assert (i == sid).all()
        P[nm] = p
    sg = np.load(os.path.join(a.sub, 'upstanding-firefly_' + NM['res'] + '.res.npy')) != 0
    k = pq.read_table(os.path.join(a.sub, 'ranking.parquet'), columns=['MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'STAND_mvt'])
    kid = k.column(0).to_numpy()
    o = np.argsort(kid)
    ix = o[np.searchsorted(kid[o], sid)]
    assert (kid[ix] == sid).all()
    sapt = k.column(1).to_numpy(zero_copy_only=False).astype(str)[ix]
    smo = pc.month(k.column(2)).to_numpy(zero_copy_only=False)[ix]
    sday = pc.day(k.column(2)).to_numpy(zero_copy_only=False)[ix]
    stand = k.column(3).to_numpy(zero_copy_only=False)[ix]
    sc = pq.read_table(os.path.join(a.dir, 'sc_table_all_full.parquet'))
    sck = set(sc.column(0).to_numpy(zero_copy_only=False).tolist())
    schit = np.array([(f'{a_}|{s_}' in sck) if s_ is not None else False for a_, s_ in zip(sapt, stand)])

    def smask(kk):
        ap_, m = kk[:4], int(kk[5:7])
        mk = (sapt == ap_) & (smo == m)
        if kk.endswith('pre'):
            mk &= sday <= 28
        elif kk.endswith('post'):
            mk &= sday >= 29
        return mk

    def ms(d, mk):
        x = d[mk]
        return dict(n=int(mk.sum()), mean=round(float(x.mean()), 2), rms=round(float(np.sqrt((x ** 2).mean())), 2)) if mk.any() else None

    shifts = {}
    for kk in cells:
        mk = smask(kk) & sg
        shifts[kk] = dict(tk=ms(P['variant_merged'] - P['variant_simple'], mk), sc=ms(P['variant_merged_sc'] - P['variant_merged'], mk), tk_and_sc=ms(P['variant_merged_sc'] - P['variant_simple'], mk),
                          variant_norevert_minus_variant_merged_sc=ms(P['variant_norevert'] - P['variant_merged_sc'], mk), sc_hit=round(float(schit[mk].mean()), 4) if mk.any() else None)
    out['shifts_2026_gated'] = shifts

    # Cell verdict. Values written to the report (also read by run_submission_chain.sh):
    # insufficient_sample, revert_to_simple_variant, column_downgrade_needs_retrain, observe_only, keep
    verdict = {}
    for kk in cells:
        s = shifts[kk]['tk']
        S = s['mean'] if s else None
        cv = CV_SHIFT.get(kk[:5] + CMAP.get(kk[5:7], kk[5:7]))
        b = S is not None and abs(S) >= S_T and (cv is None or abs(S) >= S_REL * abs(cv))
        if res.get(kk) is None:
            v = 'insufficient_sample'
        else:
            K = trig[kk]['K']
            v = 'revert_to_simple_variant' if (K >= K_T and b) else ('column_downgrade_needs_retrain' if (1 <= K < K_T and b) else 'observe_only' if K >= K_T else 'keep')
        verdict[kk] = dict(verdict=v, K=trig.get(kk, {}).get('K'), tk=trig.get(kk, {}).get('tk'), K_AB=trig.get(kk, {}).get('K_AB'),
                           ab=trig.get(kk, {}).get('ab'), S=S, S_cv=cv, b=bool(b), gate_ratio=gr[kk].get('ratio'))
    out['verdict'] = verdict
    out['stats'] = res

    # Size of the revert: gated rows of cells judged revert (EDDF-01 by part if its two parts differ) are set back to variant_simple
    rb = np.zeros(len(sid), bool)
    for kk, v in verdict.items():
        if v['verdict'] != 'revert_to_simple_variant':
            continue
        if kk == 'EDDF-01' and ({verdict['EDDF-01pre']['verdict'], verdict['EDDF-01post']['verdict']} - {'revert_to_simple_variant'}):
            continue  # parts differ: revert by part
        rb |= smask(kk) & sg
    new = P['variant_norevert'].copy()
    new[rb] = P['variant_simple'][rb]
    d = new - P['variant_norevert']
    out['rollback'] = dict(rows=int(rb.sum()), changed=int((d != 0).sum()), mean_all=round(float(d.mean()), 3),
                           rms_all=round(float(np.sqrt((d ** 2).mean())), 3),
                           mean_rows=round(float(d[rb].mean()), 3) if rb.any() else 0.0,
                           rms_rows=round(float(np.sqrt((d[rb] ** 2).mean())), 3) if rb.any() else 0.0,
                           gated_variant_norevert_eq_variant_merged_sc=bool((P['variant_norevert'][sg] == P['variant_merged_sc'][sg]).all()))

    # Side evidence: EDDF 2026-01 against the winter pool Nov/Dec (tk_ok==1 as gating proxy)
    tp = pq.read_table(os.path.join(a.dir, 'tk_pool.parquet'), columns=['date', 'ADEP_mvt', 'tk_ok', 'tk_stop'] +
                       [c for c in TK_ALL if c != 'tk_stop_code'])
    pa_ = tp.column('ADEP_mvt').to_numpy(zero_copy_only=False).astype(str)
    pok = tp.column('tk_ok').to_numpy() == 1
    PT = {c: col(tp, c) for c in TK_ALL}
    pm = (pa_ == 'EDDF') & pok
    pool = {}
    for kk in ('EDDF-01pre', 'EDDF-01post'):
        _, obj = cells[kk]
        pool[kk] = {c: stats(PT[c][pm], TK[c][obj & gate]) for c in TK_ALL}
    ref25 = cells['EDDF-01'][0] & gate
    pool['pool_vs_2025-01'] = {c: stats(TK[c][ref25], PT[c][pm]) for c in TK_ALL}
    out['eddf_pool'] = dict(n_pool=int(pm.sum()), cmp=pool)
    # Side evidence (not in the verdict): winter pool, seven airports: number of TK columns with PSI >= 0.25 for pool vs 2025-01 and 2026-01 vs pool, and the 2026-01 TK mean shift
    wp = {}
    for ap_ in ('EDDF', 'EDDM', 'EGLL', 'EHAM', 'LEBL', 'LSZH', 'LEMD'):
        pmk = (pa_ == ap_) & pok
        r25 = cells[f'{ap_}-01'][0] & gate
        o26 = cells[f'{ap_}-01'][1] & gate
        e = dict(n_pool=int(pmk.sum()), n25=int(r25.sum()), n26=int(o26.sum()), S26=(shifts[f'{ap_}-01']['tk'] or {}).get('mean'))
        if pmk.sum() >= MIN_N and o26.sum() >= MIN_N:
            p26 = {c: psi(PT[c][pmk], TK[c][o26]) for c in TK_ALL}
            e['n_psi_hi_26_vs_pool'] = sum(v >= PSI_T for v in p26.values())
            e['mean_psi_26_vs_pool'] = round(float(np.mean(list(p26.values()))), 3)
        if pmk.sum() >= MIN_N and r25.sum() >= MIN_N:
            p25 = {c: psi(TK[c][r25], PT[c][pmk]) for c in TK_ALL}
            e['n_psi_hi_pool_vs_25'] = sum(v >= PSI_T for v in p25.values())
            e['mean_psi_pool_vs_25'] = round(float(np.mean(list(p25.values()))), 3)
        wp[ap_] = e
    out['winter_pool_side'] = wp

    # CV-based cost of the revert (estimate): per reverted cell, CV delta-MSE per gated row (TK full + stand correction) x number of reverted 2026 rows / N26;
    # Opposite extreme: if the change is unrelated to the truth, the mean square of the change = upper bound of the revert gain
    cvj = os.path.join(os.path.dirname(a.out), '20260926-v22-cv_eval.json')
    if os.path.exists(cvj):
        cv = json.load(open(cvj))['var']['all']
        N26 = len(sid)
        cost, risk = {}, {}
        for kk, v in verdict.items():
            if v['verdict'] != 'revert_to_simple_variant' or (kk == 'EDDF-01' and 'EDDF-01post' in verdict):
                continue
            n = int((smask(kk) & sg).sum())
            c7 = kk[:7]
            tk_ = cv['full']['cell'].get(c7, {}).get('dmse_per_gated_row', 0.0)
            sc_ = cv['full_sc_vs_full']['cell'].get(c7, {}).get('dmse_per_gated_row', 0.0)
            cost[kk] = dict(n=n, tk=round(n * tk_ / N26, 2), sc=round(n * sc_ / N26, 2))
            dd = (P['variant_merged_sc'] - P['variant_simple'])[smask(kk) & sg]
            risk[kk] = round(float((dd ** 2).sum() / N26), 2)
        dm = sum(x['tk'] + x['sc'] for x in cost.values())
        rk = sum(risk.values())
        out['rollback_cost'] = dict(cells=cost, cv_dmse_lost=round(dm, 2), change_msq_per_row=risk,
                                    extreme_msq=round(rk, 2))

    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(dict(null=out['null'], global_drift_cols=out['global_drift_cols'], rollback=out['rollback']), ensure_ascii=False, default=str))
    for kk, v in verdict.items():
        print(kk, gr[kk], shifts[kk]['tk'], v)


if __name__ == '__main__':
    main()
