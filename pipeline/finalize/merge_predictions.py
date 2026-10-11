"""merge_predictions: on top of merge_base.py (candidate on gated rows + w * D tabular term on non-gated, non-B rows), set the gated rows of the cells
named by the drift-gate verdict (airport x month, optionally with a day condition) back to the values of the simpler residual variant (variant_simple);
non-gated rows (baseline + w * D) are untouched.

Relation to merge_base.py: imports load / diff / md5 / constants from merge_base.py; the merge logic is copied line by line. Without --revert-cells the output
is byte-identical to that of merge_base.py (checked).
Additional arguments:
  --revert-cells  comma-separated cells, each `APT:MM[condition]`, the condition being optionally `>=D`, `<=D` or `=D1-D2` (UTC day of month, taken from
                  ranking.MVT_TIME_UTC_mvt), e.g. `EDDF:07,EDDF:01>=29,LSZH:07`; default = no revert
  --simple_file           variant_simple parquet (required with --revert-cells); only the values of its gated rows are used
  --ref           optional reference file (e.g. the merged file without reverts): reports the row-wise difference of out against ref and checks that
                  "rows outside the reverted cells equal ref" and "rows above 20,000 s equal ref"
The other arguments are as in merge_base.py (--tab can be any tabterm-family npz, e.g. the 8-seed version; --n26 / --n-gated as in merge_base.py, final-phase options, default = old behaviour).
Usage (reproduces the merged file with reverts):
  python3 merge_predictions.py --cand SP/sub/upstanding-firefly_variant_merged_sc.parquet --refpred SP/sub/upstanding-firefly_v20.parquet --ensavg SP/sub/upstanding-firefly_v8.parquet \
    --res SP/sub/upstanding-firefly_variant_full.res.npy --tab SP/sub/tabterm_all_te2.npz --keys SP/sub/ranking.parquet --w 0.25 \
    --revert-cells EDDF:07,EDDF:01>=29,LSZH:07 --simple_file SP/sub/upstanding-firefly_variant_simple.parquet --ref SP/sub/upstanding-firefly_v21.parquet \
    --out SP/sub/upstanding-firefly_variant_reverted.parquet --report merge_predictions_report.json
"""
import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc_
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_base import SCHEMA, add_n_args, diff, load, md5, resolve_n26  # noqa: E402

CELL_RE = re.compile(r'^([A-Z]{4}):(\d{1,2})(?:(>=|<=)(\d{1,2})|=(\d{1,2})-(\d{1,2}))?$')


def parse_cells(s):
    out = []
    for tok in [t.strip() for t in s.split(',') if t.strip()]:
        m = CELL_RE.match(tok)
        assert m, f'bad cell format: {tok}'
        apt, mo = m.group(1), int(m.group(2))
        lo, hi = 1, 31
        if m.group(3) == '>=':
            lo = int(m.group(4))
        elif m.group(3) == '<=':
            hi = int(m.group(4))
        elif m.group(5):
            lo, hi = int(m.group(5)), int(m.group(6))
        out.append(dict(spec=tok, apt=apt, mo=mo, lo=lo, hi=hi))
    return out


def st(d, mk):
    x = d[mk].astype(np.float64)
    return dict(rows=int(mk.sum()), changed=int((x != 0).sum()),
                mean=round(float(x.mean()), 3) if len(x) else 0.0, rms=round(float(np.sqrt((x ** 2).mean())), 3) if len(x) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    for k in ('cand', 'refpred', 'ensavg', 'res', 'tab', 'keys', 'out', 'report'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--w', type=float, default=0.25)
    ap.add_argument('--revert-cells', default='')
    ap.add_argument('--simple_file', default=None)
    ap.add_argument('--ref', default=None)
    add_n_args(ap)
    a = ap.parse_args()
    n26 = resolve_n26(a)
    cells = parse_cells(a.revert_cells) if a.revert_cells else []
    assert not cells or a.simple_file, '--simple_file is required with --revert-cells'

    # ---- same as main() of merge_base.py up to the output checks ----
    idc, pc = load(a.cand, n26); id20, p20 = load(a.refpred, n26); id8, p8 = load(a.ensavg, n26)
    assert (idc == id20).all() and (id8 == id20).all(), 'row order differs'
    assert len(np.unique(id20)) == n26
    gate = np.load(a.res) != 0
    assert gate.shape == (n26,) and (a.n_gated == 'auto' or int(gate.sum()) == int(a.n_gated)), int(gate.sum())
    assert (pc[~gate] == p8[~gate]).all() and (p20[~gate] == p8[~gate]).all(), 'non-gated rows differ from ensavg (basis mismatch)'
    k = pq.read_table(a.keys, columns=['MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt'])
    kid = k.column(0).to_numpy(); pos = {v: i for i, v in enumerate(kid)}
    ix = np.array([pos[v] for v in id20])
    apt = k.column(1).to_numpy(zero_copy_only=False)[ix]
    mo = pc_.month(k.column(2)).to_numpy(zero_copy_only=False)[ix]
    day = pc_.day(k.column(2)).to_numpy(zero_copy_only=False)[ix]
    rep = dict(w=a.w, revert_cells=a.revert_cells or None, files={}, checks={})
    if a.tab == 'none':
        base = pc.copy()
        rep['checks']['tab'] = 'none: tabular term not added, base = cand'
    else:
        z = np.load(a.tab)
        tpos = {v: i for i, v in enumerate(z['MVT_ID'])}
        assert len(tpos) == n26 and set(tpos) == set(id20.tolist()), 'MVT_ID of tab does not match the submission rows'
        ti = np.array([tpos[v] for v in id20])
        D = z['D'][ti].astype(np.float64); B = z['B'][ti].astype(bool)
        assert np.isfinite(D).all() and (D[B] == 0).all()
        mk = ~gate & ~B
        base = pc.copy()
        base[mk] = np.clip(np.rint(pc[mk] + a.w * D[mk]), 30, None).astype(np.int64)
        assert (base[gate] == pc[gate]).all() and (base[B] == pc[B]).all()
        rep['checks'].update(tab_rows_applied=int(mk.sum()), tab_B_rows_nongated=int((~gate & B).sum()),
                             wD_nongated=dict(mean=round(float(a.w * D[mk].mean()), 3), rms=round(float(a.w * np.sqrt((D[mk] ** 2).mean())), 3)))

    # ---- revert: gated rows in the named cells -> variant_simple ----
    new = base.copy()
    rb = np.zeros(n26, bool)
    if cells:
        id_simple, p_simple = load(a.simple_file, n26)
        assert (id_simple == id20).all(), 'variant_simple row order differs'
        blocks = {}
        for c in cells:
            m = gate & (apt == c['apt']) & (mo == c['mo']) & (day >= c['lo']) & (day <= c['hi'])
            assert not (m & rb).any(), f'cells overlap: {c["spec"]}'
            rb |= m
            blocks[c['spec']] = m
        new[rb] = p_simple[rb]
        d = new - base
        assert (new[~rb] == base[~rb]).all() and (new[~gate] == base[~gate]).all()
        rep['revert'] = dict(rows=int(rb.sum()), changed=int((d != 0).sum()),
                             blocks={s: st(d, m) for s, m in blocks.items()}, rows_only=st(d, rb), all=st(d, np.ones(n26, bool)))

    if a.tab == 'none' and not cells:
        shutil.copyfile(a.cand, a.out)  # as in merge_base.py: byte-identical to cand
    else:
        pq.write_table(pa.table([pa.array(id20, pa.float64()), pa.array(new.astype(np.int32), pa.int32())], schema=SCHEMA), a.out)

    # ---- output checks ----
    ido, po = load(a.out, n26)
    assert (ido == id20).all(), 'output row order differs'
    assert (po == new).all()
    rep['checks'].update(rows=n26, n_gated=int(gate.sum()), schema_eq_refpred=True, no_null=True, row_order_eq_refpred=True,
                         gated_rows_eq_cand_outside_revert=bool((po[gate & ~rb] == pc[gate & ~rb]).all()),
                         nongated_rows_eq_base=bool((po[~gate] == base[~gate]).all()),
                         min_pred=int(po.min()), max_pred=int(po.max()))
    rep['diff_vs_cand'] = diff(pc, po, apt, mo, gate)
    rep['diff_vs_refpred'] = diff(p20, po, apt, mo, gate)
    if a.ref:
        idr, pr = load(a.ref, n26)
        assert (idr == id20).all(), 'ref row order differs'
        dr = po - pr
        big = (pr > 20000) | (po > 20000)
        rep['vs_ref'] = dict(changed_rows=int((dr != 0).sum()), changed_outside_revert=int((dr[~rb] != 0).sum()),
                             changed_rows_all_in_revert_gated=bool(((dr != 0) <= (rb & gate)).all()),
                             gt20000=dict(rows=int(big.sum()), equal=bool((po[big] == pr[big]).all())),
                             all=st(dr, np.ones(n26, bool)), revert_rows=st(dr, rb),
                             by_apt_mo_nonzero={kk: v for kk, v in diff(pr, po, apt, mo, gate)['by_apt_mo'].items() if v['changed'] > 0})
    for nm, p in (('out', a.out), ('cand', a.cand), ('refpred', a.refpred), ('ensavg', a.ensavg), ('res', a.res), ('tab', a.tab if a.tab != 'none' else None),
                  ('simple_file', a.simple_file), ('ref', a.ref)):
        if p:
            rep['files'][nm] = dict(path=p, md5=md5(p))
    with open(a.report, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print(json.dumps({kk: rep[kk] for kk in ('files', 'checks', 'revert', 'vs_ref') if kk in rep}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
