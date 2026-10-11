"""merge_base: merge of the residual-layer candidate with the shrunken tabular term (w * (new arm - base arm), added only on non-gated, non-B rows;
B = LIRF artefact class); writes the candidate file and runs all checks.

Inputs (all read-only, seconds; pyarrow + numpy are enough, polars is not needed):
  --cand  residual-layer candidate (default: the merged full variant variant_merged_sc); --refpred earlier-phase prediction file; --ensavg baseline prediction file
          (non-gated rows must equal --ensavg row by row)
  --res   any residual .res.npy of the variant_full / variant_simple family (res != 0 = gated rows; the gated sets of refpred / variant_simple / variant_full were checked to be identical, 223,745 rows)
  --tab   .npz of tabular_term.py (MVT_ID, D, B); none = no tabular term, the output is byte-identical to --cand
  --keys  raw/ranking.parquet (only MVT_ID_mvt, ADEP_mvt, MVT_TIME_UTC_mvt, used to report differences by airport x month)
  --w 0.25; --out output parquet; --report checks json
Final-phase options: --n26 344841|auto (auto = row count of --ensavg), --n-gated 223745|auto (auto = derived from --res, not asserted);
      the defaults reproduce the old behaviour byte for byte. In the final phase --refpred is the retrained baseline file, so the assertion "non-gated rows = baseline" holds naturally.
Checks: row count --n26 (default 344,841), row order equal to refpred, schema equal to refpred (MVT_ID_mvt double, TAXITIME_SEC_mvt int32), no nulls; gated rows equal to --cand row by row;
      B rows unchanged; row-wise differences against refpred and against --cand (overall / gated / non-gated / airport x month); md5 of every file.
Usage: python3 merge_base.py --cand SP/sub/upstanding-firefly_variant_merged_sc.parquet --refpred SP/sub/upstanding-firefly_v20.parquet --ensavg SP/sub/upstanding-firefly_v8.parquet \
        --res SP/sub/upstanding-firefly_variant_full.res.npy --tab SP/sub/tabterm_all_te2.npz --keys SP/sub/ranking.parquet --w 0.25 \
        --out SP/sub/upstanding-firefly_merged.parquet --report merge_base_report.json
"""
import argparse
import hashlib
import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc_
import pyarrow.parquet as pq

N26, N_GATED = 344841, 223745
SCHEMA = pa.schema([('MVT_ID_mvt', pa.float64()), ('TAXITIME_SEC_mvt', pa.int32())])


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def load(p, n=None):
    t = pq.read_table(p)
    assert t.schema.remove_metadata().equals(SCHEMA), (p, t.schema)
    assert t.num_rows == (N26 if n is None else n) and t.column(0).null_count == 0 and t.column(1).null_count == 0, p
    return t.column(0).to_numpy(), t.column(1).to_numpy().astype(np.int64)


def add_n_args(ap):
    ap.add_argument('--n26', default=str(N26), help='number of submission rows; auto = row count of the --ensavg file (default 344841 = old behaviour)')
    ap.add_argument('--n-gated', default=str(N_GATED), help='assertion on the number of gated rows; auto = derived from --res, not asserted (default 223745 = old behaviour)')


def resolve_n26(a):
    return pq.read_metadata(a.ensavg).num_rows if a.n26 == 'auto' else int(a.n26)


def diff(a, b, apt, mo, gate):
    d = (b - a).astype(np.float64)

    def st(mk):
        x = d[mk]
        return dict(n=int(mk.sum()), changed=round(float((x != 0).mean()), 4) if len(x) else 0.0,
                    mean=round(float(x.mean()), 3) if len(x) else 0.0, rms=round(float(np.sqrt((x ** 2).mean())), 3) if len(x) else 0.0)
    out = dict(all=st(np.ones(len(d), bool)), gated=st(gate), nongated=st(~gate), mean_d2=round(float((d ** 2).mean()), 2), by_apt_mo={})
    for ap_ in sorted(set(apt)):
        for m in sorted(set(mo)):
            mk = (apt == ap_) & (mo == m)
            if mk.any():
                out['by_apt_mo'][f'{ap_}-{m:02d}'] = dict(st(mk), ng_changed=round(float((d[mk & ~gate] != 0).mean()), 4) if (mk & ~gate).any() else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ('cand', 'refpred', 'ensavg', 'res', 'tab', 'keys', 'out', 'report'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--w', type=float, default=0.25)
    add_n_args(ap)
    a = ap.parse_args()
    n26 = resolve_n26(a)
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
    rep = dict(w=a.w, files={}, checks={})
    if a.tab == 'none':
        shutil.copyfile(a.cand, a.out)
        rep['checks']['tab'] = 'none: output = cand byte for byte'
    else:
        z = np.load(a.tab)
        tpos = {v: i for i, v in enumerate(z['MVT_ID'])}
        assert len(tpos) == n26 and set(tpos) == set(id20.tolist()), 'MVT_ID of tab does not match the submission rows'
        ti = np.array([tpos[v] for v in id20])
        D = z['D'][ti].astype(np.float64); B = z['B'][ti].astype(bool)
        assert np.isfinite(D).all() and (D[B] == 0).all()
        mk = ~gate & ~B
        new = pc.copy()
        new[mk] = np.clip(np.rint(pc[mk] + a.w * D[mk]), 30, None).astype(np.int64)
        assert (new[gate] == pc[gate]).all() and (new[B] == pc[B]).all()
        pq.write_table(pa.table([pa.array(id20, pa.float64()), pa.array(new.astype(np.int32), pa.int32())], schema=SCHEMA), a.out)
        rep['checks'].update(tab_rows_applied=int(mk.sum()), tab_B_rows_nongated=int((~gate & B).sum()),
                             wD_nongated=dict(mean=round(float(a.w * D[mk].mean()), 3), rms=round(float(a.w * np.sqrt((D[mk] ** 2).mean())), 3)))
    ido, po = load(a.out, n26)
    assert (ido == id20).all(), 'output row order differs'
    rep['checks'].update(rows=n26, n_gated=int(gate.sum()), schema_eq_refpred=True, no_null=True, row_order_eq_refpred=True, gated_rows_eq_cand=bool((po[gate] == pc[gate]).all()),
                         min_pred=int(po.min()), max_pred=int(po.max()))
    rep['diff_vs_cand'] = diff(pc, po, apt, mo, gate)
    rep['diff_vs_refpred'] = diff(p20, po, apt, mo, gate)
    for nm, p in (('out', a.out), ('cand', a.cand), ('refpred', a.refpred), ('ensavg', a.ensavg), ('res', a.res), ('tab', a.tab if a.tab != 'none' else None)):
        if p:
            rep['files'][nm] = dict(path=p, md5=md5(p))
    with open(a.report, 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ('files', 'checks')}, ensure_ascii=False, indent=1))
    print('vs_cand', rep['diff_vs_cand']['all'], 'nongated', rep['diff_vs_cand']['nongated'])
    print('vs_refpred', rep['diff_vs_refpred']['all'])


if __name__ == '__main__':
    main()
