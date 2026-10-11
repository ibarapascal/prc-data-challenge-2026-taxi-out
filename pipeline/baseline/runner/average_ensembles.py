"""average_ensembles: averaging script for baseline = mean(ensemble A, ensemble B) (rebuilt version).

Note: the original script that produced the averaged baseline is no longer available; this script is rebuilt from the recipe "pandas: merge the three versions -> (A+B)/2 round -> int32".
It was checked against the previously produced files: taking (A+B)/2 on the integer values of the prediction files and rounding **half to even (np.rint = pandas round)**
matches the reference averaged file row by row (0 differences); floor and round-half-up each differ on many rows. The row count is not hard-coded: it follows --template (submitting.parquet).
Usage: python average_ensembles.py --a <A.parquet> --b <B.parquet> --out <avg.parquet> [--template submitting.parquet] [--check <reference averaged .parquet>]
Dependencies: pyarrow, numpy (runs in seconds).
"""
import argparse
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([('MVT_ID_mvt', pa.float64()), ('TAXITIME_SEC_mvt', pa.int32())])


def load(p):
    t = pq.read_table(p)
    assert t.schema.remove_metadata().equals(SCHEMA), (p, t.schema)
    assert t.column(0).null_count == 0 and t.column(1).null_count == 0, p
    return t.column(0).to_numpy(), t.column(1).to_numpy().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--template'); ap.add_argument('--check')
    a = ap.parse_args()
    ia, pa_ = load(a.a); ib, pb = load(a.b)
    assert len(ia) == len(ib) and (ia == ib).all(), 'the two versions have different row order'
    assert len(np.unique(ia)) == len(ia), 'duplicate MVT_ID'
    if a.template:
        tid = pq.read_table(a.template, columns=['MVT_ID_mvt']).column(0).to_numpy()
        assert len(tid) == len(ia) and (tid == ia).all(), 'row order / row count differs from the submission template'
    p = np.rint((pa_ + pb) / 2.0).astype(np.int32)
    pq.write_table(pa.table([pa.array(ia, pa.float64()), pa.array(p, pa.int32())], schema=SCHEMA), a.out)
    rep = dict(rows=int(len(ia)), mean=round(float(p.mean()), 3), out=a.out, md5=hashlib.md5(open(a.out, 'rb').read()).hexdigest())
    if a.check:
        ic, pc = load(a.check)
        assert (ic == ia).all(), 'the reference file has a different row order'
        d = p.astype(np.int64) - pc
        rep['check'] = dict(ref=a.check, n_diff=int((d != 0).sum()), max_abs=int(np.abs(d).max()), rms=round(float(np.sqrt((d ** 2).mean())), 4))
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == '__main__':
    main()
