"""Crop ADSB.lol globe_history traces by day: keep only points on the ground or at low altitude inside the bounding boxes of the 10 airports, written to parquet.

Usage: python crop_day.py --parts <volumes...> --airports airports.json --out <day.parquet> [--max-alt 4000]
      or --parts - (read the concatenated tar stream from stdin, so that volumes can be piped in from another machine)
- Streams the tar (several volumes concatenated in the given order) without unpacking to disk; parses one trace JSON at a time.
- Coarse filter: samples every 16th point and checks whether it falls inside the enlarged airport box (+-0.6 deg); if not, the whole trace is skipped.
- Output columns: hex, apt(0..9), t(epoch s), lat, lon, alt(int, ground=-1000, missing=-9999), gs, trk,
  flags, src, cs(flight from the most recent metadata), cs_age(s), actype, reg.
- Traces above 16 MB compressed or 32 MB decompressed are skipped (counted in skipped_big).
- The writer is flushed every 400k rows, so memory is independent of the day's data volume; fails immediately if RSS exceeds --max-rss-mb.
"""
import argparse
import gzip
import io
import json
import os
import sys
import tarfile
import time

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

GROUND = -1000
MISSING = -9999
SCHEMA = pa.schema([
    ('hex', pa.string()), ('apt', pa.int8()), ('t', pa.float64()),
    ('lat', pa.float64()), ('lon', pa.float64()), ('alt', pa.int32()),
    ('gs', pa.float32()), ('trk', pa.float32()), ('flags', pa.int16()),
    ('src', pa.string()), ('cs', pa.string()), ('cs_age', pa.float32()),
    ('actype', pa.string()), ('reg', pa.string()),
])
COLS = [f.name for f in SCHEMA]


class Parts(io.RawIOBase):
    def __init__(self, parts):
        self.parts = iter(parts)
        self.cur = None

    def readable(self):
        return True

    def readinto(self, buf):
        used = 0
        while used < len(buf):
            if self.cur is None:
                try:
                    self.cur = open(next(self.parts), 'rb', buffering=0)
                except StopIteration:
                    break
            n = self.cur.readinto(memoryview(buf)[used:])
            if not n:
                self.cur.close()
                self.cur = None
            used += n
        return used


def rss_mb():
    """Peak RSS (MB); Windows has no resource module, so the current RSS from psutil is used as an approximation."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 / 1024 if sys.platform == 'darwin' else r / 1024
    except ImportError:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='+', required=True)
    ap.add_argument('--airports', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-alt', type=int, default=4000)
    ap.add_argument('--max-rss-mb', type=int, default=900)
    ap.add_argument('--max-member-mb', type=float, default=16)
    a = ap.parse_args()

    apts = json.load(open(a.airports))
    names = sorted(apts)
    boxes = [tuple(apts[n]['bbox']) for n in names]  # south, west, north, east
    S = min(b[0] for b in boxes) - 0.6
    N = max(b[2] for b in boxes) + 0.6
    wide = [(b[0] - .6, b[1] - .6, b[2] + .6, b[3] + .6) for b in boxes]

    tmp = a.out + '.part'
    writer = pq.ParquetWriter(tmp, SCHEMA, compression='zstd')
    buf = {c: [] for c in COLS}
    stats = dict(members=0, docs_kept=0, rows=0, bad=0, skipped_big=0)
    t0 = time.time()

    def flush():
        if buf['t']:
            writer.write_table(pa.table(buf, schema=SCHEMA))
            stats['rows'] += len(buf['t'])
            for c in COLS:
                buf[c].clear()

    raw = sys.stdin.buffer.raw if a.parts == ['-'] else Parts(a.parts)
    with raw, io.BufferedReader(raw, 1 << 22) as stream, \
            tarfile.open(fileobj=stream, mode='r|*') as tar:
        for m in tar:
            if not m.isfile() or 'trace_full' not in m.name:
                continue
            stats['members'] += 1
            if m.size > a.max_member_mb * 2**20:  # oversized single trace (ground vehicles / anomalies) is skipped directly to protect the peak
                stats['skipped_big'] += 1
                continue
            packed = tar.extractfile(m).read()
            try:
                if packed[:2] == b'\x1f\x8b':
                    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as gz:
                        packed = gz.read(int(a.max_member_mb * 2 * 2**20) + 1)
                    if len(packed) > a.max_member_mb * 2 * 2**20:
                        stats['skipped_big'] += 1
                        continue
                doc = orjson.loads(packed)
                tr = doc['trace']
                ts = doc['timestamp']
            except Exception:
                stats['bad'] += 1
                continue
            # coarse filter: does the sampled point enter any enlarged airport box
            hit = False
            for r in tr[::16] + tr[-1:]:
                la = r[1]
                if S <= la <= N:
                    lo = r[2]
                    for b in wide:
                        if b[0] <= la <= b[2] and b[1] <= lo <= b[3]:
                            hit = True
                            break
                    if hit:
                        break
            if not hit:
                continue
            hexid = doc.get('icao')
            actype = doc.get('t')
            reg = doc.get('r')
            cs = None
            cs_t = None
            kept = 0
            for r in tr:
                t = ts + r[0]
                md = r[8] if len(r) > 8 else None
                if md:
                    fl = md.get('flight')
                    cs = fl.strip() or None if isinstance(fl, str) else None
                    cs_t = t
                la = r[1]
                if not (S <= la <= N):
                    continue
                alt = r[3]
                if alt == 'ground':
                    altv = GROUND
                elif alt is None:
                    altv = MISSING
                else:
                    try:
                        altv = int(alt)
                    except (TypeError, ValueError):
                        continue
                    if altv > a.max_alt:
                        continue
                lo = r[2]
                k = -1
                for i, b in enumerate(boxes):
                    if b[0] <= la <= b[2] and b[1] <= lo <= b[3]:
                        k = i
                        break
                if k < 0:
                    continue
                buf['hex'].append(hexid)
                buf['apt'].append(k)
                buf['t'].append(t)
                buf['lat'].append(la)
                buf['lon'].append(lo)
                buf['alt'].append(altv)
                buf['gs'].append(r[4])
                buf['trk'].append(r[5])
                buf['flags'].append(r[6])
                src = r[9] if len(r) > 9 else None
                buf['src'].append(src if isinstance(src, str) else None)
                buf['cs'].append(cs)
                buf['cs_age'].append(None if cs_t is None else t - cs_t)
                buf['actype'].append(actype)
                buf['reg'].append(reg)
                kept += 1
            if kept:
                stats['docs_kept'] += 1
            if len(buf['t']) >= 400_000:
                flush()
                if rss_mb() > a.max_rss_mb:
                    writer.close()
                    os.remove(tmp)
                    print(json.dumps({'error': 'rss_limit', 'rss_mb': rss_mb(), **stats}))
                    sys.exit(3)
    flush()
    writer.close()
    os.replace(tmp, a.out)
    stats.update(elapsed=round(time.time() - t0, 1), peak_rss_mb=round(rss_mb(), 1),
                 airports=names, parts=a.parts)
    with open(a.out + '.json', 'w') as f:
        json.dump(stats, f)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
