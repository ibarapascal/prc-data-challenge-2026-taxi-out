"""PRC final phase: download the ADSB.lol prod-0 archive volumes of February and June 2026 (plus the boundary days 03-01 and 05-31) to the data directory; download only, no unpacking.

Usage: python download_adsb.py (in the same directory as manifest.json)
- 4 concurrent curl processes, each with --limit-rate 750000, total rate <= 3 MB/s.
    The environment variable DL_RATE_EACH changes the per-connection limit; the limit is fixed when curl starts and cannot be changed for running downloads.
- Resumable downloads (.part); renamed only after the size and source SHA256 check (most 2025 releases carry no source digest: only the size is checked and the local SHA256 is recorded); existing files are skipped, so the script can be rerun from scratch.
- The same script is also used for the supplementary 2025 download directory (with a different manifest).
- A STOP file stops it immediately; it also stops when free disk space < 100 GiB + remaining download size.
- When all volumes of a day are complete, writes raw/<date>.READY (consumed by the cropping task).
"""
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RATE_EACH = int(os.environ.get('DL_RATE_EACH', 750_000))
WORKERS = 4
lock = threading.Lock()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(name, value):
    path = ROOT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fetch(a, raw, state):
    target, part = raw / a['name'], raw / (a['name'] + '.part')
    if (ROOT / 'STOP').exists():
        raise RuntimeError('STOP marker')
    if target.exists():
        if target.stat().st_size != a['size']:
            raise RuntimeError('existing file wrong size: ' + a['name'])
        path = target
    else:
        print(now(), 'START', a['name'], a['size'], flush=True)
        cmd = ['/usr/bin/curl', '--location', '--fail', '--silent', '--show-error',
               '--limit-rate', str(RATE_EACH), '--continue-at', '-', '--retry', '6',
               '--retry-delay', '20', '--retry-all-errors', '--connect-timeout', '30',
               '--proto', '=https', '--proto-redir', '=https', '--output', str(part), a['url']]
        for attempt in range(4):
            if subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode == 0 and part.stat().st_size == a['size']:
                break
            if (ROOT / 'STOP').exists():
                raise RuntimeError('STOP marker')
            time.sleep(60)
        else:
            raise RuntimeError('download failed: ' + a['name'])
        path = part
    digest = sha256(path)
    if a.get('digest') and a['digest'] != 'sha256:' + digest:
        if path == part:
            part.unlink()
        raise RuntimeError('SHA256 mismatch: ' + a['name'])
    if path == part:
        os.replace(part, target)
    with lock:
        state['completed'].append({'name': a['name'], 'date': a['date'], 'size': a['size'],
                                   'sha256': digest, 'completed_at': now()})
        state['updated_at'] = now()
        write_json('status.json', state)
    print(now(), 'VERIFIED', a['name'], flush=True)
    return a


def main():
    m = json.loads((ROOT / 'manifest.json').read_text())
    assert m['root'] == str(ROOT)
    assert sum(a['size'] for a in m['assets']) == m['total_bytes']
    raw = ROOT / 'raw'
    raw.mkdir(exist_ok=True)
    remaining = sum(a['size'] for a in m['assets'] if not (raw / a['name']).exists())
    if shutil.disk_usage(ROOT).free < m['minimum_disk_reserve_bytes'] + remaining:
        raise RuntimeError('disk reserve below 100GiB + remaining')
    state = {'started_at': now(), 'pid': os.getpid(), 'state': 'downloading', 'workers': WORKERS,
             'rate_each_Bps': RATE_EACH, 'total_bytes': m['total_bytes'], 'completed': []}
    write_json('status.json', state)
    by_day = {}
    for a in m['assets']:
        by_day.setdefault(a['date'], []).append(a['name'])
    done = set()
    try:
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            futs = [ex.submit(fetch, a, raw, state) for a in m['assets']]
            for f in cf.as_completed(futs):
                try:
                    a = f.result()
                except BaseException:
                    (ROOT / 'STOP').write_text('auto-stop after error')
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
                done.add(a['name'])
                if all(n in done for n in by_day[a['date']]):
                    (raw / (a['date'] + '.READY')).write_text(now())
        state.update(state='complete', completed_at=now(),
                     verified_bytes=sum(x['size'] for x in state['completed']))
        write_json('DOWNLOAD_COMPLETE.json', state)
        write_json('status.json', state)
        print(now(), 'DOWNLOAD_COMPLETE', state['verified_bytes'], flush=True)
    except BaseException as exc:
        (ROOT / 'STOP').write_text('auto-stop after error: ' + repr(exc))
        state.update(state='stopped', error=repr(exc), updated_at=now())
        write_json('status.json', state)
        print(now(), 'STOPPED', repr(exc), flush=True)
        raise


if __name__ == '__main__':
    main()
