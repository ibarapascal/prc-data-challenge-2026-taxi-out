"""PRC final phase: consumes the raw/<date>.READY markers of the February/June ADS-B downloader and crops day by day with the existing crop_day.py; exits only when the download is complete and all days are cropped.

Usage:
  /path/to/prc-root/.venv/bin/python -u crop_ready_days.py \
      --src /path/to/prc-archive/adsb-final-20261008 \
      --out /path/to/prc-root/adsb/crop \
      --crop-day /path/to/prc-root/adsb/code/crop_day.py \
      --airports /path/to/prc-root/adsb/code/airports.json --workers 2
  Options: --once (scan once, process the days that are ready now and exit), --dates 2026-02-01,2026-02-02 (only these days), --dry-run (print the plan only).

Behaviour:
- Pending days = days among all dates in manifest.json for which raw/<date>.READY exists and neither <out>/<yyyy>/<date>.parquet nor .parquet.json exists.
    Besides READY, a second check: the set of volume file names of that day in raw (excluding .part) must equal the manifest set for that day and each volume size must equal the manifest value; otherwise the day is skipped with a warning (nothing is deleted).
    Volume names accept prod-0 and staging-0; a day that mixes both releases, or has a whole .tar next to .tar.aa volumes, is recorded as failed at startup (needs a manual decision) and not cropped.
- Each day calls crop_day.py with the same arguments as the earlier cropping run: --parts <volumes sorted by file name> --airports <airports.json>
    --out <...>, other options at the crop_day defaults (--max-alt 4000, --max-rss-mb 900, --max-member-mb 16).
- Atomic output: crop_day first writes to <out>/<yyyy>/.tmp_final/<date>.parquet (it renames through .part itself); with rc=0 and the .json present,
    the parquet and then parquet.json are renamed into <out>/<yyyy>/; a day counts as complete when both exist. Log <out>/<yyyy>/<date>.log, claim file <date>.claim (compatible with the earlier cropping script).
- Can be rerun from scratch: finished days are skipped; leftover .tmp_final files from an interruption are overwritten next time. Single instance: flock on <out>/.crop_ready_days.lock, a second instance exits immediately.
- A failed day is retried at most --retries times (default 2); if it still fails it is recorded in the status file and the other days continue; exit code 2 if any day failed at the end.
- When idle (no day to do, no running process) it scans every --interval seconds (default 300); it exits only when the downloader has written DOWNLOAD_COMPLETE.json and all manifest dates are cropped
    (or judged permanently failed). Exit code 4 after --max-idle-hours (default 36) hours without progress (so the job does not hang forever if the downloader stalls).
    A STOP file / state=stopped in status.json only produces a warning and the script keeps waiting (the downloader may be restarted).
- Status file <src>/crop_ready_days.status.json (atomic write): done / failed / running / pending dates and timestamps.
- Memory: each crop_day process peaks at about 0.3-0.7 GiB (bounded by --max-rss-mb 900); with --workers 2 the total is <= about 2 GiB. This script itself only scans files.
Dependencies: standard library only (fcntl needs POSIX); crop_day.py needs orjson and pyarrow.
"""
import argparse
import datetime
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Some days in the manifest are published only as staging-0, so the pattern accepts both prod-0 and staging-0.
PAT = re.compile(r'^v(\d{4})\.(\d{2})\.(\d{2})-planes-readsb-(prod|staging)-0\.tar(\.[a-z]{2})?$')
HERE = Path(__file__).resolve().parent


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def log(*x):
    print(now(), *x, flush=True)


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def load_manifest(src, only=None):
    """When only (--dates) is given, only the volumes of those dates are parsed (names of other days that do not match the pattern must not cause an exit when restricted to some dates); default = old behaviour."""
    m = json.loads((src / 'manifest.json').read_text())
    want = {}
    for it in m['assets']:
        if only and it['date'] not in only:
            continue
        if not PAT.match(it['name']):
            raise SystemExit(f'manifest contains an unrecognised volume name: {it["name"]}')
        want.setdefault(it['date'], {})[it['name']] = int(it['size'])
    return want


def ambiguous_day(want_day):
    """The volumes of one day must come from the same release (prod-0 or staging-0) and have a single form (one whole .tar, or only .tar.aa/.ab...); otherwise the concatenated tar stream cannot be trusted; returns the reason.
    Example: one day has both a staging-0.tar and .tar.aa + .tar.ab volumes of different sizes; which one to use needs a manual decision."""
    ms = [PAT.match(n) for n in want_day]
    kinds = {m.group(4) for m in ms}
    whole = [n for n, m in zip(want_day, ms) if m.group(5) is None]
    if len(kinds) > 1:
        return f'a mix of several releases on the same day {sorted(kinds)}'
    if whole and len(want_day) > 1:
        return f'whole file {whole} coexists with split volumes ({len(want_day)} names)'
    return None


def day_parts(raw, date, want_day):
    """Check beyond READY: the set of file names and the sizes must equal the manifest. Returns the ordered list of volume paths, or (None, reason)."""
    y, mth, d = date.split('-')
    have = {p.name: p for p in raw.glob(f'v{y}.{mth}.{d}-planes-readsb-*-0.tar*') if PAT.match(p.name)}
    if set(have) != set(want_day):
        return None, f'volume set mismatch have={sorted(have)} want={sorted(want_day)}'
    for n, sz in want_day.items():
        if have[n].stat().st_size != sz:
            return None, f'{n} size {have[n].stat().st_size} != manifest {sz}'
    return [have[n] for n in sorted(have)], None


def is_done(out, date):
    yd = out / date[:4]
    return (yd / f'{date}.parquet').exists() and (yd / f'{date}.parquet.json').exists()


def start(a, out, date, parts):
    yd = out / date[:4]
    tmpd = yd / '.tmp_final'
    tmpd.mkdir(parents=True, exist_ok=True)
    for f in (tmpd / f'{date}.parquet', tmpd / f'{date}.parquet.json', tmpd / f'{date}.parquet.part'):
        f.unlink(missing_ok=True)
    (yd / f'{date}.claim').write_text(f'{os.getpid()} {now()}\n')
    logf = open(yd / f'{date}.log', 'w')
    cmd = [a.python, str(a.crop_day), '--parts', *map(str, parts),
           '--airports', str(a.airports), '--out', str(tmpd / f'{date}.parquet')]
    logf.write('$ ' + ' '.join(cmd) + '\n')
    logf.flush()
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT), logf


def finish(out, date, rc):
    """rc=0 and all outputs present -> rename to the final location (parquet first, json last; the json is the completion marker). Returns whether it succeeded."""
    yd = out / date[:4]
    tp, tj = yd / '.tmp_final' / f'{date}.parquet', yd / '.tmp_final' / f'{date}.parquet.json'
    if rc != 0 or not tp.exists() or not tj.exists():
        (yd / f'{date}.claim').unlink(missing_ok=True)
        return False
    os.replace(tp, yd / f'{date}.parquet')
    os.replace(tj, yd / f'{date}.parquet.json')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/path/to/prc-archive/adsb-final-20261008')
    ap.add_argument('--out', default='/path/to/prc-root/adsb/crop')
    ap.add_argument('--crop-day', default=str(HERE.parent / 'adsb_crop' / 'crop_day.py'))
    ap.add_argument('--airports', default=None, help='default = airports.json in the same directory as crop_day.py')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--interval', type=int, default=300)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--max-idle-hours', type=float, default=36)
    ap.add_argument('--dates', default='', help='comma-separated, only process these dates')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    raw = src / 'raw'
    a.crop_day = Path(a.crop_day).resolve()
    a.airports = Path(a.airports) if a.airports else a.crop_day.with_name('airports.json')
    for p in (src / 'manifest.json', a.crop_day, a.airports):
        if not p.exists():
            raise SystemExit(f'missing input: {p}')
    only = {x for x in a.dates.split(',') if x}
    want = load_manifest(src, only)
    if only - set(want):
        raise SystemExit(f'--dates contains dates outside the manifest: {sorted(only - set(want))}')
    dates = sorted(only or want)

    out.mkdir(parents=True, exist_ok=True)
    lockf = open(out / '.crop_ready_days.lock', 'w')
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another crop_ready_days instance is running (lock .crop_ready_days.lock), exiting')
    lockf.write(f'{os.getpid()} {now()}\n')
    lockf.flush()

    state_path = src / 'crop_ready_days.status.json'
    tries, failed, running, warned = {}, {}, {}, {}
    for d in dates:
        why = ambiguous_day(want[d])
        if why and not is_done(out, d):
            failed[d] = dict(rc=None, reason='volume form is not unique, needs a manual decision: ' + why, at=now())
            log('ambiguous', d, why)
    last_progress = time.time()
    log('start', dict(src=str(src), out=str(out), crop_day=str(a.crop_day), days=len(dates), workers=a.workers,
                      done_already=sum(is_done(out, d) for d in dates)))

    def save(phase):
        write_json(state_path, dict(updated_at=now(), phase=phase, pid=os.getpid(), crop_day=str(a.crop_day), out=str(out),
                                    done=[d for d in dates if is_done(out, d)], running=sorted(running),
                                    failed=failed, pending=[d for d in dates if not is_done(out, d) and d not in failed
                                                            and d not in running]))

    while True:
        # collect finished processes
        for d, (p, lf) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            ok = finish(out, d, rc)
            del running[d]
            last_progress = time.time()
            if ok:
                log('done', d)
            else:
                tries[d] = tries.get(d, 0) + 1
                log('fail', d, 'rc', rc, 'try', tries[d], '/', a.retries + 1)
                if tries[d] > a.retries:
                    failed[d] = dict(rc=rc, tries=tries[d], at=now())

        # days that can be done
        ready = []
        for d in dates:
            if d in running or d in failed or is_done(out, d) or not (raw / f'{d}.READY').exists():
                continue
            parts, why = day_parts(raw, d, want[d])
            if parts is None:
                if warned.get(d) != why:
                    log('skip', d, why)
                    warned[d] = why
                continue
            ready.append((d, parts))

        if a.dry_run:
            log('dry-run ready', [d for d, _ in ready], 'done', sum(is_done(out, d) for d in dates), 'of', len(dates))
            return

        while ready and len(running) < a.workers:
            d, parts = ready.pop(0)
            running[d] = start(a, out, d, parts)
            log('start', d, 'parts', len(parts))
        save('running' if running else 'waiting')

        complete = (src / 'DOWNLOAD_COMPLETE.json').exists()
        all_done = all(is_done(out, d) or d in failed for d in dates)
        if not running and all_done and (complete or only):
            save('finished')
            log('ALL DONE', 'done', sum(is_done(out, d) for d in dates), 'failed', sorted(failed))
            sys.exit(2 if failed else 0)
        if a.once and not running and not ready:
            save('once-exit')
            log('once: nothing more to do now', 'pending', sum(not is_done(out, d) and d not in failed for d in dates))
            sys.exit(2 if failed else 0)
        if not running and not ready:
            if (src / 'STOP').exists():
                log('warn: downloader STOP file exists, continue waiting (the downloader may be restarted)')
            if time.time() - last_progress > a.max_idle_hours * 3600:
                save('idle-timeout')
                log('idle timeout', a.max_idle_hours, 'h; pending', [d for d in dates if not is_done(out, d) and d not in failed])
                sys.exit(4)
            time.sleep(a.interval)
        else:
            time.sleep(5)


if __name__ == '__main__':
    main()
