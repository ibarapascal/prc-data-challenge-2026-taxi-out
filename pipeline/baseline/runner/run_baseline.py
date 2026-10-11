"""run_baseline: parameterised top-level orchestration that regenerates the baseline predictions from the raw data with the same recipe (only the baseline part, "reproduce the averaged baseline", is implemented).

Principles:
  - No default behaviour of any pipeline script is changed: baseline_models.py / schedule_match_classifier.py are called as they are, and the code state used for the averaged baseline is specified explicitly by environment variables (FEAT_SPLIT=0, LIRF_REFIT=0,
    LOWMEM=0, PI_NESTED unset, ... see BASELINE_ENV); model saving and the CatBoost cache are handled by patching at run time in model_runner.py.
  - Row counts and months are not hard-coded: training files are selected from the input directory by "year + month list"; evaluation months and row counts are derived from ranking / submitting and written to a manifest; if test_months is given, it is asserted to match.
  - Each stage can be run on its own and writes a completion marker <work>/_done/<stage>.json; completed stages are skipped by default (= resume from the middle); --force reruns the given stages.
  - The pipeline directories are read-only: inputs are linked file by file under <work>/raw/ (baseline_models.py writes prediction files to raw/../sub, so they land in <work>/sub/ and never touch the pipeline's sub/).
Usage:
  python run_baseline.py --config configs/ensavg_repro.json --list                 # list stages and completion status
  python run_baseline.py --config ... [--stages a,b | --from A --to B] [--force] [--dry-run]
  Override options: --raw-dir --work --train-year --train-months 1,2,... --test-months 2026-01,2026-07 --threads
Stages (baseline part implemented): inputs -> selftest -> feat -> featdiff -> ensa_lgb -> ensa_cat_taxi -> ensa_cat_delta -> ensa_assemble -> pi -> ensb_lgb -> ensb_cat_taxi
  -> ensb_cat_delta -> ensb_assemble -> ensavg -> compare (ensa = ensemble A, ensb = ensemble B, ensavg = their average); the later ADS-B / residual / track-context / tabular / merge stages are TODO (calling one raises an error).
Final phase: configs/baseline_3seed.json (raw points to final_*, four test months, env_override adds FEAT_SPLIT=1);
  new config key ensavg_check (default true = old behaviour: the ensavg stage does a row-by-row --check against ref.subs.ensavg; false = skipped when the row counts differ, diagnosed only by compare via MVT_ID).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)

# Code state used when the averaged baseline (ensavg) was produced. Listed explicitly so that defaults changed later are not picked up.
BASELINE_ENV = {'FEAT_SPLIT': '0', 'LIRF_REFIT': '0', 'LOWMEM': '0', 'ENS_OBJ': 'mean', 'LIRFSTAND': '1', 'FEATSET': 'sr', 'P01B': 'none', 'P03': 'none'}
BASELINE_UNSET = ('PI_NESTED', 'NFILES', 'FOLDS', 'MEMBERS', 'CATPART', 'REUSE', 'PI', 'NSEED', 'CATSEED', 'PREV', 'FALLBACK', 'ENS_NONA', 'LIRF', 'DROP_COLS')

STAGES = ['inputs', 'selftest', 'feat', 'featdiff', 'ensa_lgb', 'ensa_cat_taxi', 'ensa_cat_delta', 'ensa_assemble', 'pi',
          'ensb_lgb', 'ensb_cat_taxi', 'ensb_cat_delta', 'ensb_assemble', 'ensavg', 'compare']
TODO = {  # stage name -> reference to the planned stage (not implemented)
    'adsb_download': 'ADS-B download', 'adsb_crop': 'ADS-B cropping', 'adsb_feats': 'ADS-B ground features', 'track_diag': 'track-context rows', 'track_table': 'track-context table',
    'oof_base': 'baseline OOF', 'resid_variant_simple': 'residual model (simpler variant)', 'tk_variant_merged_sc': 'residual model with track context and stand correction', 'tab_pf': 'tabular feature store', 'tab_fcv': 'tabular CV', 'tab_fullyear': 'tabular full-year 2026 arms', 'merge_tab': 'merge',
}


def log(*x):
    print(time.strftime('%H:%M:%S'), *x, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class Run:
    def __init__(s, cfg, dry=False):
        s.c = cfg; s.dry = dry; s.W = cfg['work']; s.py = cfg['python']; s.th = str(cfg.get('threads', 6))
        s.raw = os.path.join(s.W, 'raw'); s.models = os.path.join(s.W, 'models'); s.cb = os.path.join(s.W, 'models', 'cb_cache')
        s.rep = os.path.join(s.W, 'report'); s.sub = os.path.join(s.W, 'sub')

    # ---------- Basics ----------
    def done_path(s, st): return os.path.join(s.W, '_done', st + '.json')

    def mark(s, st, t0, **info):
        os.makedirs(os.path.join(s.W, '_done'), exist_ok=True)
        json.dump(dict(stage=st, finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'), sec=round(time.time() - t0), **info),
                  open(s.done_path(st), 'w'), ensure_ascii=False, indent=1)

    def sh(s, st, args, env_add=None, unset=(), cwd=None, fatal=True):
        env = dict(os.environ)
        for k in unset: env.pop(k, None)
        env.update({'POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR': 'load_as_storage', 'PYTHONUNBUFFERED': '1'})
        env.update(env_add or {})
        logf = os.path.join(s.W, 'logs', st + '.log')
        shown = {k: env[k] for k in sorted(env_add or {})}
        log(f'[{st}] $', ' '.join(args), '| env', shown, '| log', logf)
        if s.dry: return
        os.makedirs(os.path.dirname(logf), exist_ok=True)
        with open(logf, 'a') as fh:
            fh.write(f'\n===== {time.strftime("%F %T")} {st}\n$ {" ".join(args)}\nenv {json.dumps(shown, ensure_ascii=False)}\n'); fh.flush()
            rc = subprocess.call(args, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=cwd or s.W)
        if rc != 0:
            tail = open(logf, errors='replace').read().splitlines()[-40:]
            print('\n'.join(tail), flush=True)
            if not fatal:
                log(f'[{st}] diagnostic step failed rc={rc} (does not block the main chain), log {logf}'); return rc
            raise SystemExit(f'[{st}] failed rc={rc}, log {logf}')
        return 0

    def models_env(s, sub, ctx, extra):
        e = dict(BASELINE_ENV); e.update({'RERUN_MODEL_DIR': os.path.join(s.models, sub), 'RERUN_CTX': ctx, 'RERUN_CB_CACHE': s.cb})
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def run_models(s, st, work, stages, tag, sub, ctx, extra):
        # fullens writes raw/../sub even in a partial member stage. Prepare it
        # here as well as on cold starts so resuming a failed stage is safe.
        if 'fullens' in stages and not s.dry:
            os.makedirs(s.sub, exist_ok=True)
        s.sh(st, [s.py, '-u', os.path.join(HERE, 'model_runner.py'), 'basemodels', s.raw, work, *stages, '--threads', s.th, '--tag', tag],
             s.models_env(sub, ctx, extra), unset=BASELINE_UNSET)

    # ---------- Stages ----------
    def st_inputs(s, t0):
        c = s.c; rd = c['raw_dir']; yr = int(c['train_year']); months = [int(m) for m in c['train_months']]
        import glob
        train = []
        for m in months:
            pat = os.path.join(rd, c.get('train_glob', 'training_{year}-{mm}-01_*.parquet').format(year=yr, mm=f'{m:02d}'))
            hit = sorted(glob.glob(pat)); assert len(hit) == 1, (pat, hit); train.append(hit[0])
        rank = os.path.join(rd, c.get('rank_file', 'ranking.parquet')); subm = os.path.join(rd, c.get('submit_file', 'submitting.parquet'))
        assert os.path.exists(rank) and os.path.exists(subm), (rank, subm)
        log('[inputs] train files', len(train), '| rank', rank, '| submit', subm)
        if s.dry: return
        if os.path.isdir(s.raw): shutil.rmtree(s.raw)
        os.makedirs(s.raw)
        for f in train: os.symlink(f, os.path.join(s.raw, os.path.basename(f)))
        os.symlink(rank, os.path.join(s.raw, 'ranking.parquet')); os.symlink(subm, os.path.join(s.raw, 'submitting.parquet'))
        if c.get('ext_dir'):
            met = sorted(glob.glob(os.path.join(c['ext_dir'], 'metar', '*.csv')))
            assert met, f"ext_dir/metar is empty: {c['ext_dir']} (the METAR features of the baseline script would be missing entirely)"
        ext = os.path.join(s.W, 'ext')
        if os.path.islink(ext): os.unlink(ext)
        if c.get('ext_dir'): os.symlink(c['ext_dir'], ext)
        import polars as pl
        r = pl.read_parquet(rank, columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt'])
        dep = r.filter(pl.col('PHASE_mvt') == 'DEP')
        ym = sorted(dep.select(pl.col('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m')).unique().to_series().to_list())
        sid = pl.read_parquet(subm, columns=['MVT_ID_mvt'])['MVT_ID_mvt']
        assert sid.n_unique() == len(sid), 'duplicate MVT_ID in submitting'
        assert sid.is_in(dep['MVT_ID_mvt']).all(), 'submitting has MVT_ID values that are not DEP rows of ranking'
        if c.get('test_months'): assert ym == sorted(c['test_months']), ('ranking DEP months do not match test_months', ym, c['test_months'])
        tr_rows = {os.path.basename(f): pl.scan_parquet(f).select(pl.len()).collect().item() for f in train}
        man = dict(raw_dir=rd, train_year=yr, train_months=months, train_files=tr_rows, rank=rank, rank_rows=r.height, rank_dep_rows=dep.height,
                   test_months=ym, submit=subm, n_submit=len(sid), md5={os.path.basename(p): md5(p) for p in train + [rank, subm]},
                   ext_dir=c.get('ext_dir'), ext_md5={os.path.basename(p): md5(p) for p in sorted(glob.glob(os.path.join(c.get('ext_dir') or '/nonexistent', 'metar', '*.csv')))}, code={f: md5(os.path.join(SRC, f)) for f in ('baseline_models.py', 'schedule_match_classifier.py', 'features_base.py', 'features_extended.py')}
                   | {'runner/' + f: md5(os.path.join(HERE, f)) for f in ('run_baseline.py', 'model_runner.py', 'average_ensembles.py', 'compare_predictions.py')}, python=s.py)
        os.makedirs(s.rep, exist_ok=True); json.dump(man, open(os.path.join(s.rep, 'manifest_inputs.json'), 'w'), ensure_ascii=False, indent=1)
        log('[inputs] rank DEP rows', dep.height, 'submit rows', len(sid), 'test months', ym)
        s.mark('inputs', t0, n_submit=len(sid), test_months=ym)

    def st_selftest(s, t0):
        s.sh('selftest', [s.py, '-u', os.path.join(HERE, 'model_runner.py'), 'selftest'],
             {'RERUN_MODEL_DIR': os.path.join(s.W, 'tmp_selftest', 'm'), 'RERUN_CB_CACHE': os.path.join(s.W, 'tmp_selftest', 'cb'), 'RERUN_CTX': 'selftest'})
        if not s.dry: shutil.rmtree(os.path.join(s.W, 'tmp_selftest'), ignore_errors=True); s.mark('selftest', t0)

    def st_feat(s, t0):
        fw = os.path.join(s.W, 'feat')
        if not s.dry: os.makedirs(fw, exist_ok=True)
        s.run_models('feat', fw, ['feat'], 'feat', 'feat', 'feat', dict(s.c.get('env_override', {})))
        if not s.dry: s.mark('feat', t0, feat=os.path.join(fw, 'feat.parquet'), md5=md5(os.path.join(fw, 'feat.parquet')))

    def st_featdiff(s, t0):
        ref = s.c.get('ref', {}).get('feat')
        if not ref:
            log('[featdiff] no ref.feat, skipped')
            if not s.dry: s.mark('featdiff', t0, skipped=True)
            return
        s.sh('featdiff', [s.py, '-u', os.path.join(HERE, 'compare_predictions.py'), 'feat', '--new', os.path.join(s.W, 'feat', 'feat.parquet'), '--ref', ref,
                          '--out', os.path.join(s.rep, 'featdiff.json')], fatal=False)
        if not s.dry: s.mark('featdiff', t0)

    def setup_arm(s, arm):
        """Prepares <work>/<arm>/ (arm ensa = ensemble A, ensb = ensemble B): feat (A: symlink, B: copy rewritten by the pi stage), results.tsv (round-count metadata), ens_weights.json (member weights)."""
        cfg = s.c[arm]; wd = os.path.join(s.W, arm)
        if s.dry: return wd
        os.makedirs(wd, exist_ok=True)
        fp = os.path.join(wd, 'feat.parquet')
        if arm == 'ensa' and not os.path.exists(fp): os.symlink(os.path.join(s.W, 'feat', 'feat.parquet'), fp)
        if cfg.get('env', {}).get('PI') == '1':
            import pyarrow.parquet as pq
            assert os.path.exists(fp) and 'pi_sched' in pq.read_schema(fp).names, f'{arm} needs a feat with pi_sched (run the pi stage first): {fp}'
        rt = os.path.join(wd, 'results.tsv')
        if not os.path.exists(rt): shutil.copy(s.c['iter_source'], rt)
        json.dump(cfg['weights'], open(os.path.join(wd, 'ens_weights.json'), 'w'))
        for m in cfg.get('reuse_from', {}).get('members', []):
            src = os.path.join(s.W, cfg['reuse_from']['arm'], f'full_{m}.npy'); dst = os.path.join(wd, f'full_{m}.npy')
            assert os.path.exists(src), f'{arm} must reuse {src}; run {cfg["reuse_from"]["arm"]} first'
            if not os.path.exists(dst): shutil.copy(src, dst)
        return wd

    def check_iters(s, wd):
        exp = s.c.get('expect_iters') or {}
        if not exp or s.dry: return
        rows = {}
        for l in open(os.path.join(wd, 'results.tsv')):
            p = l.split('\t'); rows.setdefault(p[0], []).append(json.loads(p[5]))
        for name, (it, itd) in exp.items():
            r = rows.get(name); assert r, f'{name} not in results.tsv'
            got = (int(sum(x.get('it', 0) for x in r) / len(r) * 1.15), int(sum(x.get('itd', 0) for x in r) / len(r) * 1.15))
            assert got == (it, itd), (f'{name} rounds {got} != expected {(it, itd)} (was an experiment of the same name appended to results.tsv?)')
        log('[iters] ok', exp)

    def arm_stage(s, st, t0):
        arm, part = st.split('_', 1); cfg = s.c[arm]; wd = s.setup_arm(arm); tag = cfg['tag']
        base = dict(s.c.get('env_override', {}), **cfg.get('env', {}), REUSE='1')
        if part == 'lgb':
            s.check_iters(wd)
            s.run_models(st, wd, ['fullens'], tag, tag, f'{tag}_lgb', dict(base, MEMBERS=','.join(cfg['lgb_members'])))
            if not s.dry:
                for m in cfg['lgb_members']: assert os.path.exists(os.path.join(wd, f'full_{m}.npy')), f'{st}: full_{m}.npy not generated'
        elif part in ('cat_taxi', 'cat_delta'):
            # The two CatBoost targets run in separate processes (both in one process once reached 10.9 GB); LOWMEM=1 only affects the CatBoost input dtype (CatBoost uses float32 internally anyway); LGB members do not go through here.
            s.run_models(st, wd, ['fullens'], tag, tag, f'{tag}_{part}', dict(base, MEMBERS=cfg['cat_member'], CATPART=part.split('_')[1], LOWMEM='1'))
        elif part == 'assemble':
            s.run_models(st, wd, ['fullens'], tag, tag, f'{tag}_asm', base)
            if not s.dry:
                out = os.path.join(s.sub, f'upstanding-firefly_{tag}.parquet'); assert os.path.exists(out), out
                s.mark(st, t0, out=out, md5=md5(out), members=list(cfg['weights'])); return
        if not s.dry: s.mark(st, t0)

    def st_pi(s, t0):
        wd = os.path.join(s.W, 'ensb')
        if not s.dry:
            os.makedirs(wd, exist_ok=True); fp = os.path.join(wd, 'feat.parquet')
            if not os.path.exists(fp): shutil.copy(os.path.join(s.W, 'feat', 'feat.parquet'), fp)
        s.sh('pi', [s.py, '-u', os.path.join(HERE, 'model_runner.py'), 'pi_sched', wd, '--threads', s.th],
             {'RERUN_MODEL_DIR': os.path.join(s.models, 'pi'), 'RERUN_CTX': 'pi', 'RERUN_CB_CACHE': s.cb}, unset=BASELINE_UNSET)
        if not s.dry: s.mark('pi', t0, md5=md5(os.path.join(wd, 'feat.parquet')))

    def st_ensavg(s, t0):
        a = os.path.join(s.sub, f"upstanding-firefly_{s.c['ensa']['tag']}.parquet"); b = os.path.join(s.sub, f"upstanding-firefly_{s.c['ensb']['tag']}.parquet")
        out = os.path.join(s.sub, f"upstanding-firefly_{s.c.get('ensavg_tag', 'ensavg_rerun')}.parquet")
        args = [s.py, '-u', os.path.join(HERE, 'average_ensembles.py'), '--a', a, '--b', b, '--out', out, '--template', os.path.join(s.raw, 'submitting.parquet')]
        ref = s.c.get('ref', {}).get('subs', {}).get('ensavg')
        if ref and s.c.get('ensavg_check', True): args += ['--check', ref]  # ensavg_check=false: the reference has a different row count from this run (final phase); only compare checks it, by MVT_ID
        s.sh('ensavg', args)
        if not s.dry: s.mark('ensavg', t0, out=out, md5=md5(out))

    def st_compare(s, t0):
        ref = s.c.get('ref', {}); rs = ref.get('subs', {}); pairs = []
        for arm, tag in (('ensa', s.c['ensa']['tag']), ('ensb', s.c['ensb']['tag']), ('ensavg', s.c.get('ensavg_tag', 'ensavg_rerun'))):
            if rs.get(arm): pairs.append(f"{arm}={os.path.join(s.sub, f'upstanding-firefly_{tag}.parquet')}:{rs[arm]}")
        py = [s.py, '-u', os.path.join(HERE, 'compare_predictions.py')]
        if pairs:
            s.sh('compare', py + ['subs', '--pairs', ','.join(pairs), '--keys', os.path.join(s.raw, 'ranking.parquet'), '--out', os.path.join(s.rep, 'compare_subs.json')], fatal=False)
        for arm in ('ensa', 'ensb'):
            mm = ref.get('members', {}).get(arm)
            if mm and ref.get('feat'):
                wd = os.path.join(s.W, arm); tag = s.c[arm]['tag']
                rmap = ','.join(f'{k}={v}' for k, v in mm.items())
                nmap = ','.join(f'{k}={os.path.join(wd, "pred_" + tag + ".npy")}' for k in mm if k.startswith('pred'))
                s.sh('compare', py + ['members', '--new-feat', os.path.join(wd, 'feat.parquet'), '--new-dir', wd, '--ref-feat', ref['feat'],
                                      '--ref-map', rmap] + (['--new-map', nmap] if nmap else []) + ['--out', os.path.join(s.rep, f'compare_members_{arm}.json')], fatal=False)
        if not s.dry:
            summ = {}
            for f in ('compare_subs.json', 'compare_members_ensa.json', 'compare_members_ensb.json', 'featdiff.json'):
                p = os.path.join(s.rep, f)
                if os.path.exists(p):
                    d = json.load(open(p))
                    summ[f] = ({k: {kk: v.get(kk) for kk in ('n_diff', 'max_abs', 'rms', 'mean_diff', 'corr')} for k, v in d.items() if isinstance(v, dict)}
                               if f != 'featdiff.json' else dict(n_cols_neq=d.get('n_cols_neq'), rows=(d.get('rows_new'), d.get('rows_ref')),
                                                                 worst=sorted(((k, v.get('n_neq', 0)) for k, v in d['cols'].items()), key=lambda x: -x[1])[:10]))
            json.dump(summ, open(os.path.join(s.rep, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
            log('[compare] summary', json.dumps(summ.get('compare_subs.json'), ensure_ascii=False))
            s.mark('compare', t0)

    def run(s, stages, force):
        for st in stages:
            if st in TODO: raise SystemExit(f'stage {st} is not implemented yet (TODO {TODO[st]})')
            if os.path.exists(s.done_path(st)) and st not in force:
                log(f'[{st}] already done, skipped ({s.done_path(st)})'); continue
            t0 = time.time(); log(f'===== stage {st}')
            if st.startswith(('ensa_', 'ensb_')): s.arm_stage(st, t0)
            else: getattr(s, 'st_' + st)(t0)
            log(f'===== stage {st} done {time.time() - t0:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--stages'); ap.add_argument('--from', dest='frm'); ap.add_argument('--to')
    ap.add_argument('--force', default='', help='comma-separated: stages to rerun even if completed; all = every selected stage')
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--list', action='store_true')
    ap.add_argument('--raw-dir'); ap.add_argument('--work'); ap.add_argument('--train-year'); ap.add_argument('--train-months'); ap.add_argument('--test-months'); ap.add_argument('--threads')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    for k, v in (('raw_dir', a.raw_dir), ('work', a.work), ('train_year', a.train_year), ('threads', a.threads)):
        if v: cfg[k] = v
    if a.train_months: cfg['train_months'] = [int(x) for x in a.train_months.split(',')]
    if a.test_months: cfg['test_months'] = a.test_months.split(',')
    R = Run(cfg, a.dry_run)
    if a.list:
        for st in STAGES: print(f"{st:14s} {'done' if os.path.exists(R.done_path(st)) else '-'}")
        for st, sec in TODO.items(): print(f'{st:14s} TODO {sec}')
        return
    if a.stages: sel = a.stages.split(',')
    else:
        i = STAGES.index(a.frm) if a.frm else 0; j = STAGES.index(a.to) + 1 if a.to else len(STAGES); sel = STAGES[i:j]
    force = set(sel) if a.force == 'all' else set(x for x in a.force.split(',') if x)
    if not a.dry_run:
        os.makedirs(cfg['work'], exist_ok=True)
        with open(os.path.join(cfg['work'], 'rerun_invocations.log'), 'a') as fh:
            fh.write(f'{time.strftime("%F %T")} {" ".join(sys.argv)}\n')
    log('work', cfg['work'], 'stages', sel, 'force', sorted(force), 'dry' if a.dry_run else '')
    R.run(sel, force)
    log('RERUN DONE', sel)


if __name__ == '__main__':
    main()
