#!/usr/bin/env python3
"""Frozen 20-fold confirmation: train all folds before any outer evaluation."""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.summarize_mvcl_sacti_matched import read, digest, require

SEAL = ROOT/'results/mvcl_formal_confirmation_protocol_v1/seal.json'


def jobs_for(policy, output, python):
    jobs, evaluations = [], []
    for dataset in policy['datasets']:
        for fold in policy['folds']:
            base = output/dataset/('fold_%d'%fold)
            shared = ROOT/'results/mvcl_confirmation_data_v1'/dataset/('fold_%d'%fold)
            data = shared if shared.exists() else base/'data'
            common = ['--dataset',dataset,'--fold',str(fold)]
            stages = []
            if not shared.exists():
                stages.append(('prepare',[python,'tools/prepare_mvcl_confirmation_data.py',*common,'--prepare','--output',str(data)]))
            worker = [python,'tools/train_mvcl_confirmation_fold.py',*common,'--data',str(data),'--device','cuda']
            stages += [('preflight',worker+['--dry-run','--output',str(base/'preflight')]),
                       ('train',worker+['--output',str(base/'training')])]
            for stage, command in stages:
                jobs.append(dict(dataset=dataset,fold=fold,stage=stage,command=command,status='PENDING'))
            evaluations.append(dict(dataset=dataset,fold=fold,stage='evaluate',status='PENDING',
                command=[python,'tools/evaluate_mvcl_confirmation.py','--run',str(base/'training'),
                         '--output',str(base/'evaluation'),'--device','cuda']))
    return jobs+evaluations


def summarize(output, jobs):
    (output/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    rows = []
    for job in jobs:
        if job['stage']=='evaluate' and job['status']=='OK':
            path = output/job['dataset']/('fold_%d'%job['fold'])/'evaluation/summary.json'
            rows.append(dict(dataset=job['dataset'],fold=job['fold'],**read(path)))
    complete = all(j['status']=='OK' for j in jobs)
    aggregates = {}
    keys = ['AUPR','AUC','MRR','Precision@10','Precision@20','Recall@10','Recall@20']
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [r for r in rows if r['dataset']==dataset]
        aggregates[dataset] = {arm:{k:dict(mean=statistics.mean(r['metrics'][arm][k] for r in group),
            sample_sd=statistics.stdev(r['metrics'][arm][k] for r in group) if len(group)>1 else None)
            for k in keys} for arm in ('mvcl','prototype','jaccard','raw_equal')}
    (output/'summary.json').write_text(json.dumps(dict(complete=complete,completed_outer_folds=len(rows),
        folds=rows,aggregates=aggregates,new_lockbox_claim=False,model_selection_reopened=False),indent=2)+'\n')
    lines = ['# MVCL Matched Four-Dataset Confirmation','',
        'Complete: %s; completed outer folds: %d/20.'%(complete,len(rows)), '',
        '| Dataset | Fold | Stage | Status |','|---|---:|---|---|']
    lines += ['| %s | %d | %s | %s |'%(j['dataset'],j['fold'],j['stage'],j['status']) for j in jobs]
    lines += ['', '| Dataset | Method | Folds | AP mean (sample SD) | MRR mean | R@20 mean |',
              '|---|---|---:|---:|---:|---:|']
    for dataset, arms in aggregates.items():
        n = sum(r['dataset']==dataset for r in rows)
        for arm,m in arms.items():
            sd = 'NA' if m['AUPR']['sample_sd'] is None else '%.6f'%m['AUPR']['sample_sd']
            lines.append('| %s | %s | %d | %.6f (%s) | %.6f | %.6f |'%(
                dataset,arm,n,m['AUPR']['mean'],sd,m['MRR']['mean'],m['Recall@20']['mean']))
    lines += ['', 'All 20 training checkpoints are frozen before outer evaluation. No tuning or automatic retry.',
        'All results retained. Partial results are not full confirmation. CV folds are not independent replicates.',
        'Relation-only adaptation; seed901026 was used historically, not a fresh lockbox.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    require(Path.cwd().resolve()==ROOT,'Run from repository root.')
    require(not args.output.exists(),'Use a new output directory; no implicit resume.')
    formal=read(SEAL)
    hashes=dict(formal['source_hashes'])
    files=[SEAL,Path(__file__),ROOT/'tools/train_mvcl_confirmation_fold.py',
        ROOT/'tools/prepare_mvcl_confirmation_data.py',ROOT/'tools/mvcl_confirmation_inputs.py',
        ROOT/'tools/evaluate_mvcl_confirmation.py',ROOT/'tools/evaluate_mvcl_ranking.py',
        ROOT/'tools/run_mvcl_cti_pilot.py',ROOT/'tools/summarize_mvcl_sacti_matched.py',
        ROOT/'util/checkpoint_ranking.py',ROOT/'tools/evaluate_non_neural_cold_start_baselines.py',
        *sorted((ROOT/'util').glob('mvcl*.py'))]
    for f in files:
        hashes.setdefault(str(f.resolve()),digest(f))
    require(all(digest(f)==h for f,h in hashes.items()),'Frozen source changed.')
    jobs=jobs_for(formal['protocol'],args.output,sys.executable)
    if not args.execute:
        for j in jobs:
            print(shlex.join(j['command']))
        print('Plan only: 20 training jobs then 20 outer evaluations. No training launched.')
        return
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(source_hashes=hashes,
        all_training_before_outer=True,new_lockbox_claim=False,formal_seal=str(SEAL)),indent=2)+'\n')
    summarize(args.output,jobs)
    frozen_checkpoints = None
    for job in jobs:
        try:
            require(all(digest(f)==h for f,h in hashes.items()),'Sources changed; stopped.')
            if job['stage']=='evaluate':
                if frozen_checkpoints is None:
                    require(all(j['status']=='OK' for j in jobs if j['stage']!='evaluate'),
                            'All training must finish before outer evaluation.')
                    frozen_checkpoints={}
                    for j in jobs:
                        if j['stage']=='train':
                            base=args.output/j['dataset']/('fold_%d'%j['fold'])/'training'
                            for name in ('best_weights.pt','seal.json','summary.json','confirmation_state.json'):
                                f=base/name; frozen_checkpoints[str(f.resolve())]=digest(f)
                    (args.output/'all_checkpoints_frozen.json').write_text(json.dumps(frozen_checkpoints,indent=2)+'\n')
                require(all(digest(f)==h for f,h in frozen_checkpoints.items()),'Frozen checkpoint changed.')
            log=args.output/('%s_fold%d_%s.log'%(job['dataset'],job['fold'],job['stage']))
            job.update(status='RUNNING',log=str(log));summarize(args.output,jobs)
            print(job['dataset'],job['fold'],job['stage'],'log:',log,flush=True)
            with log.open('x') as stream:
                result=subprocess.run(job['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            job['returncode']=result.returncode
            require(result.returncode==0,'Worker failed: '+str(log))
            require(all(digest(f)==h for f,h in hashes.items()),'Sources changed during worker.')
            if frozen_checkpoints is not None:
                require(all(digest(f)==h for f,h in frozen_checkpoints.items()),'Frozen checkpoint changed during worker.')
            job['status']='OK'
        except BaseException:
            job['status']='FAILED_OR_INTERRUPTED';summarize(args.output,jobs)
            raise
        summarize(args.output,jobs)
    print('Summary:',args.output/'summary.md')


if __name__=='__main__':
    main()
