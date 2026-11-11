#!/usr/bin/env python3
"""Snapshot local 10K metrics and produce manuscript tables; never runs inference."""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
rows=json.loads((a.source/'reports/RESULTS.json').read_text())
rows=[r for r in rows if r['step']==10000 and r['split'] in ('tiny','full')]
assert len(rows)==6
manifest=[]
for r in rows:
    src=Path(r['path']); actual=json.loads(src.read_text())
    assert actual['integrity']['unique']==actual['N']
    assert actual['integrity']['nonempty']==actual['N']
    assert actual['N']==(1000 if r['split']=='tiny' else 7211)
    for k in ('Overall','Low','Middle','High','per_task'):
        assert actual[k]==r[k], (src,k)
    for k in ('empty_answers','truncations','scorer_exceptions','wrong_latents'):
        assert actual.get('integrity',{}).get(k,0)==0
    entry={'arm':r['arm'],'split':r['split'],'step':10000,'metrics':actual,
           'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
    if r['arm']!='B0':
        protocol=json.loads((src.parent/'protocol.json').read_text())
        complete=json.loads((src.parent/'COMPLETE.json').read_text())
        assert protocol['checkpoint'].endswith('/'+r['arm']+'/checkpoint-step10000')
        assert protocol['max_pixels']==230400 and protocol['max_new_tokens']==4096
        assert protocol['seed']==0 and protocol['BF16greedy']
        checkpoint_manifest=Path(protocol['checkpoint'])/'MODEL.json'
        assert hashlib.sha256(checkpoint_manifest.read_bytes()).hexdigest()==protocol['checkpoint_sha256']
        assert complete['status']=='COMPLETE' and complete['count']==actual['N']
        entry['protocol']={k:v for k,v in protocol.items() if k!='checkpoint'}
        entry['checkpoint_label']=r['arm']+'/checkpoint-step10000'
    else:
        entry['audit_limitation']='Historical reuse; complete original source hash manifest unavailable.'
    manifest.append(entry)
(root/'evidence/step10000.json').write_text(json.dumps(manifest,indent=2)+'\n')
def row(arm,split):
    return next(r for r in rows if r['arm']==arm and r['split']==split)
lines=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Model & Overall & Low & Middle & High \\',r'\midrule']
for split in ('tiny','full'):
    lines.append(r'\multicolumn{5}{c}{'+split.title()+r'} \\')
    for arm in ('B0','GENERIC','CGMF'):
        r=row(arm,split)
        lines.append(f'{arm} & '+' & '.join(f'{r[k]:.3f}' for k in ('Overall','Low','Middle','High'))+r' \\')
    if split=='tiny':lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Verified seed0, 10K SPAR-Bench scores. B0 is historical; GENERIC and CGMF additionally use measured depth and calibration. Overall is a task macro average.}',r'\label{tab:main}',r'\end{table}']
(root/'sec/results_main.tex').write_text('\n'.join(lines)+'\n')
# Compact names retain all task distinctions without shrinking official fonts.
names=['Camera motion','Depth OC','Depth OC (MV)','Depth OO','Depth OO (MV)','Center dist. OO','Center dist. OO (MV)','Distance OC','Distance OC (MV)','Distance OO','Distance OO (MV)','Relation OC (MV)','Relation OO','Relation OO (MV)','Position matching','Imagination OC','Imagination OC (MV)','Imagination OO','Imagination OO (MV)','View change']
keys=sorted(row('B0','full')['per_task'])
assert len(keys)==len(names)==20
lines=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{@{}lrrr@{}}',r'\toprule',r'Full task & B0 & GENERIC & CGMF \\',r'\midrule']
for name,k in zip(names,keys):
    lines.append(name+' & '+' & '.join(f"{row(arm,'full')['per_task'][k]:.3f}" for arm in ('B0','GENERIC','CGMF'))+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\caption{All 20 task scores at 10K on Full. OC: object--camera; OO: object--object; MV: multi-view. Values follow the task-specific benchmark scoring rules.}',r'\label{tab:tasks}',r'\end{table}']
(root/'sec/results_tasks.tex').write_text('\n'.join(lines)+'\n')
print('Verified six metrics files and generated two tables.')
