#!/usr/bin/env python3
"""Verify existing prediction artifacts and generate CVPR tables without inference.

The legacy prepare_evidence.py preserves the earlier 10K-only snapshot. This
script supports the complete through-30K results and portable offline rebuilding.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = (10000, 15000, 20000, 25000, 30000)
ARMS = ('B0', 'GENERIC', 'CGMF')
GROUPS = json.loads((ROOT / 'evidence/task_groups.json').read_text())
TASKS = [t for group in GROUPS.values() for t in group]

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)

def verify(source):
    report = source / 'reports/all_nodes_20261114'
    source_records = read(report / 'RESULTS.json')
    source_hashes = read(report / 'SOURCES.json')
    records = [r for r in source_records if
               (r['step'] in STEPS and r['split'] in ('tiny', 'full') and r['condition'] == 'correct') or
               (r['arm'] == 'CGMF' and r['step'] == 25000 and r['split'] == 'dev')]
    assert len(records) == 31
    out, identities, prediction_maps = [], {}, {}
    for row in records:
        metrics_path = Path(row['path'])
        metrics = read(metrics_path)
        prediction_path = metrics_path.with_name('predictions.jsonl')
        predictions = [json.loads(line) for line in prediction_path.read_text().splitlines() if line.strip()]
        expected_n = {'tiny': 1000, 'full': 7211, 'dev': 600}[row['split']]
        assert len(predictions) == len({p['uid'] for p in predictions}) == metrics['N'] == expected_n
        per_task = defaultdict(list)
        for p in predictions:
            assert math.isfinite(p['score']) and 0 <= p['score'] <= 1
            assert p['latent_count'] == 8 and not p['truncated'] and p['scorer_error'] is None
            per_task[p['task']].append(p['score'])
        assert set(per_task) == set(TASKS)
        recomputed = {t: sum(v) / len(v) * 100 for t, v in per_task.items()}
        for t, v in recomputed.items():
            assert close(v, metrics['per_task'][t]) and close(v, row['per_task'][t])
        for name, tasks in {'Overall': TASKS, **GROUPS, 'PoseScore': GROUPS['Middle'], 'MetricScore': GROUPS['Low']}.items():
            value = sum(recomputed[t] for t in tasks) / len(tasks)
            assert close(value, metrics[name]) and close(value, row[name]), (row['arm'], row['step'], name)
        nonempty = sum(bool(p['prediction'].strip()) for p in predictions)
        assert nonempty == metrics['integrity']['nonempty']
        recorded_hash = source_hashes[str(metrics_path)]
        assert sha(metrics_path) == recorded_hash['metrics_sha256']
        assert sha(prediction_path) == recorded_hash['predictions_sha256']
        identity = {p['uid']: (p['task'], p['site'], p['question_sha256'], p['answer']) for p in predictions}
        if row['split'] in identities:
            assert identity == identities[row['split']], 'Unmatched questions across arms/checkpoints'
        else:
            identities[row['split']] = identity
        key = (row['arm'], row['step'], row['split'], row['condition'])
        prediction_maps[key] = {p['uid']: p for p in predictions}
        entry = {k: row[k] for k in ('seed', 'arm', 'step', 'split', 'condition', 'source')}
        entry.update(metrics=metrics, artifact_label=str(metrics_path.relative_to(source)),
                     metrics_sha256=sha(metrics_path), predictions_sha256=sha(prediction_path),
                     empty_answers=expected_n - nonempty)
        if row['source'] != 'historical_B0_reused':
            protocol = read(metrics_path.with_name('protocol.json'))
            complete = read(metrics_path.with_name('COMPLETE.json'))
            assert complete['status'] == 'COMPLETE' and complete['count'] == expected_n
            assert protocol == complete['protocol']
            assert protocol['split'] == row['split'] and protocol['condition'] == row['condition']
            assert protocol['max_pixels'] == (50176 if row['split'] == 'dev' else 230400)
            assert protocol['max_new_tokens'] == 4096 and protocol['seed'] == 0 and protocol['BF16greedy']
            checkpoint = Path(protocol['checkpoint'])
            assert checkpoint.name.endswith(str(row['step'])) and checkpoint.parent.name == row['arm']
            model = checkpoint / 'MODEL.json'
            assert sha(model) == protocol['checkpoint_sha256']
            model_info = read(model)
            assert model_info['step'] == row['step'] and model_info['arm'] == row['arm']
            entry['checkpoint_label'] = str(checkpoint.relative_to(source))
            entry['protocol'] = {k: v for k, v in protocol.items() if k != 'checkpoint'}
        else:
            entry['provenance_limit'] = 'Historical B0 reuse; complete original source-code hash manifest unavailable.'
        out.append(entry)
    assert not any(r['arm'] == 'B0' and r['step'] == 20000 for r in out)
    paired = read(report / 'PAIRED_STATISTICS.json')
    # Check point estimates against original sample scores; retain supplied bootstrap intervals verbatim.
    for split in ('tiny', 'full'):
        for step in STEPS:
            for comparison, a, b in [('CGMF-B0', 'CGMF', 'B0'), ('GENERIC-B0', 'GENERIC', 'B0'), ('CGMF-GENERIC', 'CGMF', 'GENERIC')]:
                key = f'seed0/step{step}/{split}/{comparison}'
                if key not in paired:
                    continue
                left = prediction_maps[(a, step, split, 'correct')]
                right = prediction_maps[(b, step, split, 'correct')]
                assert left.keys() == right.keys()
                for metric, tasks in [('Overall', TASKS), ('PoseScore', GROUPS['Middle'])]:
                    means = [sum((left[u]['score'] - right[u]['score']) for u in left if left[u]['task'] == t) /
                             sum(left[u]['task'] == t for u in left) * 100 for t in tasks]
                    assert close(sum(means) / len(means), paired[key][metric]['difference_pp'])
    # Verify that interventions use a fixed, same-view-count, different-scene donor mapping.
    donor_map = read(source / 'data/dev_donors.json')
    intervention_keys = [f'seed0/step25000/dev/correct-{condition}' for condition in ('wrong_pose', 'wrong_geometry')]
    for condition, key in zip(('wrong_pose', 'wrong_geometry'), intervention_keys):
        left = prediction_maps[('CGMF', 25000, 'dev', 'correct')]
        right = prediction_maps[('CGMF', 25000, 'dev', condition)]
        assert left.keys() == right.keys() and set(left).issubset(donor_map)
        for uid in left:
            donor = left[donor_map[uid]]
            assert donor['site'] != left[uid]['site']
            assert donor['image_count'] == left[uid]['image_count']
        for metric, tasks in [('Overall', TASKS), ('PoseScore', GROUPS['Middle'])]:
            d = sum(sum(left[u]['score'] - right[u]['score'] for u in left if left[u]['task'] == t) /
                    sum(left[u]['task'] == t for u in left) * 100 for t in tasks) / len(tasks)
            assert close(d, paired[key][metric]['difference_pp'])
    snapshot = {'as_of': '2026-11-15', 'records': out, 'paired_statistics': paired,
                'verification': {'records_verified': len(out), 'raw_scores_reaggregated': True,
                    'original_scorer_rerun': False, 'bootstrap_intervals_recomputed': False,
                    'checkpoint_manifests_rehashed': True, 'checkpoint_tensor_files_rehashed': False,
                    'matched_sample_ids_and_questions': True,
                    'missing_B0_step20000': True,
                    'source_report_sha256': sha(report / 'RESULTS.json'),
                    'paired_statistics_sha256': sha(report / 'PAIRED_STATISTICS.json'),
                    'continuation_protocol_sha256': sha(source / 'continuation_step30000/protocol/CONTINUATION.json'),
                    'donor_mapping_sha256': sha(source / 'data/dev_donors.json')}}
    (ROOT / 'evidence/current_verified.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    return snapshot

def tables(snapshot):
    records = snapshot['records']
    def get(arm, step, split, condition='correct'):
        return next((r['metrics'] for r in records if (r['arm'], r['step'], r['split'], r['condition']) == (arm, step, split, condition)), None)
    def table(columns, header, rows, caption, label):
        return '\n'.join([r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{@{}'+columns+r'@{}}',
                          r'\toprule', header+r' \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}',
                          r'\caption{'+caption+'}', r'\label{'+label+'}', r'\end{table}', ''])
    rows = []
    for step in STEPS:
        if rows: rows.append(r'\midrule')
        for arm in ARMS:
            values = [get(arm, step, split) for split in ('tiny', 'full')]
            rows.append(f'{step//1000}K & {arm} & '+ ' & '.join('--' if v is None else f"{v['Overall']:.3f}" for v in values)+r' \\')
    (ROOT/'sec/results_main.tex').write_text(table('llrr', 'Steps & Model & Tiny & Full', rows,
        'SPAR-Bench Overall scores at all evaluated checkpoints, seed0. B0 uses RGB; GENERIC and CGMF additionally use measured depth and calibration. A dash denotes an unevaluated checkpoint, not zero. All three arms continue from 25K to 30K.', 'tab:main'))
    rows=[]
    for split in ('tiny', 'full'):
        rows.append(r'\multicolumn{5}{c}{'+split.title()+', 30K'+r'} \\')
        for arm in ARMS:
            m=get(arm,30000,split)
            rows.append(arm+' & '+' & '.join(f'{m[k]:.3f}' for k in ('Overall','Low','Middle','High'))+r' \\')
        if split=='tiny': rows.append(r'\midrule')
    (ROOT/'sec/results_levels.tex').write_text(table('lrrrr', 'Model & Overall & Low & Middle & High',rows,
        'Reasoning-level scores at the terminal 30K checkpoint. Overall averages 20 task scores; Low, Middle and High average 8, 3 and 9 tasks, respectively.', 'tab:levels'))
    names=['Camera motion','Depth OC','Depth OC (MV)','Depth OO','Depth OO (MV)','Center dist. OO','Center dist. OO (MV)',
           'Distance OC','Distance OC (MV)','Distance OO','Distance OO (MV)','Relation OC (MV)','Relation OO','Relation OO (MV)',
           'Position matching','Imagination OC','Imagination OC (MV)','Imagination OO','Imagination OO (MV)','View change']
    tasks=sorted(TASKS)
    rows=[name+' & '+' & '.join(f"{get(arm,30000,'full')['per_task'][task]:.3f}" for arm in ARMS)+r' \\' for name,task in zip(names,tasks)]
    (ROOT/'sec/results_tasks.tex').write_text(table('lrrr','Full task & B0 & GENERIC & CGMF',rows,
        'All 20 Full task scores at 30K. OC: object--camera; OO: object--object; MV: multi-view. Values use the task-specific benchmark scoring rules.', 'tab:tasks'))
    rows=[]
    for condition, label in [('correct','Correct inputs'),('wrong_pose','Wrong pose'),('wrong_geometry','Wrong geometry')]:
        m=get('CGMF',25000,'dev',condition)
        rows.append(label+' & '+' & '.join(f'{m[k]:.3f}' for k in ('Overall','PoseScore'))+r' \\')
    (ROOT/'sec/results_interventions.tex').write_text(table('lrr','CGMF condition & Overall & PoseScore',rows,
        'Paired input substitutions on 600 development samples at 25K. PoseScore averages the three Middle tasks. The development image budget differs from Tiny/Full, so these scores are not directly comparable to test scores.', 'tab:interventions'))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, help='Original experiment root; omit to regenerate from the checked-in portable snapshot.')
    args=parser.parse_args()
    snapshot=verify(args.source) if args.source else read(ROOT/'evidence/current_verified.json')
    tables(snapshot)
    print(f"Verified snapshot: {len(snapshot['records'])} records; regenerated four tables.")
