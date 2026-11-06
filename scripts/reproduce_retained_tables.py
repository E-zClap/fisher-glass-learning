"""Verify published hashes and regenerate tables from retained numeric inputs.

This replays numerical reporting, not model fitting, image composition, or gate
discovery. Historical model weights are intentionally excluded from the release.
Run from any directory; outputs default to <repository>/build/reproduced_tables.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = ('landbird/land', 'landbird/water', 'waterbird/land', 'waterbird/water')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=2e-10, abs_tol=2e-12):
        raise AssertionError(f'{label}: {actual} != {expected}')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_hashes():
    manifest = read(ROOT / 'ARTIFACT_MANIFEST.json')
    for entry in manifest['files']:
        data = (ROOT / entry['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['published_sha256']:
            raise AssertionError('Published SHA256 mismatch: ' + entry['path'])
    return len(manifest['files'])


def normal_summary(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return dict(n=len(values), mean=mean, sample_sd=sd,
                nominal_normal_95_halfwidth=1.96 * sd / math.sqrt(len(values)))


def matched_selectors(output):
    base = ROOT / 'artifacts/prospective_v2'
    expected = read(ROOT / 'reanalysis/matched_selectors/results.json')
    selectors = (
        ('min_surviving_fraction', 'projected_over_raw', min),
        ('min_validation_accuracy', 'accuracy', min),
        ('max_validation_nll', 'nll', max),
        ('min_raw_information', 'raw_fisher', min),
        ('min_absolute_projected_information', 'projected_fisher', min),
        ('min_normalized_probability_weighted_fraction', 'projected_over_raw_calibrated', min),
    )
    records = []
    for seed in range(22, 27):
        folder = base / 'runs' / f'erm_lam0_seed{seed}'
        selection, shifts = read(folder / 'selection.json'), read(folder / 'shift_results.json')
        groups = selection['families']['groups']['slices']
        threshold = statistics.median(g['nll'] for g in groups)
        eligible = [g for g in groups if g['accuracy'] >= selection['val_overall_accuracy'] and g['nll'] <= threshold]
        assert [g['name'] for g in eligible] == selection['families']['groups']['rule']['benign_names']
        outcomes = {g['slice']: g for g in shifts['by_group']}
        for pool_name, pool in [('original_eligible_set', eligible), ('all_four_groups', groups)]:
            for name, field, choose in selectors:
                group = choose(pool, key=lambda g: g[field])
                result = outcomes[group['name']]
                drop = 100 * (result['acc_same'] - result['acc_opposite'])
                close(drop, 100 * result['delta_nuisance'], 'stored controlled drop')
                records.append(dict(seed=seed, candidate_set=pool_name, selector=name,
                                    selected_group=group['name'], controlled_drop_pp=drop))
    summaries = []
    for original in expected['selector_summaries']:
        selected = [r for r in records if r['candidate_set'] == original['candidate_set'] and r['selector'] == original['selector']]
        summary = normal_summary([r['controlled_drop_pp'] for r in selected])
        close(summary['mean'], original['controlled_drop_pp']['mean'], 'matched mean')
        close(summary['nominal_normal_95_halfwidth'], original['controlled_drop_pp']['normal_95_halfwidth'], 'matched nominal normal interval')
        counts = dict(collections.Counter(r['selected_group'] for r in selected))
        assert counts == original['selection_counts']
        summaries.append(dict(candidate_set=original['candidate_set'], selector=original['selector'],
                              selection_counts=json.dumps(counts, sort_keys=True), **summary))
    write_csv(output / 'matched_selectors/per_seed.csv', records)
    write_csv(output / 'matched_selectors/summary.csv', summaries)
    return len(records)


def matrix_constructions(output):
    base = ROOT / 'reanalysis/matrix_constructions'
    expected = read(base / 'results.json')
    records, selections = [], []
    for seed in range(22, 27):
        with np.load(base / 'validation_inputs' / f'erm_lam0_seed{seed}.npz', allow_pickle=False) as arrays:
            coords = np.column_stack([arrays['z_val'], arrays['b_val']])
            p = 1 / (1 + np.exp(-arrays['val_logits'] / float(arrays['temperature'])))
            weights = p * (1 - p)
            for original in [r for r in expected['records'] if r['seed'] == seed]:
                mask = arrays['val_group'] == original['gid']
                x, q = coords[mask], weights[mask]
                assert len(x) == original['n']
                mu = x.mean(axis=0)
                close(mu[0], original['mu_z'], 'mean task coordinate')
                close(mu[1], original['mu_b'], 'mean nuisance coordinate')
                matrices = {
                    'uncentered': x.T @ x / len(x),
                    'centered': (x - mu).T @ (x - mu) / len(x),
                    'predictive_fisher': (q[:, None] * x).T @ x / len(x),
                    'normalized_weight_comparator': (q[:, None] * x).T @ x / q.sum(),
                }
                for name, matrix in matrices.items():
                    a, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 1]
                    projected = max(float(a - c * c / (d + 0.001)), 0.0)
                    fraction = projected / a
                    for key, value in [('raw', a), ('cross', c), ('nuisance', d), ('projected', projected), ('fraction', fraction)]:
                        close(value, original['constructions'][name][key], f'{seed}/{name}/{key}')
                    records.append(dict(seed=seed, group=original['group'], construction=name,
                                        n=len(x), mean_z=mu[0], mean_b=mu[1], raw=a, cross=c,
                                        nuisance=d, projected=projected, fraction=fraction,
                                        controlled_drop=original['controlled_drop']))
        for name in matrices:
            chosen = min([r for r in records if r['seed'] == seed and r['construction'] == name], key=lambda r: r['fraction'])
            original_choice = next(r for r in expected['selections'] if r['seed'] == seed and r['construction'] == name)
            assert chosen['group'] == original_choice['group']
            selections.append({k: chosen[k] for k in ('seed', 'construction', 'group', 'fraction', 'controlled_drop')})
    summaries = []
    for name in matrices:
        chosen = [r for r in selections if r['construction'] == name]
        summaries.append(dict(construction=name, selection_counts=json.dumps(dict(collections.Counter(r['group'] for r in chosen)), sort_keys=True),
                              **normal_summary([100 * r['controlled_drop'] for r in chosen])))
    write_csv(output / 'matrix_constructions/matrices.csv', records)
    write_csv(output / 'matrix_constructions/selections.csv', selections)
    write_csv(output / 'matrix_constructions/summary_pp.csv', summaries)
    return len(records)


def recompute_seed_summaries(value, path='', rows=None):
    """Recalculate all retained hidden/group-free Student-t seed summaries."""
    if rows is None:
        rows = []
    if isinstance(value, dict):
        values = value.get('values')
        if isinstance(values, list) and values and all(isinstance(v, (int, float)) for v in values) and 'mean' in value:
            mean = float(np.mean(values)); sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
            se = sd / math.sqrt(len(values))
            confidence = float(value.get('confidence', .95))
            half = float(t.ppf(.5 + confidence/2, len(values)-1)) * se if len(values)>1 else 0.
            close(mean, value['mean'], path + '/mean')
            for key in ('std', 'standard_deviation'):
                if key in value:close(sd, value[key], path+'/'+key)
            for key in ('se', 'standard_error'):
                if key in value:close(se, value[key], path+'/'+key)
            if 'half_width' in value:close(half, value['half_width'], path+'/half_width')
            # Rebuild rendered endpoints, retaining the declared bounded interval convention.
            lo, hi = mean - half, mean + half
            if 'bounds' in value and value['bounds'] is not None:
                lo=max(lo,value['bounds'][0]); hi=min(hi,value['bounds'][1])
            for key,new in [('lo',lo),('hi',hi),('low',lo),('high',hi)]:
                if key in value:close(new,value[key],path+'/'+key);value[key]=new
            value['mean']=mean
            rows.append(dict(metric_path=path,n=len(values),mean=mean,sample_sd=sd,standard_error=se,student_t_95_halfwidth=half,
                             per_seed_values=json.dumps(values)))
        for key, child in value.items():recompute_seed_summaries(child, path+'/'+key, rows)
    elif isinstance(value,list):
        for index,child in enumerate(value):recompute_seed_summaries(child,path+'/'+str(index),rows)
    return rows


def hidden_flat(row):
    """Same columns as experiments/waterbirds_hidden_slice/aggregate.py."""
    flat={k:row[k] for k in ('key','method','selector','nuisance','group_labels_used','group_label_role','role')}
    for metric,s in row['metrics'].items():
        for suffix in ('mean','lo','hi','n'):flat[f'{metric}_{suffix}']='' if s is None else s[suffix]
    for metric in ('score_correlation','top_overlap'):
        s=(row.get('gate_stability') or {}).get(metric)
        for suffix in ('mean','lo','hi'):flat[f'gate_{metric}_{suffix}']='' if s is None else s[suffix]
    c=row.get('composition')
    flat['dominant_metadata_group']=c['dominant_metadata_group'] if c else ''
    flat['dominant_group_purity_mean']=c['dominant_group_purity']['mean'] if c else ''
    for group in GROUP_NAMES:flat['composition_'+group.replace('/','_')+'_mean']=c['weighted_group_composition'][group]['mean'] if c else ''
    b=row.get('accuracy_budget');r=row.get('repair_selection')
    flat.update(accuracy_budget_all_satisfied='' if b is None else b['all_satisfied'],accuracy_budget_satisfied_seeds='' if b is None else b['satisfied_seeds'],repair_no_op_seeds='' if r is None else r['no_op_seeds'],repair_active_seeds='' if r is None else r['active_repair_seeds'])
    return flat


def group_free_flat(rows,frontier=False):
    result=[]
    for row in rows:
        flat={'lambda_fg':row['lambda_fg']} if frontier else dict(policy=row['key'],label=row['label'],source=row['source'],selection_supervision=row['selection_supervision'],selected_lambda_counts=json.dumps(row['lambda_counts'],sort_keys=True))
        for metric,s in row['metrics'].items():
            for name,key in [('mean','mean'),('ci_low','low'),('ci_high','high')]:flat[f'{metric}_{name}']=s[key]
        if frontier:
            for policy,count in row['selected_counts'].items():flat[f'selected_by_{policy}']=count
        result.append(flat)
    return result


def check_csv(rows,path):
    with path.open(encoding='utf-8-sig',newline='') as handle:expected=list(csv.DictReader(handle))
    assert len(rows)==len(expected)
    for actual,old in zip(rows,expected):
        assert set(actual)==set(old)
        for key,value in actual.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):close(value,old[key],str(path)+'/'+key)
            elif str(value)!=old[key]:raise AssertionError(f'{path}/{key}: {value} != {old[key]}')


def seed_tables(output):
    count=0
    for cohort in ('hidden_slice_corrected_v2','group_free_v2'):
        base=ROOT/'artifacts'/cohort
        aggregate=read(base/'aggregate.json')
        summaries=recompute_seed_summaries(aggregate)
        count+=len(summaries)
        write_csv(output/cohort/'seed_summaries.csv',summaries)
        tables={'main_table.csv':[hidden_flat(r) for r in aggregate['main_table']]} if cohort=='hidden_slice_corrected_v2' else {
            'main_table.csv':group_free_flat(aggregate['policies']),
            'frontier.csv':group_free_flat(aggregate['frontier'],frontier=True)}
        for filename,rows in tables.items():
            check_csv(rows,base/filename)
            write_csv(output/cohort/filename,rows)
    return count


def selected_classes(output):
    base=ROOT/'reanalysis/statistical_reporting'
    with (base/'selected_task_classes_per_seed.csv').open(encoding='utf-8-sig',newline='') as handle:
        records=list(csv.DictReader(handle))
    for row in records:
        source=read(ROOT/'artifacts/hidden_slice_corrected_v2/runs'/f"seed_{row['seed']}"/'blind_test_results.json')
        key=row['source_key'].split(' ')[0].split('.')
        current=source
        for part in key:current=current[part]
        assert int(row['class_id'])==current
    write_csv(output/'statistical_reporting/selected_task_classes_per_seed.csv',records)
    return len(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'build/reproduced_tables')
    parser.add_argument('--verify-only',action='store_true',help='Check published SHA256 hashes only; do not write tables')
    args=parser.parse_args()
    count=verify_hashes()
    print(f'Verified {count} published source/data hashes.')
    if args.verify_only:return
    print(f'Reproduced {matched_selectors(args.output_dir)} matched selector decisions.')
    print(f'Reproduced {matrix_constructions(args.output_dir)} matrix/group/seed records.')
    print(f'Recomputed {seed_tables(args.output_dir)} hidden-slice and group-free seed summaries.')
    print(f'Verified {selected_classes(args.output_dir)} selected-class records against per-seed outputs.')
    print(f'Tables written to {args.output_dir}')


if __name__=='__main__':main()
