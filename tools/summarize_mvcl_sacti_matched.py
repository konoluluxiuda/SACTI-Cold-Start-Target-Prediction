#!/usr/bin/env python3
"""Merge frozen inner reports after provenance checks; no model inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_identity(mvcl, sacti, bridge):
    p, s = mvcl['protocol'], sacti['split']
    require(not p['outer_test_scored'] and not s['outer_test_scored'], 'Outer scoring detected.')
    require(p['training_optimizer_steps'] == sacti['training_optimizer_steps'] == 0,
            'Expected pure-inference reports.')
    require(sacti['evaluation_type'] == 'checkpoint_only_pure_inference_validation',
            'Expected full SACTI validation report.')
    require(sacti['config']['model_name'] == 'SACTI', 'Not the complete SACTI model.')
    for key, old in [('train_sha256', 'ranking_train_sha256'),
                     ('validation_sha256', 'ranking_test_sha256')]:
        require(p[key] == s[old] == bridge[key], 'Split mismatch: '+key)
    require(p['candidate_ids_sha256'] == bridge['candidate_ids_sha256'], 'Candidate mismatch.')
    require(bridge['matched_entities'] and bridge['matched_relations']
            and bridge['checkpoint_unchanged'] and not bridge['outer_test_scored'],
            'Invalid matched representation bridge.')
    require(sacti['fixed_candidate_protocol']['tie_break'] == 'protein_id_ascending',
            'Unexpected tie policy.')
    a, b = mvcl['metrics']['mvcl'], sacti['fixed_candidate_metrics']
    for key in ('evaluated_compounds', 'held_out_positive_pairs', 'candidate_pairs',
                'candidate_count_per_compound', 'aggregation'):
        require(a[key] == b[key], 'Ranking scope mismatch: '+key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    folder = ROOT/'results/batch_runs/mvcl_cti_tcmsuite_ranking_292_v1'
    bridge_path = ROOT/'results/sacti_matched_representations_tcmsuite/summary.json'
    bridge = read(bridge_path)
    sacti_path = Path(bridge['source_report'])
    mvcl, sacti = read(folder/'summary.json'), read(sacti_path)
    check_identity(mvcl, sacti, bridge)
    sources = [folder/'summary.json', bridge_path, sacti_path]
    for path, expected in mvcl['protocol']['source_hashes'].items():
        require(digest(ROOT/path) == expected, 'MVCL source changed: '+path)
    for path, expected in bridge['checkpoint_hashes'].items():
        require(digest(path) == expected, 'SACTI checkpoint changed: '+path)
    checkpoint = sacti['checkpoint']
    require(bridge['checkpoint_hashes'][checkpoint['prefix']+'.index'] == checkpoint['index_sha256'],
            'Bridge uses another checkpoint.')
    for name, expected in checkpoint['data_shards'].items():
        require(bridge['checkpoint_hashes'][str(Path(checkpoint['prefix']).parent/name)] == expected,
                'Bridge checkpoint shard mismatch.')
    ids_path = bridge_path.parent/'ids.json'
    require(digest(ids_path) == bridge['ids_sha256'], 'Bridge entity IDs changed.')
    data = ROOT/'results/mvcl_cti_data_tcmsuite_pilot_v2'
    require(read(ids_path) == read(data/'ids.json'), 'Actual entity IDs differ.')
    # The source bridge established equality; do not infer it from counts alone.
    validation = read(data/'validation.json')
    from tools.run_mvcl_cti_pilot import record_hash
    require(record_hash(validation) == mvcl['protocol']['validation_sha256'], 'Validation changed.')
    positives = Counter(str(c) for c, _, y in validation if y == 1)
    table_path = sacti_path.parent/'per_compound_metrics.tsv'
    with table_path.open() as f:
        per_query = list(csv.DictReader(f, delimiter='\t'))
    require(len(per_query) == len(positives) == len({r['compound_id'] for r in per_query}),
            'Query count/duplicates differ.')
    require({r['compound_id']: int(r['test_positive_count']) for r in per_query} == dict(positives),
            'Query positive counts differ.')
    require(all(int(r['filtered_train_positives']) == 0 and int(r['candidate_count']) == 7258
                for r in per_query), 'Unexpected cold-query filtering.')
    for column, metric in [('reciprocal_rank', 'MRR'), ('recall@20', 'Recall@20'),
                           ('precision@20', 'Precision@20')]:
        value = sum(float(r[column]) for r in per_query)/len(per_query)
        require(abs(value-sacti['fixed_candidate_metrics'][metric]) < 1e-12,
                'Historical aggregate inconsistent: '+metric)
    mvcl_rank = read(folder/'mvcl.json')
    require(mvcl_rank['protocol']['tie_break'] == 'protein_id_ascending', 'MVCL tie policy differs.')
    metrics = dict(mvcl['metrics'])
    sampled = sacti['sampled_pair_metrics_for_comparison_only']
    metrics['sacti_full'] = dict(sacti['fixed_candidate_metrics'], AUPR=sampled['AUPR'], AUC=sampled['AUC'])
    sources += [ids_path, data/'ids.json', data/'validation.json', table_path, folder/'mvcl.json']
    result = dict(metrics=metrics, matched=True, evaluation_split='inner_validation',
                  training_optimizer_steps=0, new_model_forward_calls=0, outer_test_scored=False,
                  source_hashes={str(p): digest(p) for p in sources},
                  boundary='Matched evaluation, not matched training budgets or independent confirmation.',
                  legacy_fields='SACTI outer_train/test protocol names are generic legacy labels; '
                                'actual inner record hashes and zero-filter query rows were checked.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Matched TCM-Suite Inner Comparison', '',
             '| Method | Sampled AP | Sampled AUC | MRR | Macro R@20 | Top20 hits |',
             '|---|---:|---:|---:|---:|---:|']
    for name in ('sacti_full', 'mvcl', 'prototype', 'jaccard', 'raw_equal'):
        m = metrics[name]
        hits = round(m['MicroRecall@20']*m['held_out_positive_pairs'])
        lines.append('| %s | %.6f | %.6f | %.6f | %.6f | %d |' % (
            name, m['AUPR'], m['AUC'], m['MRR'], m['Recall@20'], hits))
    lines += ['', '95 queries, 7258 candidates per query, 4538 held-out known positives.',
              'Source SACTI is the full frozen model, not the representation probe head.',
              result['boundary'], result['legacy_fields'],
              'No training, new inference, threshold choice, or outer scoring. All methods retained.']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Summary:', args.output/'summary.md')


if __name__ == '__main__':
    main()
