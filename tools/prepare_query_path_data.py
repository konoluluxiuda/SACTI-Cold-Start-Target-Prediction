#!/usr/bin/env python3
"""Connect frozen inner partitions to path graphs without fitting or scoring."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.audit_triplet_donor_inputs import mapped_pd
from tools.evaluate_non_neural_cold_start_baselines import read_relation_pairs, ordered_ids
from tools.run_donor_reweighting_pilot import load_dataset, dump
from tools.evaluate_checkpoint_ranking import sha256_file
from util.config import ModelConf
from util.query_path_graph import QueryPathGraph


def require_existing_split(config):
    conf = ModelConf(str(config))
    if (conf['experiment.protocol'] != 'strict'
            or conf['split.strategy'] != 'compound_cold_start'
            or conf['split.reuse'].lower() != 'true'
            or conf['evaluation.outer.test'].lower() != 'false'):
        raise ValueError('Expected frozen inner-only compound-cold configuration.')
    split = ROOT / conf['split.dir']
    paths = [Path(config), split / 'manifest.json', split / 'fold_assignments.tsv']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError('Will not create or replace a split: %s' % path)
    manifest = json.loads(paths[1].read_text())
    if manifest['assignments_sha256'] != sha256_file(paths[2]):
        raise ValueError('Strict assignment hash mismatch.')
    for name in ('H_C', 'C_P', 'P_D'):
        source = manifest['sources']['side_relations'][name]
        path = Path(source['path'])
        if sha256_file(path) != source['sha256']:
            raise ValueError('Strict source changed: %s' % name)
        paths.append(path)
    return manifest, {str(path): sha256_file(path) for path in paths}


def validate_training_matrix(matrices, train, validation):
    """Compare exact pairs, not just edge counts, against supplied train records."""
    expected = set()
    for c, p, label in train:
        if float(label) not in (0, 1):
            raise ValueError('Training labels must be binary.')
        if float(label) == 1:
            expected.add((str(c), str(p)))
    if {str(r[0]) for r in train} & {str(r[0]) for r in validation}:
        raise ValueError('Inner compounds overlap.')
    rows, cols = matrices['cp'].nonzero()
    actual = {(str(matrices['compound_ids'][c]), str(matrices['protein_ids'][p]))
              for c, p in zip(rows, cols)}
    if actual != expected:
        raise ValueError('Graph C-P differs from exact inner-training positives.')


def load_query_path_dataset(dataset):
    """Return graph and ID map separately from supervised train/validation records.

    Existing strict loading reads outer records and full CP for partition checks
    and the established entity universe, never as propagation edges or scores.
    This entry currently supports only the frozen first-fold development setup.
    """
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Load from repository root for existing config paths.')
    config = ROOT / dataset['candidate_config']
    manifest, sources = require_existing_split(config)
    matrices, train, validation, metadata = load_dataset(dataset)
    validate_training_matrix(matrices, train, validation)
    pd_path = manifest['sources']['side_relations']['P_D']['path']
    pd, diseases = mapped_pd(matrices['protein_ids'], read_relation_pairs(pd_path))
    if sha256_file(pd_path) != metadata['relation_hashes']['P_D']:
        raise ValueError('P-D does not match frozen side information.')
    graph = QueryPathGraph(matrices['hc'], matrices['cp'], pd)
    ids = dict(compound=matrices['compound_ids'], protein=matrices['protein_ids'],
               disease=diseases)
    # The shared matrix builder uses ordered_ids for herb rows.
    hc_path = manifest['sources']['side_relations']['H_C']['path']
    ids['herb'] = ordered_ids(h for h, _ in read_relation_pairs(hc_path))
    checks = []
    for local_query in matrices['test_compounds']:
        if matrices['cp'][int(local_query)].nnz:
            raise ValueError('Validation query has training C-P support.')
    # Exercise one warm and one cold input, without candidate scoring.
    warm = np.flatnonzero(np.diff(matrices['cp'].indptr))
    queries = ([int(warm[0])] if len(warm) else []) + [int(matrices['test_compounds'][0])]
    for query in queries:
        sample = graph.for_query(query)
        if sample['removed_cp_positives'] != matrices['cp'][query].nnz:
            raise ValueError('Query masking did not remove the complete training row.')
        checks.append(dict(query_local_id=query,
            removed_cp_positives=sample['removed_cp_positives'],
            masked_graph_sha256=sample['masked_graph_sha256']))
    for path, digest in sources.items():
        if sha256_file(path) != digest:
            raise ValueError('Input changed while loading: %s' % path)
    metadata.update(source_hashes=sources, node_counts=graph.node_counts,
        training_positive_edges=int(matrices['cp'].nnz),
        validation_queries=len(matrices['test_compounds']),
        id_order_sha256=hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(),
        sample_mask_checks=checks, optimizer_steps=0, metrics_computed=False,
        outer_test_scored=False)
    return graph, ids, train, validation, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/query_path_data')
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from the repository root for existing config paths.')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / 'configs/sacti_inner_ranking_audit.json'
    entries = json.loads(protocol.read_text())['datasets']
    reports = []
    for dataset in entries:
        print('Loading path inputs: %s' % dataset['label'], flush=True)
        graph, ids, _, _, metadata = load_query_path_dataset(dataset)
        target = args.output / dataset['key']
        target.mkdir()
        dump(target / 'entity_ids.json', ids)
        dump(target / 'provenance.json', metadata)
        reports.append(metadata)
        del graph
    dump(args.output / 'summary.json', dict(reports=reports,
        protocol_sha256=sha256_file(protocol), metrics_computed=False, optimizer_steps=0))
    lines = ['# Query Path Data Integration', '',
        'Frozen first-fold inner inputs only. No training or ranking evaluation.', '',
        '| Dataset | Compounds | Proteins | Training C-P | Validation queries |',
        '|---|---:|---:|---:|---:|']
    for r in reports:
        lines.append('| %s | %d | %d | %d | %d |' % (r['dataset'],
            r['node_counts']['compound'], r['node_counts']['protein'],
            r['training_positive_edges'], r['validation_queries']))
    lines += ['', 'Full C-P / outer records are read by the established loader only for',
              'partition integrity and entity-universe construction, not graph edges or scoring.',
              'This is an input-integrity check, not an algorithm performance Gate.']
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('Summary: %s' % (args.output / 'summary.md'))


if __name__ == '__main__':
    main()
