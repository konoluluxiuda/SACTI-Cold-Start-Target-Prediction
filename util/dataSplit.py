import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from .io import FileIO, NEGATIVE_FILE_CANDIDATES, resolve_optional_dataset_file, sample_negative_records


STRICT_MANIFEST_VERSION = 3
PAIR_STRATIFIED_SPLIT = 'pair_stratified'
COMPOUND_COLD_START_SPLIT = 'compound_cold_start'
SUPPORTED_SPLIT_STRATEGIES = (
    PAIR_STRATIFIED_SPLIT,
    COMPOUND_COLD_START_SPLIT,
)
SIDE_RELATION_FILE_CANDIDATES = {
    'H_C': ('H_C.txt', 'herb-compound.txt', 'HI.txt'),
    'C_P': ('C_P.txt', 'compound-protein.txt', 'IT.txt'),
    'P_D': ('P_D.txt', 'target-disease.txt', 'TD.txt'),
}


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _side_relation_sources(dataset_dir):
    sources = {}
    for relation_name, candidates in SIDE_RELATION_FILE_CANDIDATES.items():
        path_value = resolve_optional_dataset_file(str(dataset_dir), candidates)
        path = Path(path_value).resolve() if path_value else None
        sources[relation_name] = {
            'path': str(path) if path else None,
            'sha256': _sha256(path) if path else None,
        }
    return sources


def _deduplicate(records, positive):
    output = []
    seen = set()
    for left_id, right_id, rating in records:
        is_positive = float(rating) > 0
        if is_positive != positive:
            continue
        pair = (str(left_id), str(right_id))
        if pair in seen:
            continue
        seen.add(pair)
        output.append([pair[0], pair[1], 1.0 if positive else 0.0])
    return output


def _sample_negative_file(conf, path, count, positives, rng):
    reservoir = []
    selected = set()
    eligible_count = 0
    for left_id, right_id, _ in FileIO.iterDataSet(conf, path, default_rating=0.0):
        pair = (str(left_id), str(right_id))
        if pair in positives or pair in selected:
            continue
        eligible_count += 1
        record = [pair[0], pair[1], 0.0]
        if len(reservoir) < count:
            reservoir.append(record)
            selected.add(pair)
            continue
        replacement = rng.randrange(eligible_count)
        if replacement < count:
            previous = reservoir[replacement]
            selected.remove((previous[0], previous[1]))
            reservoir[replacement] = record
            selected.add(pair)
    if len(reservoir) < count:
        raise ValueError(
            'Negative file %s yielded only %d eligible pairs; %d are required.' %
            (path, len(reservoir), count)
        )
    return reservoir, eligible_count


def _write_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    temp_path.write_text(content, encoding='utf-8')
    os.replace(str(temp_path), str(path))


def _records_sha256(records):
    lines = [
        '%s\t%s\t%d' % (str(left_id), str(right_id), int(float(label) > 0))
        for left_id, right_id, label in records
    ]
    return hashlib.sha256(
        ('\n'.join(sorted(lines)) + '\n').encode('utf-8')
    ).hexdigest()


def _pairs_sha256(pairs):
    lines = ['%s\t%s' % (str(left_id), str(right_id)) for left_id, right_id in pairs]
    return hashlib.sha256(
        ('\n'.join(sorted(lines)) + '\n').encode('utf-8')
    ).hexdigest()


def _stable_seed(seed, *parts):
    value = '|'.join([str(int(seed))] + [str(part) for part in parts])
    return int(hashlib.sha256(value.encode('utf-8')).hexdigest()[:16], 16)


def _read_relation_pairs(path):
    pairs = set()
    with open(path, encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError('Invalid relation row %d in %s.' % (line_number, path))
            pairs.add((str(parts[0]), str(parts[1])))
    return pairs


def _context_indices(pairs, entity_position):
    entity_contexts = {}
    context_entities = {}
    for left_id, right_id in pairs:
        entity_id, context_id = (
            (right_id, left_id) if entity_position == 1 else (left_id, right_id)
        )
        entity_contexts.setdefault(entity_id, set()).add(context_id)
        context_entities.setdefault(context_id, set()).add(entity_id)
    return entity_contexts, context_entities


def _rank_shared_context_peers(entity_id, entity_contexts, context_entities, allowed_ids):
    contexts = entity_contexts.get(entity_id, set())
    peers = set()
    for context_id in contexts:
        peers.update(context_entities.get(context_id, set()))
    peers.discard(entity_id)
    peers.intersection_update(allowed_ids)

    def similarity(peer_id):
        peer_contexts = entity_contexts.get(peer_id, set())
        union_size = len(contexts | peer_contexts)
        return 0.0 if union_size == 0 else len(contexts & peer_contexts) / float(union_size)

    return sorted(peers, key=lambda peer_id: (-similarity(peer_id), str(peer_id)))


class DataSplit(object):

    def __init__(self):
        pass

    @staticmethod
    def resolveSplitStrategy(conf):
        strategy = (
            conf['split.strategy'].strip().lower()
            if conf.contains('split.strategy') else PAIR_STRATIFIED_SPLIT
        )
        aliases = {
            'pair': PAIR_STRATIFIED_SPLIT,
            'random_pair': PAIR_STRATIFIED_SPLIT,
            'compound': COMPOUND_COLD_START_SPLIT,
            'compound_coldstart': COMPOUND_COLD_START_SPLIT,
        }
        strategy = aliases.get(strategy, strategy)
        if strategy not in SUPPORTED_SPLIT_STRATEGIES:
            raise ValueError(
                'Unsupported split.strategy %r. Expected one of: %s.' %
                (strategy, ', '.join(SUPPORTED_SPLIT_STRATEGIES))
            )
        return strategy

    @staticmethod
    def crossValidation(data, k, output=True, path='./dataset/TCMsuite', order=1):
        if k <= 1 or k > 10:
            k = 3
        for i in range(k):
            trainingSet = []
            testSet = []
            for ind, line in enumerate(data):
                if ind % k == i:
                    testSet.append(line[:])
                else:
                    trainingSet.append(line[:])

            if output:
                os.makedirs(path, exist_ok=True)
                save_path = os.path.join(path, 'test_fold_%d.txt' % i)
                with open(save_path, 'w', encoding='utf-8') as f:
                    for item in testSet:
                        if isinstance(item, list):
                            f.write('\t'.join(map(str, item)) + '\n')
                        else:
                            f.write(str(item) + '\n')

            yield trainingSet, testSet

    @staticmethod
    def innerValidationSplit(data, ratio, seed):
        ratio = float(ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError('Inner validation ratio must be between 0 and 1.')

        records_by_label = {0: [], 1: []}
        pair_labels = {}
        for left_id, right_id, rating in data:
            pair = (str(left_id), str(right_id))
            label = 1 if float(rating) > 0 else 0
            if pair in pair_labels:
                if pair_labels[pair] != label:
                    raise ValueError('Conflicting labels for inner-validation pair %s.' % (pair,))
                raise ValueError('Duplicate pair in inner-validation input: %s.' % (pair,))
            pair_labels[pair] = label
            records_by_label[label].append([pair[0], pair[1], float(label)])

        if any(len(records) < 2 for records in records_by_label.values()):
            raise ValueError('Inner validation requires at least two positive and two negative records.')

        inner_train = []
        validation = []
        class_counts = {}
        for label in (0, 1):
            records = sorted(records_by_label[label], key=lambda row: (row[0], row[1]))
            random.Random(int(seed) + label).shuffle(records)
            validation_count = int(round(len(records) * ratio))
            validation_count = max(1, min(len(records) - 1, validation_count))
            validation.extend(records[:validation_count])
            inner_train.extend(records[validation_count:])
            class_counts[str(label)] = {
                'total': len(records),
                'inner_train': len(records) - validation_count,
                'validation': validation_count,
            }

        random.Random(int(seed) + 2).shuffle(inner_train)
        random.Random(int(seed) + 3).shuffle(validation)
        train_pairs = {(row[0], row[1]) for row in inner_train}
        validation_pairs = {(row[0], row[1]) for row in validation}
        if train_pairs & validation_pairs:
            raise ValueError('Inner train and validation pairs overlap.')

        assignment_lines = [
            '%s\t%s\t%d\t%s' % (row[0], row[1], int(row[2]), partition)
            for partition, records in (('train', inner_train), ('validation', validation))
            for row in records
        ]
        assignment_hash = hashlib.sha256(
            ('\n'.join(sorted(assignment_lines)) + '\n').encode('utf-8')
        ).hexdigest()
        info = {
            'seed': int(seed),
            'ratio': ratio,
            'inner_train_records': len(inner_train),
            'validation_records': len(validation),
            'class_counts': class_counts,
            'assignments_sha256': assignment_hash,
        }
        return inner_train, validation, info

    @staticmethod
    def innerCompoundValidationSplit(data, ratio, seed):
        ratio = float(ratio)
        if not 0.0 < ratio < 1.0:
            raise ValueError('Inner validation ratio must be between 0 and 1.')

        records_by_compound = {}
        pair_labels = {}
        for left_id, right_id, rating in data:
            pair = (str(left_id), str(right_id))
            label = 1 if float(rating) > 0 else 0
            if pair in pair_labels:
                if pair_labels[pair] != label:
                    raise ValueError('Conflicting labels for inner-validation pair %s.' % (pair,))
                raise ValueError('Duplicate pair in inner-validation input: %s.' % (pair,))
            pair_labels[pair] = label
            records_by_compound.setdefault(pair[0], []).append(
                [pair[0], pair[1], float(label)]
            )

        compound_ids = sorted(records_by_compound)
        if len(compound_ids) < 2:
            raise ValueError(
                'Compound cold-start inner validation requires at least two compounds.'
            )
        validation_compound_count = int(round(len(compound_ids) * ratio))
        validation_compound_count = max(
            1, min(len(compound_ids) - 1, validation_compound_count)
        )
        compound_ids.sort(
            key=lambda compound_id: (
                _stable_seed(seed, 'inner_validation', compound_id),
                str(compound_id),
            )
        )
        validation_compounds = set(compound_ids[:validation_compound_count])

        inner_train = []
        validation = []
        for compound_id in sorted(records_by_compound):
            target = validation if compound_id in validation_compounds else inner_train
            target.extend(
                sorted(records_by_compound[compound_id], key=lambda row: (row[0], row[1]))
            )
        random.Random(_stable_seed(seed, 'inner_train_records')).shuffle(inner_train)
        random.Random(_stable_seed(seed, 'validation_records')).shuffle(validation)

        train_compounds = {row[0] for row in inner_train}
        validation_compound_ids = {row[0] for row in validation}
        if train_compounds & validation_compound_ids:
            raise ValueError('Inner train and validation compounds overlap.')

        def class_counts(records):
            return {
                '0': sum(float(row[2]) <= 0 for row in records),
                '1': sum(float(row[2]) > 0 for row in records),
            }

        train_class_counts = class_counts(inner_train)
        validation_class_counts = class_counts(validation)
        if 0 in train_class_counts.values() or 0 in validation_class_counts.values():
            raise ValueError(
                'Compound cold-start inner validation requires both labels in '
                'the inner-train and validation partitions.'
            )

        assignment_lines = [
            '%s\t%s\t%d\t%s' % (row[0], row[1], int(row[2]), partition)
            for partition, records in (('train', inner_train), ('validation', validation))
            for row in records
        ]
        assignment_hash = hashlib.sha256(
            ('\n'.join(sorted(assignment_lines)) + '\n').encode('utf-8')
        ).hexdigest()
        info = {
            'strategy': COMPOUND_COLD_START_SPLIT,
            'seed': int(seed),
            'ratio': ratio,
            'inner_train_records': len(inner_train),
            'validation_records': len(validation),
            'inner_train_compounds': len(train_compounds),
            'validation_compounds': len(validation_compound_ids),
            'class_counts': {
                'inner_train': train_class_counts,
                'validation': validation_class_counts,
            },
            'assignments_sha256': assignment_hash,
        }
        return inner_train, validation, info

    @staticmethod
    def innerValidationSplitForConfig(conf, data, ratio, seed):
        strategy = DataSplit.resolveSplitStrategy(conf)
        if strategy == COMPOUND_COLD_START_SPLIT:
            return DataSplit.innerCompoundValidationSplit(data, ratio, seed)
        inner_train, validation, info = DataSplit.innerValidationSplit(data, ratio, seed)
        info['strategy'] = PAIR_STRATIFIED_SPLIT
        return inner_train, validation, info

    @staticmethod
    def applyTrainingNegativeStrategy(
            data,
            settings,
            dataset_dir,
            reserved_pairs=None,
            seed=202026,
            fold_index=0,
            manifest_dir=None):
        strategy = settings['strategy']
        hard_ratio = float(settings['hard_ratio'])
        if strategy == 'random':
            return [row[:] for row in data], {
                'strategy': 'random',
                'seed': int(seed),
                'hard_ratio': 0.0,
                'records_sha256': _records_sha256(data),
            }
        if strategy != 'mixed':
            raise ValueError('Unsupported training negative strategy: %s.' % strategy)

        dataset_dir = Path(dataset_dir).resolve()
        hc_path_value = resolve_optional_dataset_file(
            str(dataset_dir), SIDE_RELATION_FILE_CANDIDATES['H_C']
        )
        pd_path_value = resolve_optional_dataset_file(
            str(dataset_dir), SIDE_RELATION_FILE_CANDIDATES['P_D']
        )
        if not hc_path_value or not pd_path_value:
            raise FileNotFoundError('Mixed negative sampling requires H-C and P-D relation files.')
        hc_path = Path(hc_path_value).resolve()
        pd_path = Path(pd_path_value).resolve()

        positives = sorted(
            _deduplicate(data, positive=True), key=lambda row: (row[0], row[1])
        )
        random_negatives = sorted(
            _deduplicate(data, positive=False), key=lambda row: (row[0], row[1])
        )
        if not positives or not random_negatives:
            raise ValueError('Mixed negative sampling requires positive and random negative records.')

        positive_pairs = {(row[0], row[1]) for row in positives}
        original_negative_pairs = {(row[0], row[1]) for row in random_negatives}
        reserved_pairs = {
            (str(left_id), str(right_id)) for left_id, right_id in (reserved_pairs or set())
        }
        blocked_pairs = positive_pairs | original_negative_pairs | reserved_pairs
        allowed_left_ids = {str(row[0]) for row in data}
        allowed_right_ids = {str(row[1]) for row in data}

        compound_contexts, herb_compounds = _context_indices(
            _read_relation_pairs(hc_path), entity_position=1
        )
        protein_contexts, disease_proteins = _context_indices(
            _read_relation_pairs(pd_path), entity_position=0
        )

        rng = random.Random(int(seed))
        ordered_positives = sorted(positives, key=lambda row: (row[0], row[1]))
        rng.shuffle(ordered_positives)
        desired_hard_count = int(round(len(random_negatives) * hard_ratio))
        hard_records = []
        hard_pairs = set()
        hard_pair_sources = {}
        source_counts = {'H_C': 0, 'P_D': 0}
        compound_peer_cache = {}
        protein_peer_cache = {}

        def compound_peers(compound_id):
            if compound_id not in compound_peer_cache:
                compound_peer_cache[compound_id] = _rank_shared_context_peers(
                    compound_id, compound_contexts, herb_compounds, allowed_left_ids
                )
            return compound_peer_cache[compound_id]

        def protein_peers(protein_id):
            if protein_id not in protein_peer_cache:
                protein_peer_cache[protein_id] = _rank_shared_context_peers(
                    protein_id, protein_contexts, disease_proteins, allowed_right_ids
                )
            return protein_peer_cache[protein_id]

        for index, (compound_id, protein_id, _) in enumerate(ordered_positives):
            if len(hard_records) >= desired_hard_count:
                break
            directions = ('P_D', 'H_C') if index % 2 == 0 else ('H_C', 'P_D')
            selected_pair = None
            selected_source = None
            for source in directions:
                if source == 'P_D':
                    candidates = ((compound_id, peer_id) for peer_id in protein_peers(protein_id))
                else:
                    candidates = ((peer_id, protein_id) for peer_id in compound_peers(compound_id))
                for pair in candidates:
                    if pair in blocked_pairs or pair in hard_pairs:
                        continue
                    selected_pair = pair
                    selected_source = source
                    break
                if selected_pair is not None:
                    break
            if selected_pair is None:
                continue
            hard_pairs.add(selected_pair)
            hard_pair_sources[selected_pair] = selected_source
            hard_records.append([selected_pair[0], selected_pair[1], 0.0])
            source_counts[selected_source] += 1

        ordered_random_negatives = sorted(
            random_negatives, key=lambda row: (row[0], row[1])
        )
        rng.shuffle(ordered_random_negatives)
        retained_random_count = len(random_negatives) - len(hard_records)
        retained_random = ordered_random_negatives[:retained_random_count]
        transformed = positives + retained_random + hard_records
        rng.shuffle(transformed)

        assignment_rows = []
        assignment_rows.extend((row[0], row[1], 1, 'positive') for row in positives)
        assignment_rows.extend((row[0], row[1], 0, 'random') for row in retained_random)
        hard_sources = {}
        for source, count in source_counts.items():
            if count:
                hard_sources[source] = count
        for row in hard_records:
            source = hard_pair_sources[(row[0], row[1])]
            assignment_rows.append((row[0], row[1], 0, 'hard_' + source.lower()))
        assignment_content = 'left_id\tright_id\tlabel\tnegative_type\n' + ''.join(
            '%s\t%s\t%d\t%s\n' % row for row in sorted(assignment_rows)
        )
        assignment_hash = hashlib.sha256(assignment_content.encode('utf-8')).hexdigest()
        info = {
            'version': 1,
            'strategy': 'mixed',
            'seed': int(seed),
            'fold': int(fold_index),
            'hard_ratio_requested': hard_ratio,
            'hard_ratio_actual': len(hard_records) / float(len(random_negatives)),
            'positive_count': len(positives),
            'negative_count': len(random_negatives),
            'random_negative_count': len(retained_random),
            'hard_negative_count': len(hard_records),
            'hard_source_counts': hard_sources,
            'input_records_sha256': _records_sha256(data),
            'reserved_pairs_sha256': _pairs_sha256(reserved_pairs),
            'assignments_sha256': assignment_hash,
            'sources': {
                'H_C': {'path': str(hc_path), 'sha256': _sha256(hc_path)},
                'P_D': {'path': str(pd_path), 'sha256': _sha256(pd_path)},
            },
        }
        if manifest_dir is not None:
            output_dir = Path(manifest_dir) / (
                'mixed_seed_%d_ratio_%s' % (int(seed), str(hard_ratio).replace('.', 'p'))
            )
            assignments_path = output_dir / ('fold_%d.tsv' % int(fold_index))
            manifest_path = output_dir / ('fold_%d.json' % int(fold_index))
            _write_atomic(assignments_path, assignment_content)
            info['assignments_path'] = str(assignments_path)
            _write_atomic(manifest_path, json.dumps(info, ensure_ascii=False, indent=2) + '\n')
            info['manifest_path'] = str(manifest_path)
        return transformed, info

    @staticmethod
    def _sampleCompoundMatchedNegatives(conf, negative_path, positive_records, seed):
        positives_by_compound = {}
        positive_pairs = set()
        protein_ids = set()
        for compound_id, protein_id, _ in positive_records:
            compound_id = str(compound_id)
            protein_id = str(protein_id)
            positives_by_compound.setdefault(compound_id, set()).add(protein_id)
            positive_pairs.add((compound_id, protein_id))
            protein_ids.add(protein_id)

        file_candidates = {}
        file_eligible_count = 0
        if negative_path:
            for compound_id, protein_id, _ in FileIO.iterDataSet(
                    conf, str(negative_path), default_rating=0.0):
                compound_id = str(compound_id)
                protein_id = str(protein_id)
                if compound_id not in positives_by_compound:
                    continue
                if protein_id not in protein_ids:
                    continue
                pair = (compound_id, protein_id)
                if pair in positive_pairs:
                    continue
                candidates = file_candidates.setdefault(compound_id, set())
                if protein_id in candidates:
                    continue
                candidates.add(protein_id)
                file_eligible_count += 1

        all_proteins = sorted(protein_ids)
        negative_records = []
        fallback_compounds = 0
        fallback_records = 0
        cartesian_candidate_count = 0
        for compound_id in sorted(positives_by_compound):
            positive_proteins = positives_by_compound[compound_id]
            required = len(positive_proteins)
            cartesian_candidates = [
                protein_id for protein_id in all_proteins
                if protein_id not in positive_proteins
            ]
            cartesian_candidate_count += len(cartesian_candidates)
            if len(cartesian_candidates) < required:
                raise ValueError(
                    'Compound %s has only %d unobserved protein candidates, but %d '
                    'matched negatives are required.' %
                    (compound_id, len(cartesian_candidates), required)
                )

            rng = random.Random(_stable_seed(seed, 'compound_negative', compound_id))
            from_file = sorted(file_candidates.get(compound_id, set()))
            rng.shuffle(from_file)
            selected = from_file[:required]
            if len(selected) < required:
                fallback_compounds += 1
                selected_set = set(selected)
                fallback = [
                    protein_id for protein_id in cartesian_candidates
                    if protein_id not in selected_set
                ]
                rng.shuffle(fallback)
                missing = required - len(selected)
                selected.extend(fallback[:missing])
                fallback_records += missing
            negative_records.extend(
                [compound_id, protein_id, 0.0] for protein_id in selected
            )

        if negative_path is None:
            negative_mode = 'compound_matched_cartesian'
        elif fallback_records:
            negative_mode = 'compound_matched_file_with_cartesian_fallback'
        else:
            negative_mode = 'compound_matched_file'
        audit = {
            'negative_mode': negative_mode,
            'negative_file_eligible_count': file_eligible_count,
            'negative_cartesian_candidate_count': cartesian_candidate_count,
            'negative_fallback_compounds': fallback_compounds,
            'negative_fallback_records': fallback_records,
        }
        return negative_records, audit

    @staticmethod
    def _assignCompoundColdStartFolds(positive_records, negative_records, k, seed):
        records_by_compound = {}
        positive_counts = {}
        for record in positive_records + negative_records:
            compound_id = str(record[0])
            records_by_compound.setdefault(compound_id, []).append(record[:])
            if float(record[2]) > 0:
                positive_counts[compound_id] = positive_counts.get(compound_id, 0) + 1

        compound_ids = list(records_by_compound)
        if len(compound_ids) < k:
            raise ValueError(
                'Compound cold-start split requires at least %d compounds; found %d.' %
                (k, len(compound_ids))
            )
        compound_ids.sort(
            key=lambda compound_id: (
                -positive_counts.get(compound_id, 0),
                _stable_seed(seed, 'outer_fold', compound_id),
                str(compound_id),
            )
        )

        fold_compounds = [[] for _ in range(k)]
        fold_positive_loads = [0 for _ in range(k)]
        for compound_id in compound_ids:
            fold_index = min(
                range(k),
                key=lambda index: (
                    fold_positive_loads[index],
                    len(fold_compounds[index]),
                    index,
                ),
            )
            fold_compounds[fold_index].append(compound_id)
            fold_positive_loads[fold_index] += positive_counts.get(compound_id, 0)

        fold_records = []
        for fold_index, compounds in enumerate(fold_compounds):
            records = []
            for compound_id in sorted(compounds):
                records.extend(
                    sorted(
                        records_by_compound[compound_id],
                        key=lambda row: (row[0], row[1], -float(row[2])),
                    )
                )
            random.Random(_stable_seed(seed, 'outer_records', fold_index)).shuffle(records)
            fold_records.append(records)
        return fold_records

    @staticmethod
    def prepareStrictFolds(conf, datapath, k):
        if k <= 1 or k > 10:
            raise ValueError('Strict cross-validation requires k between 2 and 10.')

        seed = (
            int(conf['split.seed'])
            if conf.contains('split.seed')
            else int(conf['random.seed'])
            if conf.contains('random.seed')
            else 2026
        )
        split_strategy = DataSplit.resolveSplitStrategy(conf)
        dataset_path = Path(datapath).resolve()
        dataset_dir = dataset_path.parent
        if conf.contains('split.dir'):
            split_dir = Path(conf['split.dir']).resolve()
        elif split_strategy == COMPOUND_COLD_START_SPLIT:
            split_dir = dataset_dir / 'splits' / (
                'strict_compound_cold_start_seed_%d_k%d' % (seed, k)
            )
        else:
            split_dir = dataset_dir / 'splits' / ('strict_seed_%d_k%d' % (seed, k))
        reuse = _as_bool(conf['split.reuse'], True) if conf.contains('split.reuse') else True
        manifest_path = split_dir / 'manifest.json'
        assignments_path = split_dir / 'fold_assignments.tsv'

        negative_path_value = resolve_optional_dataset_file(str(dataset_dir), NEGATIVE_FILE_CANDIDATES)
        negative_path = Path(negative_path_value).resolve() if negative_path_value else None
        expected_sources = {
            'datapath': str(dataset_path),
            'datapath_sha256': _sha256(dataset_path),
            'negative_path': str(negative_path) if negative_path else None,
            'negative_sha256': _sha256(negative_path) if negative_path else None,
            'side_relations': _side_relation_sources(dataset_dir),
        }

        if reuse and manifest_path.exists() and assignments_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            DataSplit._validateStrictManifest(
                manifest,
                expected_sources,
                seed,
                k,
                assignments_path,
                split_strategy,
            )
            folds = DataSplit._loadStrictAssignments(assignments_path, k, split_strategy)
            print('Reusing strict split manifest: %s' % manifest_path)
            return folds, manifest

        positive_records = _deduplicate(FileIO.readDataSet(conf, str(dataset_path)), positive=True)
        if not positive_records:
            raise ValueError('Strict protocol found no positive records in %s.' % dataset_path)
        positive_pairs = {(row[0], row[1]) for row in positive_records}
        rng = random.Random(seed)

        if split_strategy == COMPOUND_COLD_START_SPLIT:
            negative_records, negative_audit = DataSplit._sampleCompoundMatchedNegatives(
                conf, negative_path, positive_records, seed
            )
            negative_mode = negative_audit['negative_mode']
            candidate_count = negative_audit['negative_cartesian_candidate_count']
            fold_records = DataSplit._assignCompoundColdStartFolds(
                positive_records, negative_records, k, seed
            )
            split_algorithm = 'compound_group_greedy_balance_v1'
        elif negative_path:
            negative_records, candidate_count = _sample_negative_file(
                conf, str(negative_path), len(positive_records), positive_pairs, rng
            )
            negative_mode = 'sampled_file'
        else:
            negative_records = sample_negative_records(
                positive_records, len(positive_records), rng=rng, deterministic=True
            )
            candidate_count = (
                len({row[0] for row in positive_records})
                * len({row[1] for row in positive_records})
                - len(positive_pairs)
            )
            negative_mode = 'generated_cartesian'
        if split_strategy == PAIR_STRATIFIED_SPLIT:
            rng.shuffle(positive_records)
            rng.shuffle(negative_records)
            fold_records = [[] for _ in range(k)]
            for index, record in enumerate(positive_records):
                fold_records[index % k].append(record)
            for index, record in enumerate(negative_records):
                fold_records[index % k].append(record)
            for fold_index, records in enumerate(fold_records):
                random.Random(seed + fold_index + 1).shuffle(records)
            split_algorithm = 'class_stratified_round_robin_v2'
            negative_audit = {}

        split_dir.mkdir(parents=True, exist_ok=True)
        assignment_lines = ['left_id\tright_id\tlabel\tfold\n']
        fold_stats = []
        for fold_index, records in enumerate(fold_records):
            positive_count = sum(float(row[2]) > 0 for row in records)
            negative_count = len(records) - positive_count
            fold_stats.append({
                'fold': fold_index,
                'test_records': len(records),
                'test_positives': positive_count,
                'test_negatives': negative_count,
                'test_compounds': len({str(row[0]) for row in records}),
            })
            test_lines = []
            for left_id, right_id, rating in records:
                label = 1 if float(rating) > 0 else 0
                assignment_lines.append('%s\t%s\t%d\t%d\n' % (left_id, right_id, label, fold_index))
                test_lines.append('%s\t%s\t%d\n' % (left_id, right_id, label))
            _write_atomic(split_dir / ('test_fold_%d.txt' % fold_index), ''.join(test_lines))
        _write_atomic(assignments_path, ''.join(assignment_lines))

        manifest = {
            'version': STRICT_MANIFEST_VERSION,
            'protocol': 'strict',
            'split_strategy': split_strategy,
            'split_algorithm': split_algorithm,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'seed': seed,
            'folds': k,
            'sources': expected_sources,
            'negative_mode': negative_mode,
            'negative_candidate_count': candidate_count,
            'positive_count': len(positive_records),
            'negative_count': len(negative_records),
            'assignments_path': str(assignments_path),
            'assignments_sha256': _sha256(assignments_path),
            'fold_stats': fold_stats,
            'strict_guarantees': {
                'fixed_negative_sample': True,
                'fixed_stratified_folds': split_strategy == PAIR_STRATIFIED_SPLIT,
                'fixed_group_folds': split_strategy == COMPOUND_COLD_START_SPLIT,
                'pair_disjoint_train_test': True,
                'compound_disjoint_train_test': (
                    split_strategy == COMPOUND_COLD_START_SPLIT
                ),
                'training_graph_must_use_fold_training_positives': True,
                'fixed_hd_side_information': False,
            },
        }
        manifest.update(negative_audit)
        _write_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        folds = DataSplit._loadStrictAssignments(assignments_path, k, split_strategy)
        print('Created strict split manifest: %s' % manifest_path)
        return folds, manifest

    @staticmethod
    def _validateStrictManifest(
            manifest, expected_sources, seed, k, assignments_path, split_strategy):
        errors = []
        if manifest.get('version') != STRICT_MANIFEST_VERSION:
            errors.append('manifest version')
        if manifest.get('protocol') != 'strict':
            errors.append('protocol')
        if manifest.get('seed') != seed:
            errors.append('seed')
        if manifest.get('folds') != k:
            errors.append('folds')
        manifest_strategy = manifest.get('split_strategy', PAIR_STRATIFIED_SPLIT)
        if manifest_strategy != split_strategy:
            errors.append('split strategy')
        if manifest.get('sources') != expected_sources:
            errors.append('source files or hashes')
        if manifest.get('assignments_path') != str(assignments_path):
            errors.append('assignments path')
        if manifest.get('assignments_sha256') != _sha256(assignments_path):
            errors.append('assignments hash')
        if errors:
            raise ValueError(
                'Existing strict manifest does not match %s. Use a new split.dir or set split.reuse=False.' %
                ', '.join(errors)
            )

    @staticmethod
    def _loadStrictAssignments(path, k, split_strategy=PAIR_STRATIFIED_SPLIT):
        fold_records = [[] for _ in range(k)]
        all_records = []
        with open(path, encoding='utf-8') as handle:
            header = next(handle, '').strip().split('\t')
            if header != ['left_id', 'right_id', 'label', 'fold']:
                raise ValueError('Invalid strict assignment header in %s.' % path)
            for line_number, line in enumerate(handle, start=2):
                parts = line.rstrip('\n').split('\t')
                if len(parts) != 4:
                    raise ValueError('Invalid strict assignment row %d in %s.' % (line_number, path))
                left_id, right_id, label, fold_value = parts
                fold_index = int(fold_value)
                if fold_index < 0 or fold_index >= k:
                    raise ValueError('Invalid fold %d in %s.' % (fold_index, path))
                record = [left_id, right_id, float(label)]
                fold_records[fold_index].append(record)
                all_records.append((fold_index, record))

        folds = []
        for fold_index in range(k):
            test = [record[:] for record in fold_records[fold_index]]
            train = [record[:] for assigned_fold, record in all_records if assigned_fold != fold_index]
            train_pairs = {(row[0], row[1]) for row in train}
            test_pairs = {(row[0], row[1]) for row in test}
            if train_pairs & test_pairs:
                raise ValueError('Strict split contains train/test pair overlap in fold %d.' % fold_index)
            if split_strategy == COMPOUND_COLD_START_SPLIT:
                train_compounds = {row[0] for row in train}
                test_compounds = {row[0] for row in test}
                if train_compounds & test_compounds:
                    raise ValueError(
                        'Compound cold-start split contains train/test compound overlap '
                        'in fold %d.' % fold_index
                    )
                compound_label_counts = {}
                for row in test:
                    counts = compound_label_counts.setdefault(row[0], [0, 0])
                    counts[1 if float(row[2]) > 0 else 0] += 1
                for compound_id, (negative_count, positive_count) in (
                        compound_label_counts.items()):
                    if positive_count != negative_count:
                        raise ValueError(
                            'Compound cold-start split requires matched positive/negative '
                            'counts for compound %s in fold %d.' %
                            (compound_id, fold_index)
                        )
            folds.append((train, test))
        return folds
