# Derived from https://github.com/tong87-bio/HDCTI (MIT License).
# Modified for reproducible strict split orchestration and inner validation.
from util.config import OptionConf
from util.dataSplit import *
from util.support_complete_split import (
    build_four_state_inner_validation,
    build_support_state_inner_validation,
    load_four_state_support_artifact,
    load_support_complete_unit,
    read_pairs,
)
from multiprocessing import Process, Manager
from util.io import FileIO
from time import strftime, localtime, time
import os
import json
from pathlib import Path
# import pandas as pd
import numpy as np
import mkl

from util.reproducibility import set_global_seed
from util.model_components import (
    requires_inner_validation,
    resolve_early_stopping,
    resolve_negative_sampling,
)
from util.training_controls import (
    permute_training_labels,
    resolve_label_permutation,
)


class HDR(object):
    def __init__(self, config):
        self.trainingData = []  # training data
        self.testData = []  # testData
        self.measure = []
        self.config = config
        self.protocol = config['experiment.protocol'].strip().lower() if config.contains('experiment.protocol') else 'legacy'
        self.splitStrategy = DataSplit.resolveSplitStrategy(config)
        self.strictFolds = None
        self.strictManifest = None
        self.supportUnitMetadata = None
        self.supportValidationData = []
        self.supportValidationDataByState = {}
        self.supportTestDataByState = {}
        self.supportInnerValidationMetadata = None
        self.ratingConfig = OptionConf(config['ratings.setup'])
        self.earlyStopping = resolve_early_stopping(config)
        self.requiresInnerValidation = requires_inner_validation(
            config, self.earlyStopping
        )
        self.negativeSampling = resolve_negative_sampling(config)
        self.labelPermutation = resolve_label_permutation(config)
        if self.requiresInnerValidation and self.protocol != 'strict':
            raise ValueError(
                'Inner-validation model selection currently requires '
                'experiment.protocol=strict.'
            )
        if self.negativeSampling['strategy'] != 'random' and self.protocol != 'strict':
            raise ValueError('Mixed negative sampling requires experiment.protocol=strict.')
        if self.config.contains('evaluation.setup'):
            self.evaluation = OptionConf(config['evaluation.setup'])
            if self.protocol == 'strict':
                has_cv = self.evaluation.contains('-cv')
                has_support_unit = self.evaluation.contains('-support-unit')
                has_four_state_unit = self.evaluation.contains(
                    '-four-state-unit'
                )
                if sum((
                        has_cv, has_support_unit, has_four_state_unit)) != 1:
                    raise ValueError(
                        'Strict protocol requires exactly one of '
                        'evaluation.setup=-cv K, -support-unit, or '
                        '-four-state-unit.'
                    )
                if has_four_state_unit:
                    self._loadFourStateSupportUnit()
                elif has_support_unit:
                    self._loadSupportUnit()
                else:
                    k = int(self.evaluation['-cv'])
                    self.strictFolds, self.strictManifest = DataSplit.prepareStrictFolds(
                        config, config['datapath'], k
                    )
                    print(
                        'Strict split: strategy=%s seed=%d' %
                        (self.strictManifest.get('split_strategy', 'pair_stratified'),
                         self.strictManifest['seed'])
                    )
            elif self.protocol == 'legacy':
                self.trainingData = FileIO.loadDataSet(config, config['datapath'])
            else:
                raise ValueError('Unknown experiment.protocol: %s' % self.protocol)
        else:
            print('Wrong configuration of evaluation!')
            exit(-1)

        print('Reading data and preprocessing...')

    def _loadSupportUnit(self):
        if self.negativeSampling['strategy'] != 'random':
            raise ValueError(
                'Support-unit manifests already freeze training negatives; '
                'negative.strategy must be random.'
            )
        required = ('support.manifest', 'support.mode')
        missing = [key for key in required if not self.config.contains(key)]
        if missing:
            raise ValueError(
                'Support-unit configuration is missing: %s.' %
                ', '.join(missing)
            )
        manifest_path = Path(self.config['support.manifest']).expanduser().resolve()
        mode = self.config['support.mode'].strip().lower()
        kwargs = {}
        if mode == 'target_cold':
            if not self.config.contains('support.target.fold'):
                raise ValueError(
                    'target_cold support unit requires support.target.fold.'
                )
            kwargs['fold'] = int(self.config['support.target.fold'])
        elif mode == 'double_cold':
            double_keys = (
                'support.compound.group',
                'support.protein.group',
            )
            missing = [
                key for key in double_keys if not self.config.contains(key)
            ]
            if missing:
                raise ValueError(
                    'double_cold support unit is missing: %s.' %
                    ', '.join(missing)
                )
            kwargs['compound_group'] = int(
                self.config['support.compound.group']
            )
            kwargs['protein_group'] = int(
                self.config['support.protein.group']
            )
        else:
            raise ValueError(
                'support.mode must be target_cold or double_cold.'
            )

        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        config_dataset_dir = Path(
            self.config['datapath']
        ).expanduser().resolve().parent
        manifest_dataset_dir = Path(
            manifest['sources']['C_P']['path']
        ).resolve().parent
        if config_dataset_dir != manifest_dataset_dir:
            raise ValueError(
                'support.manifest dataset does not match datapath directory.'
            )
        self.trainingData, self.testData, self.supportUnitMetadata = (
            load_support_complete_unit(
                manifest_path,
                mode,
                **kwargs
            )
        )
        self.strictManifest = manifest
        self.splitStrategy = 'support_complete_' + mode
        print(
            'Strict support unit: mode=%s unit=%s seed=%d '
            'train=%d (+%d/-%d) test=%d (+%d/-%d).' % (
                mode,
                self.supportUnitMetadata['unit_key'],
                self.supportUnitMetadata['seed'],
                len(self.trainingData),
                self.supportUnitMetadata['training_positive_count'],
                self.supportUnitMetadata['training_negative_count'],
                len(self.testData),
                self.supportUnitMetadata['test_positive_count'],
                self.supportUnitMetadata['test_negative_count'],
            )
        )
        if self.requiresInnerValidation:
            base_seed = (
                int(self.config['random.seed'])
                if self.config.contains('random.seed') else 2026
            )
            validation_seed = (
                int(self.config['validation.seed'])
                if self.config.contains('validation.seed')
                else base_seed + 100000
            )
            all_positive_pairs = read_pairs(
                manifest['sources']['C_P']['path']
            )
            (
                self.trainingData,
                self.supportValidationData,
                self.supportInnerValidationMetadata,
            ) = build_support_state_inner_validation(
                self.trainingData,
                all_positive_pairs,
                mode,
                self.earlyStopping['ratio'],
                validation_seed,
                self.supportUnitMetadata['unit_key'],
            )
            print(
                'Support-state inner validation: strategy=%s seed=%d '
                'train=%d (+%d/-%d) validation=%d (+%d/-%d) '
                'heldout compounds=%d proteins=%d buffer positives=%d '
                'hash=%s.' % (
                    self.supportInnerValidationMetadata['strategy'],
                    validation_seed,
                    len(self.trainingData),
                    self.supportInnerValidationMetadata[
                        'inner_train_positive_count'
                    ],
                    self.supportInnerValidationMetadata[
                        'inner_train_negative_count'
                    ],
                    len(self.supportValidationData),
                    self.supportInnerValidationMetadata[
                        'validation_positive_count'
                    ],
                    self.supportInnerValidationMetadata[
                        'validation_negative_count'
                    ],
                    self.supportInnerValidationMetadata[
                        'heldout_compounds'
                    ],
                    self.supportInnerValidationMetadata[
                        'heldout_proteins'
                    ],
                    self.supportInnerValidationMetadata[
                        'discarded_buffer_positive_count'
                    ],
                    self.supportInnerValidationMetadata[
                        'assignments_sha256'
                    ][:12],
                )
            )


    def _loadFourStateSupportUnit(self):
        if self.negativeSampling['strategy'] != 'random':
            raise ValueError(
                'Four-state manifests already freeze training negatives; '
                'negative.strategy must be random.'
            )
        if not self.config.contains('support.four.state.manifest'):
            raise ValueError(
                'Four-state configuration requires '
                'support.four.state.manifest.'
            )
        if not self.earlyStopping['enabled']:
            raise ValueError(
                'Four-state model selection requires early.stopping=True.'
            )

        manifest_path = Path(
            self.config['support.four.state.manifest']
        ).expanduser().resolve()
        artifact_manifest = json.loads(
            manifest_path.read_text(encoding='utf-8')
        )
        source_manifest_path = Path(
            artifact_manifest['source_manifest']['path']
        ).expanduser().resolve()
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding='utf-8')
        )
        config_dataset_dir = Path(
            self.config['datapath']
        ).expanduser().resolve().parent
        manifest_dataset_dir = Path(
            source_manifest['sources']['C_P']['path']
        ).resolve().parent
        if config_dataset_dir != manifest_dataset_dir:
            raise ValueError(
                'Four-state manifest dataset does not match datapath '
                'directory.'
            )

        (
            self.trainingData,
            self.supportTestDataByState,
            self.supportUnitMetadata,
        ) = load_four_state_support_artifact(manifest_path)
        state_order = (
            'warm_warm', 'cold_warm', 'warm_cold', 'cold_cold'
        )
        self.testData = [
            row
            for state in state_order
            for row in self.supportTestDataByState[state]
        ]
        self.strictManifest = source_manifest
        self.splitStrategy = 'support_complete_four_state'

        base_seed = (
            int(self.config['random.seed'])
            if self.config.contains('random.seed') else 2026
        )
        validation_seed = (
            int(self.config['validation.seed'])
            if self.config.contains('validation.seed')
            else base_seed + 100000
        )
        all_positive_pairs = read_pairs(
            source_manifest['sources']['C_P']['path']
        )
        (
            self.trainingData,
            self.supportValidationDataByState,
            self.supportInnerValidationMetadata,
        ) = build_four_state_inner_validation(
            self.trainingData,
            all_positive_pairs,
            self.earlyStopping['ratio'],
            validation_seed,
            self.supportUnitMetadata['unit_key'],
        )
        self.supportValidationData = [
            row
            for state in state_order
            for row in self.supportValidationDataByState[state]
        ]
        state_counts = ', '.join(
            '%s=+%d/-%d' % (
                state,
                self.supportInnerValidationMetadata[
                    'states'
                ][state]['positive_count'],
                self.supportInnerValidationMetadata[
                    'states'
                ][state]['negative_count'],
            )
            for state in state_order
        )
        print(
            'Strict four-state unit: unit=%s seed=%d '
            'inner train=%d (+%d/-%d); validation %s; '
            'discarded buffer positives=%d; hash=%s.' % (
                self.supportUnitMetadata['unit_key'],
                validation_seed,
                len(self.trainingData),
                self.supportInnerValidationMetadata[
                    'inner_train_positive_count'
                ],
                self.supportInnerValidationMetadata[
                    'inner_train_negative_count'
                ],
                state_counts,
                self.supportInnerValidationMetadata[
                    'discarded_buffer_positive_count'
                ],
                self.supportInnerValidationMetadata[
                    'assignments_sha256'
                ][:12],
            )
        )


    def execute(self):
        # import the model module
        importStr = 'from ' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(importStr)
        if self.evaluation.contains('-four-state-unit'):
            unit_key = self.supportUnitMetadata['unit_key']
            recommender = self.config['model.name'] + (
                "(self.config,self.trainingData,self.testData,'[1]')"
            )
            algorithm = eval(recommender)
            algorithm.validationData = self.supportValidationData
            algorithm.validationDataByState = (
                self.supportValidationDataByState
            )
            algorithm.validationAggregation = 'macro_support_states'
            seed = (
                int(self.config['random.seed'])
                if self.config.contains('random.seed') else 2026
            )
            set_global_seed(seed, reset_tensorflow_graph=True)
            print(
                'Four-state unit random seed: %d; unit: %s.' %
                (seed, unit_key)
            )
            self.measure = algorithm.execute()
            if not self.measure:
                raise ValueError(
                    'Four-state experiment returned no metrics.'
                )
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = OptionConf(self.config['output.setup'])['-dir']
            variant = (
                self.config['model.variant']
                if self.config.contains('model.variant')
                else self.config['model.name']
            )
            fileName = (
                variant + '@' + currentTime + '-four-state-unit-' +
                unit_key + '.txt'
            )
            result_lines = [
                value if value.endswith('\n') else value + '\n'
                for value in self.measure
            ]
            FileIO.writeFile(outDir, fileName, result_lines)
            print(
                'Four-state result (%s):\n%s' %
                (unit_key, ''.join(result_lines))
            )
            return self.measure
        if self.evaluation.contains('-support-unit'):
            unit_key = self.supportUnitMetadata['unit_key']
            recommender = self.config['model.name'] + (
                "(self.config,self.trainingData,self.testData,'[1]')"
            )
            algorithm = eval(recommender)
            algorithm.validationData = self.supportValidationData
            seed = (
                int(self.config['random.seed'])
                if self.config.contains('random.seed') else 2026
            )
            set_global_seed(seed, reset_tensorflow_graph=True)
            print(
                'Support unit random seed: %d; unit: %s.' %
                (seed, unit_key)
            )
            self.measure = algorithm.execute()
            if not self.measure:
                raise ValueError(
                    'Support-unit experiment returned no metrics.'
                )
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = OptionConf(self.config['output.setup'])['-dir']
            variant = (
                self.config['model.variant']
                if self.config.contains('model.variant')
                else self.config['model.name']
            )
            fileName = (
                variant + '@' + currentTime + '-support-unit-' +
                unit_key + '.txt'
            )
            result_lines = [
                value if value.endswith('\n') else value + '\n'
                for value in self.measure
            ]
            FileIO.writeFile(outDir, fileName, result_lines)
            print(
                'Support-unit result (%s):\n%s' %
                (unit_key, ''.join(result_lines))
            )
            return self.measure
        if self.evaluation.contains('-cv'):
            k = int(self.evaluation['-cv'])
            if k < 2 or k > 10:  # limit to 2-10 fold cross validation
                print("k for cross-validation should not be greater than 10 or less than 2")
                exit(-1)
            if self.config.contains('evaluation.fold.indices'):
                selected_folds = sorted({
                    int(value.strip())
                    for value in self.config['evaluation.fold.indices'].split(',')
                    if value.strip()
                })
                if not selected_folds or any(
                        value < 1 or value > k for value in selected_folds):
                    raise ValueError(
                        'evaluation.fold.indices must contain fold IDs in [1, %d].' % k
                    )
            else:
                fold_limit = (
                    int(self.config['evaluation.fold.limit'])
                    if self.config.contains('evaluation.fold.limit') else k
                )
                if fold_limit < 1 or fold_limit > k:
                    raise ValueError(
                        'evaluation.fold.limit must be between 1 and %d.' % k
                    )
                selected_folds = list(range(1, fold_limit + 1))
            fold_limit = len(selected_folds)
            mkl.set_num_threads(max(1, mkl.get_max_threads() // k))
            use_multiprocessing = True
            if self.config.contains('gpu.multiprocessing'):
                use_multiprocessing = self.config['gpu.multiprocessing'].lower() in ('1', 'true', 'yes', 'on')
            if self.protocol == 'strict' and use_multiprocessing:
                print('Strict protocol currently runs folds serially to preserve deterministic TensorFlow state.')
                use_multiprocessing = False
            # CUDA/TensorFlow is fragile after fork in WSL. Keep GPU training in one process by default.
            if not use_multiprocessing:
                mDict = {}
            else:
                manager = Manager()
                mDict = manager.dict()
            tasks = []
            dataset_dir = os.path.dirname(os.path.abspath(self.config['datapath']))
            folds = self.strictFolds if self.protocol == 'strict' else DataSplit.crossValidation(
                self.trainingData, k, path=dataset_dir
            )
            base_seed = int(self.config['random.seed']) if self.config.contains('random.seed') else 2026
            validation_seed_base = (
                int(self.config['validation.seed'])
                if self.config.contains('validation.seed') else base_seed + 100000
            )
            negative_seed_base = (
                int(self.config['negative.seed'])
                if self.config.contains('negative.seed') else base_seed + 200000
            )
            for i, (train, test) in enumerate(folds, start=1):
                if i not in selected_folds:
                    continue
                fold = '[' + str(i) + ']'
                train_for_model = train
                validation = []
                if self.requiresInnerValidation:
                    validation_seed = validation_seed_base + i - 1
                    train_for_model, validation, validation_info = (
                        DataSplit.innerValidationSplitForConfig(
                            self.config,
                            train,
                            self.earlyStopping['ratio'],
                            validation_seed,
                        )
                    )
                    validation_positive = sum(
                        1 for row in validation if float(row[2]) > 0
                    )
                    validation_negative = (
                        len(validation) - validation_positive
                    )
                    print(
                        'Fold %d inner validation: strategy=%s train %d, '
                        'validation %d (+%d/-%d), seed %d, hash %s.' % (
                            i,
                            validation_info['strategy'],
                            validation_info['inner_train_records'],
                            validation_info['validation_records'],
                            validation_positive,
                            validation_negative,
                            validation_info['seed'],
                            validation_info['assignments_sha256'][:12],
                        )
                    )
                if self.negativeSampling['strategy'] != 'random':
                    reserved_pairs = {
                        (str(row[0]), str(row[1])) for row in validation + test
                    }
                    split_dir = os.path.dirname(self.strictManifest['assignments_path'])
                    train_for_model, negative_info = DataSplit.applyTrainingNegativeStrategy(
                        train_for_model,
                        self.negativeSampling,
                        dataset_dir,
                        reserved_pairs=reserved_pairs,
                        seed=negative_seed_base + i - 1,
                        fold_index=i - 1,
                        manifest_dir=os.path.join(split_dir, 'training_negatives'),
                    )
                    print(
                        'Fold %d training negatives: mixed random=%d hard=%d '
                        '(H-C=%d, P-D=%d), actual_ratio=%.4f, seed=%d, hash=%s.' % (
                            i,
                            negative_info['random_negative_count'],
                            negative_info['hard_negative_count'],
                            negative_info['hard_source_counts'].get('H_C', 0),
                            negative_info['hard_source_counts'].get('P_D', 0),
                            negative_info['hard_ratio_actual'],
                            negative_info['seed'],
                            negative_info['assignments_sha256'][:12],
                        )
                    )
                if self.labelPermutation['enabled']:
                    train_for_model, permutation_info = (
                        permute_training_labels(
                            train_for_model,
                            self.labelPermutation['seed'] + i - 1,
                        )
                    )
                    print(
                        'Fold %d model-train label permutation: records=%d '
                        'positive=%d negative=%d changed=%d seed=%d hash=%s.' % (
                            i,
                            permutation_info['records'],
                            permutation_info['positive_count'],
                            permutation_info['negative_count'],
                            permutation_info['changed_label_count'],
                            permutation_info['seed'],
                            permutation_info['permuted_records_sha256'][:12],
                        )
                    )
                recommender = self.config['model.name'] + "(self.config,train_for_model,test,fold)"
                algorithm = eval(recommender)
                algorithm.validationData = validation
                if not use_multiprocessing:
                    fold_seed = base_seed + i - 1
                    set_global_seed(fold_seed, reset_tensorflow_graph=True)
                    print('Fold %d random seed: %d' % (i, fold_seed))
                    mDict[i] = algorithm.execute()
                    continue
                # create the process
                p = Process(target=run, args=(mDict, algorithm, i))
                tasks.append(p)
            if use_multiprocessing:
                # start the processes
                for p in tasks:
                    p.start()
                    if not self.evaluation.contains('-p'):
                        p.join()
                # wait until all processes are completed
                if self.evaluation.contains('-p'):
                    for p in tasks:
                        p.join()
            # compute the average and standard deviation of k-fold cross validation
            self.measure = [mDict[i] for i in selected_folds if i in mDict]
            res = []
            if not self.measure:
                print('No fold metrics were returned.')
                return
            if len(self.measure) != fold_limit:
                print('Warning: expected %d folds but received metrics from %d folds.' %
                      (fold_limit, len(self.measure)))
            for i in range(len(self.measure[0])):
                measure = self.measure[0][i].split(':')[0]
                values = []
                for j in range(len(self.measure)):
                    values.append(float(self.measure[j][i].split(':')[1]))
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0.0
                res.append('%s:%.6f(±%.6f)\n' % (measure, mean, std))
            # output result
            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = OptionConf(self.config['output.setup'])['-dir']
            variant = self.config['model.variant'] if self.config.contains('model.variant') else self.config['model.name']
            if selected_folds == list(range(1, k + 1)):
                evaluation_label = str(k) + '-fold-cv'
                result_title = 'The result of %d-fold cross validation' % k
            elif self.config.contains('evaluation.fold.indices'):
                fold_text = '-'.join(str(value) for value in selected_folds)
                evaluation_label = 'folds-' + fold_text + '-of-%d-pilot' % k
                result_title = 'Pilot result for fold(s) %s of %d-fold cross validation' % (
                    ','.join(str(value) for value in selected_folds), k
                )
            else:
                evaluation_label = 'first-%d-of-%d-fold-pilot' % (fold_limit, k)
                result_title = 'Pilot result for first %d fold(s) of %d-fold cross validation' % (
                    fold_limit, k
                )
            fileName = variant + '@' + currentTime + '-' + evaluation_label + '.txt'
            FileIO.writeFile(outDir, fileName, res)
            print('%s:\n%s' % (result_title, ''.join(res)))
        else:

            recommender = self.config['model.name'] + '(self.config,self.trainingData,self.testData)'
            eval(recommender).execute()

def run(measure, algor, order):
    measure[order] = algor.execute()
