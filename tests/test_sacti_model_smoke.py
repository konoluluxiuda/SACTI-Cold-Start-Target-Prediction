import ast
import json
import os
import tempfile
import unittest
from pathlib import Path


class DummyConf(object):
    def __init__(self, values):
        self.config = dict(values)

    def __getitem__(self, key):
        return self.config[key]

    def contains(self, key):
        return key in self.config


class SACTIModelSmokeTest(unittest.TestCase):
    def test_source_has_no_hdcti_dependency(self):
        source_path = Path(__file__).resolve().parents[1] / 'SACTI.py'
        tree = ast.parse(source_path.read_text(encoding='utf-8'))
        imported_modules = []
        base_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or '')
            elif isinstance(node, ast.ClassDef) and node.name == 'SACTI':
                base_names.extend(
                    base.id for base in node.bases
                    if isinstance(base, ast.Name)
                )
        self.assertFalse(any(
            module == 'HDCTI' or module.startswith('HDCTI.')
            for module in imported_modules
        ))
        self.assertNotIn('HDCTI', base_names)
        self.assertIn('herbRecommender', base_names)

    def test_independent_model_trains_and_disables_cold_id_expert(self):
        self._run_training_smoke({})

    def test_pseudo_cold_identity_only_training(self):
        self._run_training_smoke({
            'sacti.training.mask.id': 'True',
            'sacti.training.blocks': '2',
        })

    def test_query_local_prior_training(self):
        self._run_training_smoke({'sacti.transfer.exclude.query.prior': 'True'})

    def test_raw_transfer_training(self):
        self._run_training_smoke({'sacti.transfer.raw.scores': 'True'})

    def test_hc_reconstruction_training(self):
        self._run_training_smoke({'sacti.hc.reconstruction.weight': '0.1',
                                  'sacti.hc.reconstruction.samples': '2'})

    def test_pseudo_cold_evidence_training(self):
        self._run_training_smoke({
            'sacti.training.mask.id': 'True',
            'sacti.training.mask.evidence': 'True',
            'sacti.training.blocks': '2',
        })

    def _run_training_smoke(self, extra_config):
        import numpy as np
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from SACTI import SACTI

        tf.reset_default_graph()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            (dataset_dir / 'H_C.txt').write_text(
                'h0\tc0\nh0\tc1\nh0\tc2\nh1\tc1\nh2\tc2\n',
                encoding='utf-8',
            )
            (dataset_dir / 'C_P.txt').write_text(
                'c0\tp0\nc1\tp1\nc2\tp0\n', encoding='utf-8'
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np0\td1\np1\td1\n', encoding='utf-8'
            )
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text(
                'c0\tp0\t1\nc1\tp1\t1\nc2\tp0\t1\n',
                encoding='utf-8',
            )
            conf = DummyConf({
                'datapath': str(datapath),
                'ratings.setup': '-columns 0 1 2',
                'evaluation.setup': '-cv 2',
                'experiment.protocol': 'strict',
                'model.name': 'SACTI',
                'num.factors': '4',
                'num.max.epoch': '1',
                'batch_size': '4',
                'learnRate': '-init 0.005 -max 1',
                'reg.lambda': '-u 0.001 -i 0.001 -b 0.2 -s 0.2',
                'weight.reg': '0.01',
                'sacti.context.layers': '1',
                'sacti.prototype': 'True',
                'sacti.jaccard': 'True',
                'sacti.support.id.gate': 'True',
                'sacti.context.expert': 'True',
                'sacti.prototype.prior': '1.0',
                'output.setup': 'off -dir ./results/',
                'gpu.allow_growth': 'False',
                'gpu.log_device_placement': 'False',
            })
            training = [
                ['c0', 'p0', 1.0], ['c0', 'p1', 0.0],
                ['c1', 'p1', 1.0], ['c1', 'p0', 0.0],
            ]
            conf.config.update(extra_config)
            test = [['c2', 'p0', 1.0], ['c2', 'p1', 0.0]]
            previous_directory = os.getcwd()
            try:
                os.chdir(temporary_directory)
                model = SACTI(conf, training, test, '[1]')
                model.readConfiguration()
                model.initModel()
                self.assertEqual(
                    model.compound_support_degrees[
                        model.data.compound['c2']
                    ],
                    0,
                )
                operation_names = {
                    operation.name.lower()
                    for operation in tf.get_default_graph().get_operations()
                }
                self.assertFalse(any('pagerank' in name for name in operation_names))
                self.assertFalse(any('self_attention' in name for name in operation_names))
                model.sess.run(tf.global_variables_initializer())
                compound_indices = np.asarray([
                    model.data.compound['c0'], model.data.compound['c1'],
                    model.data.compound['c2'],
                ])
                protein_indices = np.zeros(3, dtype=np.int32)
                regular = model._feed(compound_indices, protein_indices)
                before = model.sess.run(model.logits, feed_dict=regular)
                np.testing.assert_allclose(
                    before, model.scorePairs(model.fetchModelState(),
                                             compound_indices, protein_indices),
                    atol=1e-6,
                )
                original_cp = model.cp_matrix.copy()
                original_prior = model.herb_prototypes['target_prevalence'].copy()
                model.prepareTrainingEpoch(0)
                masked = model._feed(compound_indices, protein_indices, training=True)
                if model.training_mask_id:
                    selected = model.training_query_mask[compound_indices]
                    gate = model.sess.run(model.pair_id_gate, feed_dict=masked)
                    self.assertTrue(np.all(gate[selected] == 0))
                    self.assertNotIn(model.training_id_keep, regular)
                    if model.training_mask_evidence:
                        expected = model.transferResiduals(
                            compound_indices[selected], protein_indices[selected],
                            evidence=model.training_transfer_state,
                        )
                        np.testing.assert_array_equal(
                            masked[model.prototype_residual][selected], expected[0])
                        np.testing.assert_array_equal(
                            masked[model.jaccard_residual][selected], expected[1])
                    else:
                        np.testing.assert_array_equal(masked[model.prototype_residual],
                                                      regular[model.prototype_residual])
                else:
                    self.assertFalse(hasattr(model, 'training_id_keep'))
                    for key in regular:
                        np.testing.assert_array_equal(regular[key], masked[key])
                # Epoch-local state must not change ordinary inference or source evidence.
                np.testing.assert_array_equal(before, model.sess.run(
                    model.logits, feed_dict=model._feed(compound_indices, protein_indices)))
                self.assertEqual((original_cp != model.cp_matrix).nnz, 0)
                np.testing.assert_array_equal(original_prior,
                                              model.herb_prototypes['target_prevalence'])
                model.trainModel()
                scores = model.predictForPairs(
                    [model.data.compound['c2'], model.data.compound['c2']],
                    [model.data.protein['p0'], model.data.protein['p1']],
                )
                self.assertTrue(np_all_finite(scores))
                metadata_paths = list(
                    (dataset_dir / 'saved_model').rglob('sacti.json')
                )
                self.assertEqual(len(metadata_paths), 1)
                metadata = json.loads(
                    metadata_paths[0].read_text(encoding='utf-8')
                )
                if model.hc_reconstruction_weight > 0:
                    self.assertGreater(metadata['hc_reconstruction']['sampler']['drawn_pairs'], 0)
                    self.assertEqual(metadata['hc_reconstruction']['sampler']['eligible_compounds'], 1)
                self.assertFalse(metadata['inherits_hdcti'])
                self.assertFalse(metadata['encoder']['uses_pagerank'])
                self.assertTrue(metadata['support']['id_gate_enabled'])
                self.assertTrue(
                    metadata['support']['context_expert_enabled']
                )
                self.assertEqual(metadata['pseudo_cold_training']['mask_id'],
                                 extra_config.get('sacti.training.mask.id') == 'True')
                self.assertEqual(metadata['transfer']['exclude_query_prior'],
                                 extra_config.get('sacti.transfer.exclude.query.prior') == 'True'
                                 or extra_config.get('sacti.transfer.raw.scores') == 'True')
                self.assertEqual(metadata['transfer']['raw_scores'],
                                 extra_config.get('sacti.transfer.raw.scores') == 'True')
                if model.training_mask_id:
                    self.assertEqual(metadata['pseudo_cold_training']['masked_records'], 2)
                model.sess.close()
            finally:
                os.chdir(previous_directory)

    def test_pair_scoring_component_switches(self):
        import numpy as np

        from SACTI import SACTI

        model = object.__new__(SACTI)
        model.compound_support_degrees = np.asarray([0, 2], dtype=np.int64)
        model.compound_context_available = np.asarray(
            [1.0, 1.0], dtype=np.float32
        )
        model.prototype_enabled = False
        model.jaccard_enabled = False
        state = {
            'compound_id': np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            'compound_context': np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            'protein': np.asarray([[1.0, 1.0]]),
            'context_interaction': np.asarray([1.0, 1.0]),
            'prototype_scale': np.asarray([0.0]),
            'jaccard_scale': np.asarray([0.0]),
        }
        compounds = np.asarray([0, 1], dtype=np.int64)
        proteins = np.asarray([0, 0], dtype=np.int64)

        model.support_id_gate_enabled = True
        model.context_expert_enabled = True
        np.testing.assert_allclose(
            model.scorePairs(state, compounds, proteins),
            np.asarray([1.0, 2.0], dtype=np.float32),
        )

        model.support_id_gate_enabled = False
        np.testing.assert_allclose(
            model.scorePairs(state, compounds, proteins),
            np.asarray([2.0, 2.0], dtype=np.float32),
        )

        model.support_id_gate_enabled = True
        model.context_expert_enabled = False
        np.testing.assert_allclose(
            model.scorePairs(state, compounds, proteins),
            np.asarray([0.0, 1.0], dtype=np.float32),
        )


def np_all_finite(values):
    import numpy as np
    return bool(np.all(np.isfinite(values)))


if __name__ == '__main__':
    unittest.main()
