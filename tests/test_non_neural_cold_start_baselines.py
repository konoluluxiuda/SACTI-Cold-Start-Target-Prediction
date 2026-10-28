import unittest

import numpy as np
import scipy.sparse as sp

from tools.evaluate_non_neural_cold_start_baselines import (
    build_matrices,
    global_target_prior,
    hc_jaccard_label_propagation,
    hc_weighted_nearest_neighbor_profiles,
    herb_prototype_profiles,
)


class NonNeuralColdStartBaselinesTest(unittest.TestCase):
    def setUp(self):
        self.hc_pairs = [
            ('h1', 'c1'),
            ('h1', 'cold'),
            ('h2', 'c2'),
        ]
        self.assignments = [
            ('cold', 'p1', 1, 0),
            ('cold', 'p2', 0, 0),
            ('c1', 'p1', 1, 1),
            ('c1', 'p2', 0, 1),
            ('c2', 'p2', 1, 1),
            ('c2', 'p1', 0, 1),
        ]

    def test_build_matrices_excludes_test_compound_edges(self):
        matrices = build_matrices(self.hc_pairs, self.assignments, 0)
        self.assertEqual(matrices['training_positive_edges'], 2)
        self.assertEqual(matrices['labels'].tolist(), [1, 0])
        train_degrees = np.asarray(
            matrices['cp'].getnnz(axis=1)).reshape(-1)
        self.assertEqual(train_degrees[matrices['test_compounds'][0]], 0)

    def test_build_matrices_includes_side_information_proteins(self):
        matrices = build_matrices(
            self.hc_pairs,
            self.assignments,
            0,
            extra_protein_ids=['p3'],
        )
        self.assertEqual(matrices['protein_ids'], ['p1', 'p2', 'p3'])
        self.assertEqual(matrices['cp'].shape[1], 3)
        self.assertEqual(matrices['entity_counts']['proteins'], 3)

    def test_global_prior_is_fold_local(self):
        cp = sp.csr_matrix([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        prior, supported = global_target_prior(cp)
        np.testing.assert_allclose(prior, [0.5, 0.5])
        self.assertEqual(supported.tolist(), [True, True, False])

    def test_prototype_and_jaccard_rank_supported_target_first(self):
        matrices = build_matrices(self.hc_pairs, self.assignments, 0)
        prototype, prototype_covered = herb_prototype_profiles(
            matrices['hc'],
            matrices['cp'],
            matrices['test_compounds'],
            prior_strength=1.0,
        )
        jaccard, jaccard_covered = hc_jaccard_label_propagation(
            matrices['hc'], matrices['cp'], matrices['test_compounds'])
        self.assertTrue(prototype_covered[0])
        self.assertTrue(jaccard_covered[0])
        self.assertGreater(prototype[0, 0], prototype[0, 1])
        self.assertGreater(jaccard[0, 0], jaccard[0, 1])

    def test_uncovered_compound_falls_back_to_prior(self):
        hc = sp.csr_matrix([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        cp = sp.csr_matrix([
            [1.0, 0.0],
            [0.0, 0.0],
        ])
        prototype, covered = herb_prototype_profiles(
            hc, cp, np.asarray([1]), prior_strength=1.0)
        prior, _ = global_target_prior(cp)
        self.assertFalse(covered[0])
        np.testing.assert_allclose(prototype[0], prior)

    def test_weighted_neighbors_use_rank_decay_and_training_support(self):
        hc = sp.csr_matrix([
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ])
        cp = sp.csr_matrix([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        profiles, covered, used = hc_weighted_nearest_neighbor_profiles(
            hc, cp, np.asarray([2]), neighbor_count=2, rank_decay=0.5)
        self.assertTrue(covered[0])
        self.assertEqual(used[0], 2)
        self.assertGreater(profiles[0, 0], profiles[0, 1])

    def test_weighted_neighbors_fall_back_without_hc_overlap(self):
        hc = sp.csr_matrix([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        cp = sp.csr_matrix([
            [1.0, 0.0],
            [0.0, 0.0],
        ])
        profiles, covered, used = hc_weighted_nearest_neighbor_profiles(
            hc, cp, np.asarray([1]), neighbor_count=5, rank_decay=0.7)
        prior, _ = global_target_prior(cp)
        self.assertFalse(covered[0])
        self.assertEqual(used[0], 0)
        np.testing.assert_allclose(profiles[0], prior)


if __name__ == '__main__':
    unittest.main()
