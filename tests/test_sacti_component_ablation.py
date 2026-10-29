import json
import tempfile
import unittest
from pathlib import Path

from tools.experiment_io import parse_config
from tools.run_sacti_component_ablation import build_jobs, summarize


ROOT = Path(__file__).resolve().parents[1]


def five_fold_log(summary_aupr, fold_aupr):
    lines = []
    for value in fold_aupr:
        lines.extend([
            'AUC: %.6f' % min(0.999999, value + 0.01),
            'AUPR: %.6f' % value,
            'Recall: 0.800000',
            'Precision: 0.810000',
            'F1-score: 0.805000',
        ])
    lines.extend([
        'The result of 5-fold cross validation:',
        'AUC:%.6f(+-0.001000)' % min(0.999999, summary_aupr + 0.01),
        'AUPR:%.6f(+-0.001000)' % summary_aupr,
        'Recall:0.800000(+-0.001000)',
        'Precision:0.810000(+-0.001000)',
        'F1-score:0.805000(+-0.001000)',
    ])
    return '\n'.join(lines) + '\n'


class SACTIComponentAblationTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((
            ROOT / 'configs' / 'sacti_component_ablation_seed52026.json'
        ).read_text(encoding='utf-8'))

    def test_builds_six_frozen_roles_for_four_datasets(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            jobs = build_jobs(self.manifest, Path(temporary_dir))
            self.assertEqual(len(jobs), 24)
            self.assertEqual(
                {job['role'] for job in jobs},
                {
                    'no_transfer', 'prototype_only', 'jaccard_only',
                    'full', 'no_support_gate', 'no_context_expert',
                },
            )
            for job in jobs:
                _, values = parse_config(Path(job['config']))
                self.assertEqual(values['model.name'], 'SACTI')
                self.assertEqual(values['split.seed'], '52026')
                self.assertEqual(values['evaluation.fold.limit'], '5')
                self.assertEqual(values['evaluation.outer.test'], 'True')

    def test_summary_computes_paired_and_factorial_effects(self):
        role_aupr = {
            'no_transfer': 0.70,
            'prototype_only': 0.72,
            'jaccard_only': 0.73,
            'full': 0.76,
            'no_support_gate': 0.71,
            'no_context_expert': 0.74,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            jobs = build_jobs(self.manifest, Path(temporary_dir))
            for job in jobs:
                value = role_aupr[job['role']]
                folds = [value - 0.002, value - 0.001, value,
                         value + 0.001, value + 0.002]
                path = Path(job['training_log'])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    five_fold_log(value, folds), encoding='utf-8'
                )
            report = summarize(self.manifest, jobs)

        self.assertFalse(report['model_selection_reopened'])
        self.assertAlmostEqual(
            report['effect_summary']['support_gate']['macro_aupr_delta'],
            0.05,
        )
        self.assertAlmostEqual(
            report['effect_summary']['context_expert']['macro_aupr_delta'],
            0.02,
        )
        self.assertAlmostEqual(
            report['effect_summary']['joint_transfer']['macro_aupr_delta'],
            0.06,
        )
        self.assertAlmostEqual(
            report['effect_summary']['prototype_jaccard_interaction'][
                'macro_aupr_delta'
            ],
            0.01,
        )


if __name__ == '__main__':
    unittest.main()
