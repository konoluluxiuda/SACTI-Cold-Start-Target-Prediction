import json
from pathlib import Path

from tools.run_sacti_outer_lockbox import (
    build_jobs,
    checkpoint_prefixes,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


def test_outer_lockbox_builds_matched_five_fold_configs(tmp_path):
    manifest = json.loads(
        (ROOT / 'configs' / 'sacti_outer_lockbox_seed901026.json')
        .read_text(encoding='utf-8')
    )

    jobs = build_jobs(manifest, tmp_path)

    assert len(jobs) == 8
    assert {job['model_name'] for job in jobs} == {'HDCTI', 'SACTI'}
    for job in jobs:
        text = Path(job['config']).read_text(encoding='utf-8')
        assert 'split.seed=901026' in text
        assert 'evaluation.fold.limit=5' in text
        assert 'evaluation.outer.test=True' in text


def test_outer_lockbox_extracts_five_unique_checkpoints(tmp_path):
    lines = []
    expected = []
    for fold in range(1, 6):
        prefix = tmp_path / ('fold_%d' % fold) / 'model.ckpt'
        prefix.parent.mkdir(parents=True)
        Path(str(prefix) + '.index').write_text('', encoding='utf-8')
        lines.append('模型权重保存成功: %s' % prefix)
        expected.append(prefix.resolve())
    log_path = tmp_path / 'training.log'
    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    assert checkpoint_prefixes(log_path, 5) == expected


def test_outer_lockbox_markdown_discloses_lockbox_boundary():
    metric = {
        'baseline_mean': 0.4,
        'candidate_mean': 0.5,
        'delta': 0.1,
    }
    report = {
        'split_seed': 901026,
        'training_seed': 52026,
        'interpretation': 'SUPPORTS_SACTI_REPLACEMENT',
        'rows': [{
            'dataset': 'Example',
            'metrics': {
                'AUPR': metric,
                'MRR': metric,
                'Recall@20': metric,
            },
        }],
        'macro': {
            name: {
                'delta': 0.1,
                'stratified_fold_bootstrap_CI': [0.05, 0.15],
                'positive_dataset_count': 4,
                'positive_fold_count': 20,
            }
            for name in ('AUPR', 'MRR', 'Recall@20')
        },
        'checks': {'seal_precedes_training': True},
    }

    text = render_markdown(report)

    assert 'Lockbox consumed: `true`' in text
    assert 'does not reopen model selection' in text
