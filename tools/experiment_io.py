"""Configuration, checksum and log helpers for the published experiments."""
import hashlib
import json
from pathlib import Path
import re

METRICS = ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score')
SUMMARY_MARKER = 'The result of 5-fold cross validation:'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_config(path):
    rows, values = [], {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(('#', ';')) or '=' not in line:
            rows.append((None, line))
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        values[key] = value.strip()
        rows.append((key, None))
    return rows, values


def parse_config_values(path):
    values = {}
    for line_number, raw in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(('#', ';')):
            continue
        if '=' not in line:
            raise ValueError('%s:%d is not key=value.' % (path, line_number))
        key, value = line.split('=', 1)
        key = key.strip()
        if key in values:
            raise ValueError('%s has duplicate key %s.' % (path, key))
        values[key] = value.strip()
    return values


def write_resolved_config(base_path, output_path, overrides, variant):
    rows, values = parse_config(base_path)
    values.update({str(key): str(value) for key, value in overrides.items()})
    values['model.variant'] = variant
    written, lines = set(), []
    for key, raw in rows:
        if key is None:
            lines.append(raw)
            continue
        if key in written:
            continue
        lines.append('%s=%s' % (key, values[key]))
        written.add(key)
    for key, value in values.items():
        if key not in written:
            lines.append('%s=%s' % (key, value))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _metric_pattern(metric, require_space):
    separator = r'\s+' if require_space else r'\s*'
    return re.compile(r'^%s:%s([0-9]+(?:\.[0-9]+)?)' % (re.escape(metric), separator), re.MULTILINE)


def parse_five_fold_log(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8', errors='replace')
    if SUMMARY_MARKER not in text:
        raise ValueError('No five-fold summary found in %s.' % path)
    fold_text, summary_text = text.rsplit(SUMMARY_MARKER, 1)
    fold_metrics, summary_metrics = {}, {}
    for metric in METRICS:
        fold_metrics[metric] = [float(x) for x in _metric_pattern(metric, True).findall(fold_text)]
        values = _metric_pattern(metric, False).findall(summary_text)
        if not values:
            raise ValueError('No summary %s found in %s.' % (metric, path))
        summary_metrics[metric] = float(values[0])
    fold_count = len(fold_metrics['AUPR'])
    if fold_count != 5:
        raise ValueError('Expected 5 fold AUPR values in %s, found %d.' % (path, fold_count))
    if any(len(values) != fold_count for values in fold_metrics.values()):
        raise ValueError('Fold metric counts disagree in %s.' % path)
    return dict(text=text, fold_metrics=fold_metrics, summary_metrics=summary_metrics)


def relation_matrix(pairs, left, right):
    from scipy import sparse
    li, ri = {x: i for i, x in enumerate(left)}, {x: i for i, x in enumerate(right)}
    return sparse.csr_matrix(([1.] * len(pairs),
                             ([li[str(a)] for a, _ in pairs], [ri[str(b)] for _, b in pairs])),
                            shape=(len(left), len(right)))
