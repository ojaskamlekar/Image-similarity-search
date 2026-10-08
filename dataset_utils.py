"""Explicit labels and reproducible, content-disjoint dataset partitions."""

import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def discover_images(folder, labels_csv=None):
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    labels = None
    if labels_csv:
        with open(labels_csv, newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if not {'filename', 'label'} <= set(reader.fieldnames or []):
                raise ValueError('Labels CSV requires filename,label columns')
            labels = {}
            for row in reader:
                if row['filename'] in labels or not row['label'].strip():
                    raise ValueError('Duplicate filename or empty label in CSV')
                labels[row['filename']] = row['label'].strip()
        if set(labels) != {p.name for p in files}:
            raise ValueError('CSV must label every dataset image exactly once')
    records, seen = [], {}
    for path in files:
        if labels is None:
            match = re.fullmatch(r'(.+?)(?:[_-]\d+|\s+\(\d+\))', path.stem)
            if not match:
                raise ValueError(f'No explicit category for {path.name}; provide --labels-csv')
            label = match.group(1)
        else:
            label = labels[path.name]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            if seen[digest] != label:
                raise ValueError(f'Identical image has conflicting labels: {path.name}')
            continue
        seen[digest] = label
        records.append({'filename': path.name, 'label': label, 'sha256': digest})
    if not records:
        raise ValueError('No dataset images found')
    return records


def split_records(records, validation_fraction=0.2, test_fraction=0.2, seed=42):
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError('Validation/test fractions must be positive and sum to less than one')
    groups = defaultdict(list)
    for record in records:
        groups[record['label']].append(record)
    if len(groups) < 2:
        raise ValueError('At least two real categories are required')
    splits = {'train': [], 'validation': [], 'test': []}
    rng = random.Random(seed)
    for label, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r['filename'])
        rng.shuffle(rows)
        n_val = max(2, round(len(rows) * validation_fraction))
        n_test = max(2, round(len(rows) * test_fraction))
        if len(rows) - n_val - n_test < 2:
            raise ValueError(f'Category {label!r} needs at least six unique images '
                             '(two per split); add correctly labeled data')
        splits['validation'].extend(rows[:n_val])
        splits['test'].extend(rows[n_val:n_val + n_test])
        splits['train'].extend(rows[n_val + n_test:])
    return splits


def split_counts(splits):
    return {name: {'images': len(rows), 'categories': dict(Counter(r['label'] for r in rows))}
            for name, rows in splits.items()}


def validate_manifest(manifest, folder):
    """Reject overlapping partitions or changed files before evaluation."""
    names, hashes = set(), set()
    for split in ('train', 'validation', 'test'):
        for record in manifest['splits'][split]:
            name, digest = record['filename'], record['sha256']
            if Path(name).name != name or name in names or digest in hashes:
                raise ValueError('Manifest has unsafe paths or overlapping images')
            if hashlib.sha256((Path(folder) / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Dataset changed since training: {name}')
            names.add(name)
            hashes.add(digest)
