"""Prepare the user-selected CC0 clothing dataset without inventing labels."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image


def prepare(source, destination, minimum_per_category=30):
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination must be new or empty; existing data will not be overwritten')
    destination.mkdir(parents=True, exist_ok=True)
    with (source / 'images.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    candidates, excluded, seen = [], [], {}
    for row in rows:
        label = row['label'].strip()
        filename = row['image'] + '.jpg'
        path = source / 'images' / filename
        if not label or label.casefold() in {'not sure', 'other', 'unknown', 'skip'}:
            excluded.append({'filename': filename, 'reason': 'ambiguous label', 'label': label})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            excluded.append({'filename': filename, 'reason': str(exc), 'label': label})
            continue
        if digest in seen:
            if seen[digest] != label:
                raise ValueError(f'Identical content has conflicting labels: {filename}')
            excluded.append({'filename': filename, 'reason': 'byte-identical duplicate', 'label': label})
            continue
        seen[digest] = label
        candidates.append({'filename': filename, 'label': label, 'sha256': digest,
                           'sender_id': row['sender_id'], 'kids': row['kids']})
    counts = Counter(row['label'] for row in candidates)
    retained = []
    for row in candidates:
        if counts[row['label']] < minimum_per_category:
            excluded.append({'filename': row['filename'], 'label': row['label'],
                             'reason': f'fewer than {minimum_per_category} unique images in category'})
        else:
            retained.append(row)
    if len({row['label'] for row in retained}) < 2:
        raise ValueError('Not enough labeled categories')
    images = destination / 'images'
    images.mkdir()
    for row in retained:
        shutil.copy2(source / 'images' / row['filename'], images / row['filename'])
    with (destination / 'labels.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['filename', 'label', 'group'])
        writer.writeheader()
        writer.writerows({'filename': row['filename'], 'label': row['label'],
                          'group': row['sender_id']} for row in retained)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    report = {'source': 'https://github.com/alexeygrigorev/clothing-dataset',
              'source_commit': commit, 'license': 'CC0-1.0', 'metadata_rows': len(rows),
              'retained_images': len(retained), 'minimum_per_category': minimum_per_category,
              'category_counts': dict(sorted(Counter(r['label'] for r in retained).items())),
              'records': retained, 'excluded': excluded}
    (destination / 'preparation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    shutil.copy2(source / 'LICENSE', destination / 'DATASET_LICENSE')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', default='data/clothing')
    parser.add_argument('--minimum-per-category', type=int, default=30)
    args = parser.parse_args()
    if args.minimum_per_category < 6:
        parser.error('Need at least six images per category for the current three-way split')
    report = prepare(args.source, args.destination, args.minimum_per_category)
    print(json.dumps({k: v for k, v in report.items() if k not in {'records', 'excluded'}}, indent=2))
    print(f"Excluded {len(report['excluded'])} images; details in preparation_report.json")


if __name__ == '__main__':
    main()
