import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from dataset_utils import discover_images, split_records, validate_manifest
from retrieval_metrics import retrieval_metrics
from similarity_scoring import distance_scores


class DatasetTests(unittest.TestCase):
    def make_dataset(self, folder, n=10):
        for label in ('shirt', 'pants'):
            for i in range(n):
                (Path(folder) / f'{label}_{i:03}.jpg').write_bytes(f'{label}:{i}'.encode())

    def test_small_named_dataset_keeps_real_labels_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_dataset(folder)
            (Path(folder) / 'shirt_999.jpg').write_bytes(b'shirt:0')
            rows = discover_images(folder)
            self.assertEqual(len(rows), 20)
            self.assertEqual({r['label'] for r in rows}, {'shirt', 'pants'})
            splits = split_records(rows)
            self.assertEqual(splits, split_records(rows))
            hashes = [{r['sha256'] for r in rows} for rows in splits.values()]
            for i, left in enumerate(hashes):
                for right in hashes[i + 1:]:
                    self.assertFalse(left & right)
            validate_manifest({'splits': splits}, folder)
            (Path(folder) / 'shirt_000.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                validate_manifest({'splits': splits}, folder)

    def test_unlabeled_numbers_require_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / '123.jpg').write_bytes(b'a')
            with self.assertRaisesRegex(ValueError, 'explicit category'):
                discover_images(folder)
            csv = Path(folder) / 'labels.csv'
            csv.write_text('filename,label\n123.jpg,shirt\n')
            self.assertEqual(discover_images(folder, csv)[0]['label'], 'shirt')

    def test_too_small_category_and_invalid_fractions_fail(self):
        rows = [{'filename': f'{label}_{i}.jpg', 'label': label}
                for label in ('shirt', 'pants') for i in range(5)]
        with self.assertRaisesRegex(ValueError, 'six unique'):
            split_records(rows)
        with self.assertRaises(ValueError):
            split_records(rows, 0.6, 0.6)

    def test_contributor_groups_are_disjoint_and_reproducible(self):
        rows = [{'filename': f'{label}_{group}_{i}.jpg', 'label': label, 'group': str(group)}
                for label in ('shirt', 'pants') for group in range(12) for i in range(3)]
        splits = split_records(rows)
        self.assertEqual(splits, split_records(rows))
        groups = [{row['group'] for row in records} for records in splits.values()]
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                self.assertFalse(left & right)
        for records in splits.values():
            self.assertEqual({r['label'] for r in records}, {'shirt', 'pants'})
        with self.assertRaisesRegex(ValueError, 'three groups'):
            split_records([{**row, 'group': 'one'} for row in rows])

    def test_conflicting_duplicate_and_overlapping_manifest_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_dataset(folder)
            (Path(folder) / 'pants_999.jpg').write_bytes(b'shirt:0')
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                discover_images(folder)
        row = {'filename': 'x.jpg', 'label': 'x', 'sha256': hashlib.sha256(b'x').hexdigest()}
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'x.jpg').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                validate_manifest({'splits': {'train': [row], 'validation': [row], 'test': []}}, folder)

    def test_default_size_guardrail_and_smoke_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_dataset(folder)
            command = [sys.executable, 'train_triplet.py', '--audit-only', '--image-folder', folder]
            normal = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(normal.returncode, 0)
            self.assertIn('held-out test images', normal.stderr)
            smoke = subprocess.run(command + ['--smoke-test'], capture_output=True, text=True)
            self.assertEqual(smoke.returncode, 0, smoke.stderr)
            self.assertEqual(json.loads(smoke.stdout)['test']['images'], 4)


class RetrievalTests(unittest.TestCase):
    def test_perfect_ranking_excludes_self_and_caps_k(self):
        report = retrieval_metrics([[0], [0.1], [10], [10.1]], ['a', 'a', 'b', 'b'],
                                   ks=(1, 10), bootstrap_samples=50)
        self.assertEqual(report['metrics']['mAP']['value'], 1)
        self.assertEqual(report['metrics']['precision@1']['value'], 1)
        self.assertEqual(report['metrics']['precision@10']['value'], 1 / 3)
        self.assertEqual(report['effective_k']['10'], 3)
        self.assertEqual(report['metrics']['recall@10']['value'], 1)

    def test_known_interleaved_ranking(self):
        report = retrieval_metrics([[0], [1], [2], [3]], ['a', 'b', 'a', 'b'],
                                   ks=(1,), bootstrap_samples=50)
        self.assertAlmostEqual(report['metrics']['mAP']['value'], (0.5 + 1/3 + 1/3 + 0.5) / 4)
        self.assertEqual(report['metrics']['precision@1']['value'], 0)

    def test_invalid_evaluation_fails(self):
        for embeddings, labels in [([[0], [1]], ['a', 'b']),
                                   ([[0], [float('nan')]], ['a', 'a'])]:
            with self.assertRaises(ValueError):
                retrieval_metrics(embeddings, labels)

    def test_score_bounds_and_invalid_distances(self):
        np.testing.assert_allclose(distance_scores([0, 0.18, 1, 2, 2.01]), [1, .91, .5, 0, 0])
        for invalid in ([float('nan')], [-1]):
            with self.assertRaises(ValueError):
                distance_scores(invalid)


if __name__ == '__main__':
    unittest.main()
