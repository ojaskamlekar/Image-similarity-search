import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from prepare_clothing_dataset import prepare


class PreparationTests(unittest.TestCase):
    def test_explicit_labels_and_audited_exclusions(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'
            (source / 'images').mkdir(parents=True)
            (source / 'LICENSE').write_text('CC0')
            rows = []
            for category, label in enumerate(('Shirt', 'Pants')):
                for i in range(6):
                    image_id = f'{label}-{i}'
                    Image.new('RGB', (8, 8), (30 * category, 20 * i, 100)).save(source / 'images' / f'{image_id}.jpg')
                    rows.append({'image': image_id, 'sender_id': '1', 'label': label, 'kids': 'False'})
            for image_id, label in [('unknown', 'Not sure'), ('rare', 'Hat')]:
                Image.new('RGB', (8, 8), (255, 255, 255)).save(source / 'images' / f'{image_id}.jpg')
                rows.append({'image': image_id, 'sender_id': '1', 'label': label, 'kids': 'False'})
            (source / 'images' / 'copy.jpg').write_bytes((source / 'images' / 'Shirt-0.jpg').read_bytes())
            rows.append({'image': 'copy', 'sender_id': '1', 'label': 'Shirt', 'kids': 'False'})
            with (source / 'images.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['image', 'sender_id', 'label', 'kids'])
                writer.writeheader()
                writer.writerows(rows)
            destination = Path(folder) / 'prepared'
            with patch('prepare_clothing_dataset.subprocess.check_output', return_value='source-sha\n'):
                report = prepare(source, destination, 6)
            self.assertEqual(report['retained_images'], 12)
            self.assertEqual(report['category_counts'], {'Pants': 6, 'Shirt': 6})
            self.assertEqual(len(report['excluded']), 3)
            self.assertEqual(report['source_commit'], 'source-sha')
            self.assertEqual(len(list((destination / 'images').iterdir())), 12)
            self.assertEqual(json.loads((destination / 'preparation_report.json').read_text())['license'], 'CC0-1.0')
            with self.assertRaisesRegex(ValueError, 'overwritten'):
                prepare(source, destination, 6)
