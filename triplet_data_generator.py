"""Sample triplets from explicit labels within one dataset partition."""

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from dataset_utils import discover_images


class TripletDataGenerator:
    def __init__(self, image_folder, target_size=(224, 224), records=None, seed=42):
        self.image_folder = Path(image_folder)
        self.target_size = target_size
        self.rng = random.Random(seed)
        self.records = discover_images(image_folder) if records is None else records
        self.label_to_images = defaultdict(list)
        for row in self.records:
            self.label_to_images[row['label']].append(str(self.image_folder / row['filename']))
        if len(self.label_to_images) < 2 or any(len(v) < 2 for v in self.label_to_images.values()):
            raise ValueError('Each partition needs two categories with two images per category')
        self.images = [str(self.image_folder / r['filename']) for r in self.records]
        self.labels = [r['label'] for r in self.records]

    def load_and_preprocess_image(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f'Could not load image: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, self.target_size).astype('float32') / 255.0

    def generate_triplet(self):
        anchor_label, negative_label = self.rng.sample(sorted(self.label_to_images), 2)
        anchor, positive = self.rng.sample(self.label_to_images[anchor_label], 2)
        negative = self.rng.choice(self.label_to_images[negative_label])
        return tuple(self.load_and_preprocess_image(p) for p in (anchor, positive, negative))

    def generate_batch(self, batch_size=32):
        return tuple(np.stack(images) for images in zip(*(self.generate_triplet() for _ in range(batch_size))))

    def get_statistics(self):
        return {'total_images': len(self.images), 'num_categories': len(self.label_to_images),
                'images_per_category': {k: len(v) for k, v in self.label_to_images.items()}}
