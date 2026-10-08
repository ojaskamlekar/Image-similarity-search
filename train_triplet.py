"""Train on explicit image partitions; reserve test images for retrieval evaluation."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from dataset_utils import discover_images, split_records, split_counts


class TripletTrainer:
    def __init__(self, triplet_model, base_network, data_generator, validation_generator, output_dir='.'):
        self.triplet_model = triplet_model
        self.base_network = base_network
        self.data_generator = data_generator
        self.validation_generator = validation_generator
        self.history = {'loss': [], 'val_loss': []}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, epochs=10, batch_size=16, steps_per_epoch=20, validation_steps=10, save_every=5):
        import numpy as np
        if min(epochs, batch_size, steps_per_epoch, validation_steps, save_every) < 1:
            raise ValueError('Training counts must be positive')
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        # Fixed validation triplets make epoch comparisons reproducible.
        validation_batches = [self.validation_generator.generate_batch(batch_size)
                              for _ in range(validation_steps)]
        dummy = np.zeros((batch_size, 1), dtype='float32')
        best_loss = float('inf')
        for epoch in range(epochs):
            started = time.monotonic()
            train_losses = []
            for step in range(steps_per_epoch):
                self.triplet_model.reset_metrics()
                inputs = self.data_generator.generate_batch(batch_size)
                train_losses.append(float(self.triplet_model.train_on_batch(list(inputs), dummy)))
                if (step + 1) % 25 == 0 or step + 1 == steps_per_epoch:
                    print(f'Epoch {epoch + 1}/{epochs}, step {step + 1}/{steps_per_epoch}: '
                          f'loss={np.mean(train_losses[-25:]):.5f}, '
                          f'elapsed={time.monotonic() - started:.0f}s', flush=True)
            val_losses = []
            for inputs in validation_batches:
                self.triplet_model.reset_metrics()
                val_losses.append(float(self.triplet_model.test_on_batch(list(inputs), dummy)))
            loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
            self.history['loss'].append(loss)
            self.history['val_loss'].append(val_loss)
            print(f'Epoch {epoch + 1}/{epochs}: loss={loss:.5f}, val_loss={val_loss:.5f}', flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                self.base_network.save(str(self.output_dir / 'triplet_base_final.h5'))
                print('Saved improved validation checkpoint', flush=True)
            if (epoch + 1) % save_every == 0:
                self.base_network.save(str(self.output_dir / 'checkpoints' / f'triplet_base_epoch_{epoch + 1}.h5'))
            (self.output_dir / 'training_history.json').write_text(json.dumps(self.history, indent=2))
        import matplotlib.pyplot as plt
        for name, values in self.history.items():
            plt.plot(range(1, epochs + 1), values, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Triplet loss (squared Euclidean distance)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png')
        plt.close()
        return self.history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-folder', default='static/dataset')
    parser.add_argument('--labels-csv', help='CSV with filename,label columns; otherwise use category prefixes')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps-per-epoch', type=int, default=20)
    parser.add_argument('--validation-steps', type=int, default=10)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--test-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-test-images', type=int, default=100,
                        help='Project guardrail, not a statistical guarantee')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Allow a small dataset; reports remain explicitly non-benchmark')
    parser.add_argument('--audit-only', action='store_true', help='Validate labels and splits without TensorFlow')
    parser.add_argument('--output-dir', default='.', help='Directory for model, manifest, and training history')
    parser.add_argument('--require-gpu', action='store_true', help='Fail rather than silently train on CPU')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.steps_per_epoch, args.validation_steps,
           args.min_test_images) < 1:
        parser.error('Counts must be positive')
    try:
        records = discover_images(args.image_folder, args.labels_csv)
        splits = split_records(records, args.validation_fraction, args.test_fraction, args.seed)
        counts = split_counts(splits)
        print(json.dumps(counts, indent=2))
        if not args.smoke_test and len(splits['test']) < args.min_test_images:
            raise ValueError(f"Only {len(splits['test'])} unique held-out test images; "
                             f"need at least {args.min_test_images}. Add labeled data, "
                             "or use --smoke-test for pipeline checks only.")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.audit_only:
        return
    import tensorflow as tf
    from triplet_data_generator import TripletDataGenerator
    from triplet_model import create_triplet_network, compile_triplet_model
    tf.keras.utils.set_random_seed(args.seed)
    devices = tf.config.list_physical_devices('GPU')
    if args.require_gpu and not devices:
        parser.error('No TensorFlow GPU available')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'triplet_base_final.h5').exists():
        parser.error('Output model already exists; choose a new --output-dir')
    config = {**vars(args), 'tensorflow_version': tf.__version__,
              'gpu_devices': [str(d) for d in devices], 'input_shape': [224, 224, 3],
              'embedding_dimension': 128}
    (output_dir / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    train_data = TripletDataGenerator(args.image_folder, records=splits['train'], seed=args.seed)
    val_data = TripletDataGenerator(args.image_folder, records=splits['validation'], seed=args.seed + 1)
    # Decode all images up front; corrupt images must not silently reduce the evaluation set.
    for row in records:
        train_data.load_and_preprocess_image(Path(args.image_folder) / row['filename'])
    triplet_model, base_network = create_triplet_network()
    compile_triplet_model(triplet_model)
    trainer = TripletTrainer(triplet_model, base_network, train_data, val_data, output_dir)
    trainer.train(args.epochs, args.batch_size, args.steps_per_epoch, args.validation_steps)
    manifest = {'schema_version': 1, 'seed': args.seed, 'smoke_test': args.smoke_test,
                'minimum_test_images': args.min_test_images, 'counts': counts, 'splits': splits,
                'model_sha256': hashlib.sha256((output_dir / 'triplet_base_final.h5').read_bytes()).hexdigest()}
    (output_dir / 'dataset_split.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Saved best validation checkpoint and dataset_split.json. Run evaluate_triplet.py next.')


if __name__ == '__main__':
    main()
