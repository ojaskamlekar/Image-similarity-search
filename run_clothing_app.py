"""Open the local demo with a trained clothing run and its matching index."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='runs/clothing-20261009')
    parser.add_argument('--image-folder', default='data/clothing/images')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    image_folder = Path(args.image_folder).resolve()
    model_path = run_dir / 'triplet_base_final.h5'
    metadata = json.loads((run_dir / 'triplet_metadata.json').read_text(encoding='utf-8'))
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata['model_sha256']:
        parser.error('Model and search index do not match; regenerate embeddings')
    for row in metadata['images']:
        if Path(row['filename']).name != row['filename']:
            parser.error('Unsafe filename in index')
        if hashlib.sha256((image_folder / row['filename']).read_bytes()).hexdigest() != row['sha256']:
            parser.error(f"Search image changed: {row['filename']}")
    os.environ['IMAGE_SIMILARITY_MODEL_PATH'] = str(model_path)
    os.environ['IMAGE_SIMILARITY_FEATURE_PREFIX'] = str(run_dir / 'triplet')
    os.environ['IMAGE_SIMILARITY_DATASET_FOLDER'] = str(image_folder)
    from app_triplet import app, model, features
    if model is None or not len(features):
        parser.error('Could not load trained model and index')
    app.run(host='127.0.0.1', port=args.port, debug=False)


if __name__ == '__main__':
    main()
