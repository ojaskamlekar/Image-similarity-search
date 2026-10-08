# AI Image Similarity Search

A Flask demo that ranks images by Euclidean distance between learned embeddings.
The model is a custom convolutional network trained from scratch with triplet loss.
Model quality has **not** been established on a representative held-out benchmark.

For the user-selected Clothing Dataset, see [training and local demo instructions](CLOTHING_TRAINING.md).

## Setup

Use **Python 3.10 or 3.11** in a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are pinned in one place: [requirements.txt](requirements.txt).
OpenCV 4.8.1 is used with NumPy 1.24.3 and TensorFlow 2.13; the previous
OpenCV 4.12 pin required an incompatible NumPy version. TensorFlow 2.13
does not support Python 3.12 or newer. CPU training can be slow; no latency
or training-time guarantees are claimed.

## Label and audit your data

Place JPG/JPEG/PNG images directly in `static/dataset/`. Supply real semantic
labels with filenames such as `shirt_001.jpg`, `pants-002.jpg`, or
`shirt (12).jpg`. For arbitrary filenames, supply a CSV:

```csv
filename,label
1163.jpg,shirt
1525.jpg,pants
```

The CSV must contain exactly one row for every image. Labels determine what
counts as relevant during training and evaluation, so inspect them manually.
**Do not use `rename_images.py` to generate training labels:** that legacy utility
distributes files arbitrarily and cannot determine what is in an image.
Numeric filename ranges are no longer treated as categories.

```bash
python train_triplet.py --audit-only
# With a CSV:
python train_triplet.py --labels-csv labels.csv --audit-only
```

The audit removes byte-identical copies before splitting and rejects conflicting
labels. Each category needs enough unique images for at least two images in each
of train, validation, and test (at least six per category with default fractions).
No rare category is silently dropped. Review resized/cropped duplicates and
keep related product, subject, or capture-session images in the same split before
using this pipeline; byte hashes do not detect perceptual duplicates.

The bundled 97 files are demonstration data, with categories too sparse for
these splits. Add correctly labeled data before training. Default training also
requires at least **100 unique test images**. This is a project guardrail, not a
statistical sample-size guarantee; choose a representative dataset and adequate
per-category coverage for the intended use.

## Train and evaluate

Run commands from the repository root:

```bash
python train_triplet.py --epochs 10 --batch-size 16 --steps-per-epoch 20
python evaluate_triplet.py
python extract_features_triplet.py
python app_triplet.py
```

Open [localhost:5000](http://localhost:5000).

Training uses a seeded, per-category image split (approximately 60% train,
20% validation, 20% test). Triplets never cross partitions. Validation uses a
fixed set of triplets from validation images; the best validation-loss checkpoint
is saved as `triplet_base_final.h5`. Test images are never used to train or
select the checkpoint.

Outputs:

- `dataset_split.json`: exact filenames, labels, content hashes, seed, counts,
  and the selected model's hash.
- `training_history.json` and `training_history.png`: train/validation loss.
- `evaluation_report.json`: Precision@k, Recall@k, Hit Rate@k, full-ranking mAP,
  per-category mAP, macro-category mAP, effective k, and query-bootstrap intervals.
- `triplet_features.npy` and `triplet_images.npy`: the search index.

The evaluator uses **every test image** as a query against the held-out test
gallery, removes the query's own entry, and treats same-category images as
relevant. Precision divides by `min(k, gallery_size - 1)`; recall divides by
the query's total relevant peers. AP averages precision at every relevant rank.
Queries share a gallery: bootstrap intervals are descriptive and do not capture
variation across training runs or datasets. The evaluator verifies file hashes,
partition disjointness, and the model hash; stale or mismatched artifacts fail.

Use validation data for threshold/hyperparameter decisions; reserve the test set
for the final assessment. Repeat training with multiple seeds and report category
coverage, dataset provenance, labeling quality, and relevance to the deployment
domain. Category retrieval is only a proxy for human-rated visual similarity.
No measured accuracy is included in this repository.

For pipeline checks on a smaller but structurally valid dataset:

```bash
python train_triplet.py --smoke-test --epochs 1 --steps-per-epoch 1 --validation-steps 1
python evaluate_triplet.py
```

Smoke mode still requires valid partitions. Its report is marked
`smoke_test_only` and must not be presented as a model benchmark.
`--min-test-images`, `--validation-fraction`, `--test-fraction`, and
`--seed` are configurable. A lower count does not establish reliability.
Pre-existing checkpoints have no split manifest and cannot substantiate held-out
performance; retrain with this workflow.

Feature extraction indexes the full dataset for the interactive demo. The
evaluator operates separately on the held-out test partition.

## Architecture and source files

The modules imported by training are committed at the repository root:

| File | Responsibility |
| --- | --- |
| [triplet_model.py](triplet_model.py) | Shared CNN, triplet network, loss and optimizer |
| [triplet_data_generator.py](triplet_data_generator.py) | Triplets within one explicit partition |
| [dataset_utils.py](dataset_utils.py) | Labels, deduplication, splits and fingerprint validation |
| [train_triplet.py](train_triplet.py) | Training and validation checkpoint selection |
| [evaluate_triplet.py](evaluate_triplet.py) | Held-out retrieval report |
| [retrieval_metrics.py](retrieval_metrics.py) | Ranking metrics and descriptive intervals |
| [extract_features_triplet.py](extract_features_triplet.py) | Search embeddings |
| [app_triplet.py](app_triplet.py) | Flask search, gallery and stats endpoints |
| [templates/index.html](templates/index.html) | Served web UI |
| [index.html](index.html) | Standalone UI copy |

The backbone has convolutional blocks with 64, 128, 256, and 512 channels,
batch normalization, pooling/dropout, global average pooling, dense layers of
512 and 256 units, and a **128-dimensional L2-normalized output**.
It is not ResNet and does not use pretrained weights. Inputs are 224×224 RGB
images scaled to [0, 1]. All three branches share weights. Training minimizes:

```text
mean(max(||anchor - positive||² - ||anchor - negative||² + 0.2, 0))
```

The normalization layer is registered for model serialization. Architecture,
forward-pass, triplet-loss, one-batch training, and save/reload checks run in CI.

## Score interpretation and API

For unit-length embeddings, Euclidean distance lies in [0, 2]. Search returns:

```text
score = clip(1 - distance / 2, 0, 1)
```

This is a **distance-based ranking score**, not a calibrated probability,
confidence, or percentage likelihood of relevance. A score of 0.90 does not mean
a 90% chance of similarity. No probability calibration has been fitted.
The UI displays a decimal score, and its threshold is in the same [0, 1] units.

`POST /search` accepts multipart fields `image`, `num_results` (5–50),
and `threshold` (0–1). For example, a result at distance 0.18 has score 0.91:

```json
{
  "success": true,
  "query_image": "/static/uploads/query.jpg",
  "results": [{
    "image": "/dataset/shirt_002.jpg",
    "name": "shirt_002.jpg",
    "score": 0.91,
    "similarity": 0.91,
    "distance": 0.18
  }],
  "total_matches": 1,
  "method": "Triplet Network with Euclidean Distance",
  "score_type": "distance_based_similarity",
  "score_description": "Distance score: clip(1 - Euclidean distance / 2, 0, 1); not a probability",
  "calibrated_probability": false
}
```

`similarity` is retained as a compatibility alias for `score`.
`GET /stats` reports artifact availability and score semantics.
`GET /gallery` lists indexed images. Search uses an exact scan, not an HNSW index.

## Verification and troubleshooting

```bash
python -m unittest discover -s tests -v
```

The pure data/metric tests need NumPy; TensorFlow-dependent tests require the
complete requirements. CI installs dependencies on Python 3.10 and 3.11.

- **Not enough images:** add genuinely labeled images per category; do not invent labels.
- **Model missing:** complete training before feature extraction or search.
- **Embeddings missing:** run `python extract_features_triplet.py` after training.
- **Dataset/model changed:** retrain and regenerate the manifest and embeddings together.
- **Out of memory:** reduce `--batch-size`; the custom CNN can be expensive on CPU.
- **Port in use:** change port 5000 at the end of `app_triplet.py`.

The bundled Flask entry point enables debug mode for local development.

## Acknowledgments

Thanks to Intel for the project opportunity.
