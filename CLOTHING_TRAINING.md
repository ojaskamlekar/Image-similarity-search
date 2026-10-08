# Training on the Clothing Dataset

Source: [alexeygrigorev/clothing-dataset](https://github.com/alexeygrigorev/clothing-dataset),
commit `7ed86a9cc56763b6acb35e41cb35184e43a5d218`. The source declares CC0-1.0.
Images and trained artifacts stay out of Git; the preparation report preserves
source provenance, labels, hashes, contributor IDs, and every exclusion.

## Data preparation

Download the source outside this repository:

```bash
git clone --depth 1 https://github.com/alexeygrigorev/clothing-dataset.git ../clothing-data
python prepare_clothing_dataset.py --source ../clothing-data --destination data/clothing
python train_triplet.py --image-folder data/clothing/images --labels-csv data/clothing/labels.csv --audit-only
```

The importer verifies image files and retains the original semantic labels.
It excludes ambiguous labels (Not sure, Other, Skip, unknown), byte-identical
duplicates, missing/unreadable files, and categories with fewer than 30 unique
images. It refuses to overwrite an existing prepared directory.

For the pinned source, preparation retained **5,073 images in 16 categories**.
The seed-42 contributor-disjoint split has **3,208 training, 978 validation, and 887 test images**.
Source metadata has 5,403 rows; all 330 exclusions are recorded explicitly.
The dataset is imbalanced and the sampler chooses categories uniformly.

## GPU environment

The existing app requirements target TensorFlow 2.13. The isolated Linux/WSL
training environment instead pins TensorFlow 2.15.1 with its pip CUDA dependencies
in `requirements-training-gpu.txt`; it preserves Keras 2 and the model interface.
Native Windows TensorFlow 2.13 cannot use the NVIDIA GPU.
No system CUDA or Python installation is changed by these setup scripts.

From an Ubuntu/WSL terminal at the repository root:

```bash
bash scripts/setup_training_wsl.sh
bash scripts/train_clothing_wsl.sh runs/clothing-20261009
```

Setup downloads an isolated Python 3.11 and several GB of TensorFlow/CUDA packages.
The preparation CSV includes a group column containing the source contributor ID.
Grouped splitting keeps each contributor entirely within one partition, and
searches seeded candidate allocations for approximate category balance.
The training script requires a detected GPU, enables GPU memory growth, uses the
existing 224x224 triplet CNN, and trains for 10 epochs with batch size 4,
400 sampled triplet batches per epoch, and 20 fixed validation batches.
It records runtime configuration and progress, chooses the lowest validation-loss
checkpoint, evaluates all held-out test images, and builds a full-dataset index.
These are initial experiment settings, not an optimality claim.

Each run saves:

- `triplet_base_final.h5`: best validation checkpoint.
- `run_config.json`, `dataset_split.json`: runtime settings and split/model hashes.
- `training_history.json`, `training_history.png`, and logs.
- `evaluation_report.json`: retrieval metrics and contributor overlap audit.
- `triplet_features.npy`, `triplet_images.npy`, `triplet_metadata.json`: search index and fingerprints.

Existing output models are protected; choose a new run directory for another run.
Training loss and checkpoint selection never use test images.

## Try the trained model

After the run and indexing finish:

```bash
bash scripts/run_clothing_app_wsl.sh --run-dir runs/clothing-20261009
```

Open [localhost:5000](http://localhost:5000). The launcher verifies the index's
model and image fingerprints, points the app at the selected artifacts, and serves
the prepared dataset without copying it over the demo images. Debug mode is off.
This is a local demo, not a production deployment.

## Interpretation

Same-category images count as relevant. These labels do not establish that two
photos show the same product. Splits are contributor-disjoint as well as image-disjoint.
Perceptual duplicates or related products across different contributor IDs may
still inflate results. The evaluation report records contributor overlap. Test metrics describe this
dataset and split, not general real-world accuracy.

The full dataset is indexed for the interactive demo; held-out evaluation uses
only the test partition and excludes each query's own entry.
