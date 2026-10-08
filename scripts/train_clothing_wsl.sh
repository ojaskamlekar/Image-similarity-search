#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv-wsl/bin/python"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TF_CPP_MIN_LOG_LEVEL=1
export TF_NUM_INTRAOP_THREADS=6
export TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS=6
export LD_LIBRARY_PATH="$("$PY" -c 'import site,glob; print(":".join(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")))'):/usr/lib/wsl/lib"
export PATH="$PWD/.venv-wsl/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$PATH"
RUN="${1:-runs/clothing-20261009}"
mkdir -p "$RUN"
"$PY" -c 'import tensorflow as tf; print("TensorFlow", tf.__version__); print("GPU", tf.config.list_physical_devices("GPU")); assert tf.config.list_physical_devices("GPU"), "GPU unavailable"'
"$PY" train_triplet.py --image-folder data/clothing/images --labels-csv data/clothing/labels.csv --epochs 10 --batch-size 4 --steps-per-epoch 400 --validation-steps 20 --output-dir "$RUN" --require-gpu 2>&1 | tee "$RUN/training.log"
"$PY" evaluate_triplet.py --manifest "$RUN/dataset_split.json" --model "$RUN/triplet_base_final.h5" --image-folder data/clothing/images --output "$RUN/evaluation_report.json" --batch-size 16 --preparation-report data/clothing/preparation_report.json 2>&1 | tee "$RUN/evaluation.log"
"$PY" extract_features_triplet.py --model "$RUN/triplet_base_final.h5" --image-folder data/clothing/images --output-prefix "$RUN/triplet" --batch-size 16 --no-visualization 2>&1 | tee "$RUN/extraction.log"
