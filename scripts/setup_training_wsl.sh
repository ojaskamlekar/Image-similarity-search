#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.uv-cache-wsl"
export UV_PYTHON_INSTALL_DIR="$PWD/.python-wsl"
mkdir -p .tools
if [ ! -x .tools/uv-x86_64-unknown-linux-gnu/uv ]; then
  curl -fL --retry 3 https://github.com/astral-sh/uv/releases/download/0.11.28/uv-x86_64-unknown-linux-gnu.tar.gz -o .tools/uv.tar.gz
  tar -xzf .tools/uv.tar.gz -C .tools
fi
UV="$PWD/.tools/uv-x86_64-unknown-linux-gnu/uv"
"$UV" venv --python 3.11 .venv-wsl
"$UV" pip install --python .venv-wsl/bin/python -r requirements-training-gpu.txt
