#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$PWD/.venv-wsl/bin/python"
export LD_LIBRARY_PATH="$("$PY" -c 'import site,glob; print(":".join(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")))'):/usr/lib/wsl/lib"
exec "$PY" run_clothing_app.py "$@"
