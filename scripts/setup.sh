#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
git lfs version >/dev/null
git lfs install --local
if git remote get-url origin >/dev/null 2>&1; then git lfs pull; fi
git submodule update --init --recursive
unset PYTHONPATH
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
echo 'Ready: .venv/bin/python examples/reach_sim.py'
