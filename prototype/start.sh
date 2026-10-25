#!/bin/sh
set -eu
PROTOTYPE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$PROTOTYPE_DIR/.."
if [ ! -x "$PROTOTYPE_DIR/.venv/bin/python" ]; then
    python3 -m venv "$PROTOTYPE_DIR/.venv"
fi
PYTHON_BIN="$PROTOTYPE_DIR/.venv/bin/python"
if ! "$PYTHON_BIN" -c 'import catboost, sklearn, sklift, scipy, pandas, numpy' 2>/dev/null; then
    "$PYTHON_BIN" -m pip install -r "$PROTOTYPE_DIR/requirements.txt"
fi
"$PYTHON_BIN" -u "$PROTOTYPE_DIR/run.py"
"$PYTHON_BIN" -m unittest discover -s "$PROTOTYPE_DIR/tests" -v
exec "$PYTHON_BIN" -u "$PROTOTYPE_DIR/app.py" --port "${PORT:-8765}"
