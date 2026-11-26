#!/bin/bash
# Installing BM4K Autofocus: Python environment, dependencies, Dock apps and services.
set -e
cd "$(dirname "$0")"

if [ "$(uname)" != "Darwin" ]; then
  echo "macOS is required." >&2; exit 1
fi

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3 not found. Install it (https://www.python.org/downloads/ or 'xcode-select --install')." >&2
  exit 1
fi
"$PY" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] < (3, 13) else 1)' || {
  echo "Python 3.9–3.12 is required." >&2; exit 1; }

if ! command -v clang >/dev/null 2>&1; then
  echo "Need Xcode command line tools (clang): xcode-select --install" >&2; exit 1
fi

[ -d .venv ] || "$PY" -m venv .venv
# pip first: pyobjc does not build on old pip
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

read -r -p "Start the service automatically at login? [y/N] " ans
case "$ans" in
  [yY]*) .venv/bin/python build_app.py --autostart ;;
  *)     .venv/bin/python build_app.py ;;
esac

echo
echo "Done. Open 'BM4K Autofocus.app' (drag it to the Dock if you like)."
