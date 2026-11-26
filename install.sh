#!/bin/bash
# Установка BM4K Autofocus: окружение Python, зависимости, приложения для Dock и службы.
set -e
cd "$(dirname "$0")"

if [ "$(uname)" != "Darwin" ]; then
  echo "Нужна macOS / macOS is required." >&2; exit 1
fi

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3 не найден. Установите его (https://www.python.org/downloads/ или 'xcode-select --install')." >&2
  echo "Python 3 not found." >&2; exit 1
fi
"$PY" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] < (3, 13) else 1)' || {
  echo "Нужен Python 3.9–3.12 / Python 3.9–3.12 is required." >&2; exit 1; }

if ! command -v clang >/dev/null 2>&1; then
  echo "Нужны инструменты командной строки Xcode (clang): xcode-select --install" >&2; exit 1
fi

[ -d .venv ] || "$PY" -m venv .venv
# сначала pip: на старом pip pyobjc не собирается
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

read -r -p "Запускать службу автоматически при входе в систему? / Start at login? [y/N] " ans
case "$ans" in
  [yY]*) .venv/bin/python build_app.py --autostart ;;
  *)     .venv/bin/python build_app.py ;;
esac

echo
echo "Готово. Откройте 'BM4K Autofocus.app' (можно перетащить в Dock)."
echo "Done. Open 'BM4K Autofocus.app' (drag it to the Dock if you like)."
