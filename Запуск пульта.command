#!/bin/bash
# Двойной клик: запускает пульт камеры. Открывается в Terminal.app, у которого есть доступ к Bluetooth и камере.
cd "$(dirname "$0")"
exec .venv/bin/python dashboard.py "$@"
