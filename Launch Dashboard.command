#!/bin/bash
# Double-click to start the camera dashboard in Terminal.app, which can access Bluetooth and the camera.
cd "$(dirname "$0")"
exec .venv/bin/python dashboard.py "$@"
