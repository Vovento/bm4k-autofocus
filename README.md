# BM4K Autofocus

[Русская версия](README.ru.md)

Face tracking autofocus and a local camera control dashboard for the **Blackmagic Pocket Cinema Camera 4K** on macOS.

The camera supplies its own USB webcam feed to the Mac. BM4K Autofocus watches that feed with Apple Vision, tracks a face, evaluates sharpness, and sends focus commands to the camera over Bluetooth Low Energy. The browser dashboard shows the camera preview and controls focus, iris, shutter angle, ISO, white balance, recording and microphone level.

**This is not a virtual webcam or a video forwarding app.** Select the Pocket Cinema Camera itself as the video source in Zoom or another app; BM4K Autofocus runs alongside it to control the camera. The dashboard and face tracking use the same Blackmagic video feed. Built-in Mac, iPhone and generic capture cameras are rejected as autofocus sources.

> **Hardware tested:** BMPCC 4K, Sigma 18–35 mm f/1.8 Art EF, Metabones Speed Booster 0.64× EF–MFT, macOS. With this setup, focusing after motion stops was observed at roughly 1–1.5 seconds in tests. Results vary with light, distance, lens and camera settings. Other lenses and BMPCC 6K models are **not yet verified**; contributions and test results are welcome.

Unofficial community project. Not affiliated with or endorsed by Blackmagic Design. Blackmagic and Pocket Cinema Camera are trademarks of their respective owner.

## What it does

- **Precise tracking:** estimates subject distance from face size and focal length, uses the lens profile to command focus position, then verifies/refines it using image sharpness. The built-in one-shot AF is used for initial learning and fallback.
- **Camera AF fallback:** when a lens does not report focus distance, the app can use the camera's one-shot AF. That AF targets the centre of the frame, so off-centre faces may not be focused in fallback mode.
- **Local dashboard:** MJPEG preview, focus status and controls. It binds to `127.0.0.1:8765`; it is not exposed to the network.
- **Reconnects:** the background service retries Bluetooth and video capture when the camera is disconnected or restarted. The first UVC open can occasionally take another attempt.

## Requirements

- macOS, Python 3.9–3.12, and Xcode Command Line Tools (`xcode-select --install` for `clang`). The pinned dependencies have not been validated with Python 3.13+.
- Blackmagic Pocket Cinema Camera 4K with USB webcam output enabled and a USB-C data connection to the Mac.
- Bluetooth enabled on the Mac and camera (camera Menu → Setup → Bluetooth). Disconnect Blackmagic's phone control app; the camera accepts one BLE controller at a time.
- An electronically controllable lens/adapter, with its AF switch enabled where applicable. Precise tracking has only been tested with the Sigma/Metabones combination above.

The code recognises `Pocket Cinema Camera …` and `Blackmagic … Camera` video device names; this is **not** a claim that every such model supports the tested focus behaviour.

## Install and run

```bash
git clone https://github.com/Vovento/bm4k-autofocus.git
cd bm4k-autofocus
./install.sh
```

The installer creates `.venv`, installs the pinned Python dependencies, builds the Dock app and background service, and asks whether to start the service at login. Open `BM4K Autofocus.app` afterward. It opens the local dashboard at [http://127.0.0.1:8765](http://127.0.0.1:8765).

For a terminal-only start, use `.venv/bin/python dashboard.py` or double-click `Запуск пульта.command`. For an interface demo without hardware, run `.venv/bin/python dashboard.py --sim` while the service is stopped (both use port 8765).

On first launch, allow **Camera** and **Bluetooth** access for `BM4K Autofocus Service` in macOS System Settings → Privacy & Security. If the camera shows a pairing PIN, enter it on the Mac. The service log is at `~/Library/Logs/BM4K Autofocus.log`.

### Using it as a webcam

1. Connect the BMPCC to the Mac by USB-C and confirm that it appears as a webcam in macOS.
2. Open BM4K Autofocus, wait for both Bluetooth and video status to connect, and enable tracking AF in the dashboard.
3. In Zoom or your video app, choose **Pocket Cinema Camera** as the camera. The app adjusts lens focus; it does not replace the camera's video stream.

## Lens profiles and compatibility

The included [`profiles/64x-sigma-dc-18-35-1-7-hsm.json`](profiles/64x-sigma-dc-18-35-1-7-hsm.json) contains a measured focus-position-to-distance curve for the tested Sigma 18–35/Metabones setup. It is lens-specific. Personal learned data is stored separately in `focus_model.json` and is intentionally excluded from Git.

For a different lens, the precise mode attempts a one-time sweep of the focus range (about a minute) and writes a new `profiles/<lens>.json`. During that sweep the focus travels across its range. If the camera does not report distance for that lens, the profile is marked `no_distance` and tracking falls back to the camera's centre-frame AF. **This path has not been validated on other lenses:** inspect the result and treat it as experimental, rather than reusing the Sigma curve. Please share lens model, adapter, camera model, whether focus distance is reported, and a tested profile via an issue or pull request. Do not include `focus_model.json`, traces or images of people.

BMPCC 6K support is a possible community extension, not a tested feature. Device naming, USB video, BLE commands and lens feedback need real hardware validation.

## Troubleshooting

- **Camera video missing:** confirm USB-C data connection and webcam mode. The app never falls back to the MacBook/iPhone camera. It may restart itself if the UVC open hangs; check the service log.
- **Bluetooth disconnected:** check both Bluetooth switches, disconnect the phone controller, and reopen the app. If the service has retained a stale Bluetooth state, restart it from the dashboard or restart the service app.
- **Permissions missing:** grant Camera and Bluetooth to `BM4K Autofocus Service`; rebuilding the app may trigger new macOS permission prompts.
- **Port 8765 in use:** another copy may be running. Check with `lsof -nP -iTCP:8765 -sTCP:LISTEN`.
- **Installation fails on PyObjC:** use Python 3.9–3.12 and upgrade pip in the venv: `.venv/bin/python -m pip install --upgrade pip`.

## Project layout

| Path | Purpose |
| --- | --- |
| `dashboard.py`, `web/` | local control panel, WebSocket status and preview |
| `autofocus.py`, `focusmodel.py` | tracking logic and per-lens model |
| `video.py`, `bmcc.py` | Blackmagic video capture and Apple Vision face detection |
| `camera.py`, `protocol.py` | Bluetooth Camera Control link and packet encoding |
| `profiles/` | measured lens profiles; only the tested Sigma profile is shipped |
| `build_app.py`, `autostart.py`, `assets/` | macOS Dock and service app builders |
| `tests/` | protocol, model and camera selection checks |

Run the offline checks with `.venv/bin/python tests/test_protocol.py`, `.venv/bin/python tests/test_focusmodel.py` and `.venv/bin/python tests/test_video_devices.py`. Hardware behaviour requires an actual camera and lens.

See [CONTRIBUTING.md](CONTRIBUTING.md) for lens test reports and privacy guidelines. Released under the [MIT License](LICENSE).
