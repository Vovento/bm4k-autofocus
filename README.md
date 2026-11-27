# BM4K Autofocus — face-tracking autofocus for a BMPCC 4K webcam

Use a **Blackmagic Pocket Cinema Camera 4K as a webcam on macOS**, with face-tracking autofocus and a local camera control dashboard. The camera supplies the video feed over USB-C; BM4K Autofocus adjusts the lens over Bluetooth while you use that feed in Zoom or another video app.

BM4K Autofocus watches the Blackmagic video feed with Apple Vision, tracks a face, evaluates sharpness, and sends focus commands over Bluetooth Low Energy. The browser dashboard shows the camera preview and controls focus, iris, shutter angle, ISO, white balance, recording and microphone level.

**This is not a virtual webcam or a video forwarding app.** Select the Pocket Cinema Camera itself as the video source in Zoom or another app; BM4K Autofocus runs alongside it to control the camera. The dashboard and face tracking use the same Blackmagic video feed. Built-in Mac, iPhone and generic capture cameras are rejected as autofocus sources.

> **Hardware tested:** BMPCC 4K, Sigma 18–35 mm f/1.8 Art EF, Metabones Speed Booster 0.64× EF–MFT, macOS. With this setup, focusing after motion stops was observed at roughly 1–1.5 seconds in tests. Results vary with light, distance, lens and camera settings. Other lenses and BMPCC 6K models are **not yet verified**; contributions and test results are welcome.

Unofficial community project. Not affiliated with or endorsed by Blackmagic Design. Blackmagic and Pocket Cinema Camera are trademarks of their respective owner.

## Set up your BMPCC 4K webcam

Choose either installation path below. Both end with selecting **Pocket Cinema Camera** as the video source in your meeting or streaming app. BM4K Autofocus controls the physical camera's focus; it does not create a separate webcam device.

### Option A: ask your coding agent to install it

Give your coding agent the repository link and this instruction:

> Install [BM4K Autofocus](https://github.com/Vovento/bm4k-autofocus) on my Mac using its README and `install.sh`. Check that my Python version is supported, install the required dependencies, and launch the dashboard. Ask me to enable Bluetooth on the Mac and BMPCC 4K, connect the camera over USB-C in webcam mode, and approve any macOS Camera/Bluetooth permission prompts. Verify that the dashboard detects the Blackmagic video feed and Bluetooth link; do not select my MacBook or iPhone camera. Ask whether I want the service to start at login. Explain how to select **Pocket Cinema Camera** as the webcam in my video app.

Stay with the agent for physical camera settings and macOS permission prompts. The agent cannot turn on the camera's Bluetooth or approve those prompts for you.

### Option B: install it yourself

1. On the **BMPCC 4K**, enable Bluetooth in **Menu → Setup → Bluetooth**. On the **Mac**, turn Bluetooth on. Disconnect Blackmagic's phone control app if it is connected; the camera accepts one BLE controller at a time.
2. Connect the camera to the Mac with a USB-C **data** cable. Enable the camera's webcam/USB video output and confirm that macOS lists **Pocket Cinema Camera** as a camera. Keep the camera powered on.
3. Ensure you have **macOS, Python 3.9–3.12, and Xcode Command Line Tools**. If needed, install the tools with `xcode-select --install`. Check Python with `python3 --version`.
4. In Terminal, run:

   ```bash
   git clone https://github.com/Vovento/bm4k-autofocus.git
   cd bm4k-autofocus
   ./install.sh
   ```

5. When the installer asks, choose whether to start the service automatically at login. Then open **BM4K Autofocus.app** from the cloned folder. It opens the dashboard at [http://127.0.0.1:8765](http://127.0.0.1:8765).
6. On first launch, grant **Camera** and **Bluetooth** access to **BM4K Autofocus Service** in macOS **System Settings → Privacy & Security** when prompted. Complete any camera pairing prompt on the Mac.
7. In the dashboard, wait for the Blackmagic Bluetooth link and camera preview, then enable **Follow face**. The app will use only the Blackmagic video device for tracking; it will not switch to the MacBook or iPhone camera.
8. In Zoom, OBS, or your other video app, select **Pocket Cinema Camera** as the **camera/webcam**. Keep BM4K Autofocus running alongside it for autofocus. Select your preferred microphone separately in that app.

If you have another lens or a BMPCC 6K, read [Lens profiles and compatibility](#lens-profiles-and-compatibility) before relying on autofocus; only the hardware combination listed above has been tested.

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

## Other ways to run

The installer creates `.venv`, installs the pinned Python dependencies, and builds the Dock app and background service.

For a terminal-only start, use `.venv/bin/python dashboard.py` or double-click `Launch Dashboard.command`. For an interface demo without hardware, run `.venv/bin/python dashboard.py --sim` while the service is stopped (both use port 8765).

The service log is at `~/Library/Logs/BM4K Autofocus.log`.

## Lens profiles and compatibility

The included [`profiles/64x-sigma-dc-18-35-1-7-hsm.json`](profiles/64x-sigma-dc-18-35-1-7-hsm.json) contains a measured focus-position-to-distance curve for the tested Sigma 18–35/Metabones setup. It is lens-specific and is available to everyone who installs the project.

On first use, precise tracking has no local learning data yet. With a face visible, it uses the camera's one-shot AF to establish an initial focus point, then learns how face size and focal length relate to the camera's reported focus distance. As it runs, it also adjusts for focus-scale drift and records sharpness references. These numeric adjustments are saved locally in `focus_model.json` for each lens; they can vary with the subject, distance, camera setup and shooting conditions such as lighting. The file is excluded from Git so another user starts with the shared lens profile and builds adjustments for their own setup. No face images are stored in this file.

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
