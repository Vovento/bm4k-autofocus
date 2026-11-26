# Contributing

Hardware test reports, lens profiles, bug fixes and documentation improvements are welcome. Please open an issue before claiming support for a new camera or lens model.

For a lens report, include:

1. Camera model and firmware version, lens and adapter model, and macOS version.
2. Whether the camera appears as a USB video device and whether the BLE dashboard connects.
3. Whether Camera Control reports lens name (`12.9`), focal length (`12.11`) and focus distance (`12.12`).
4. Whether precise tracking or centre-frame camera AF worked; note lighting, subject distance and any hunting.
5. A `profiles/<slug>.json` file **only after** checking its curve on your lens. State how you measured it.

Before a pull request, run:

```bash
.venv/bin/python tests/test_protocol.py
.venv/bin/python tests/test_focusmodel.py
.venv/bin/python tests/test_video_devices.py
```

Keep private material out of issues and pull requests: `focus_model.json`, `settings.json`, `trace*.json`, snapshots, BLE identifiers, personal paths and Blackmagic's copyrighted PDFs. Profiles can be shared; user learning data and recordings should stay local.

The MIT license covers the project code. A contributed profile should be your own measurement, and submitting it means you agree to release that contribution under the project license.
