"""Запуск службы при входе в систему: LaunchAgent, который через `open` стартует приложение BM4K Autofocus Service."""
import os
import plistlib
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = "com.bm4k.autofocus.service"
PLIST = os.path.expanduser(f"~/Library/LaunchAgents/{LABEL}.plist")
SERVICE_APP = os.path.join(HERE, "BM4K Autofocus Service.app")


def enabled():
    return os.path.exists(PLIST)


def set_enabled(on):
    """Включить или выключить автозапуск. Уже работающую службу не трогает."""
    target = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", f"{target}/{LABEL}"], capture_output=True)
    if not on:
        if os.path.exists(PLIST):
            os.remove(PLIST)
        return
    os.makedirs(os.path.dirname(PLIST), exist_ok=True)
    with open(PLIST, "wb") as f:
        plistlib.dump({"Label": LABEL, "ProgramArguments": ["/usr/bin/open", "-g", SERVICE_APP],
                       "RunAtLoad": True}, f)
    subprocess.run(["launchctl", "bootstrap", target, PLIST], check=True, capture_output=True)
