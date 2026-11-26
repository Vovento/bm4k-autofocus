"""Сборка приложений macOS и автозапуска.

  python build_app.py              нарисовать иконку и собрать два приложения
  python build_app.py --autostart  то же + включить автозапуск при входе в систему
  python build_app.py --no-autostart  выключить автозапуск
  python build_app.py --opener-only   обновить только приложение для Dock (службу не пересобирать)

Получается:
  BM4K Autofocus.app          — то, что кладётся в Dock: запускает службу (если ещё не запущена) и открывает окно
  BM4K Autofocus Service.app  — фоновая служба без значка в Dock: Bluetooth, автофокус и сервер на 127.0.0.1:8765
Службе нужны доступы к Bluetooth и камере, поэтому она оформлена как приложение с описаниями причин (Info.plist).
"""
import math
import os
import plistlib
import shutil
import subprocess
import sys

import numpy as np
import autostart
from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
APP_NAME = "BM4K Autofocus"
SERVICE_NAME = "BM4K Autofocus Service"
BUNDLE_ID = "com.bm4k.autofocus"
URL = "http://127.0.0.1:8765"


# ---------- иконка ----------

def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))


def radial(size, center, radius, inner, outer):
    """Круговой градиент от inner в центре до outer на радиусе."""
    y, x = np.mgrid[0:size, 0:size]
    t = np.clip(np.hypot(x - center[0], y - center[1]) / radius, 0, 1)[..., None]
    arr = np.array(inner)[None, None] * (1 - t) + np.array(outer)[None, None] * t
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def diagonal(size, c0, c1):
    y, x = np.mgrid[0:size, 0:size]
    t = ((x + y) / (2 * size))[..., None]
    arr = np.array(c0)[None, None] * (1 - t) + np.array(c1)[None, None] * t
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def circle_mask(size, center, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)
    return m


def draw_icon(path, size=2048):
    s = size / 1024
    c = (size // 2, size // 2)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # тело иконки: скруглённый квадрат с тёмным градиентом
    body = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    grad = Image.new("RGB", (size, size))
    gd = ImageDraw.Draw(grad)
    for yy in range(size):
        gd.line([(0, yy), (size, yy)], fill=lerp((44, 48, 58), (13, 15, 18), yy / size))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((100 * s, 100 * s, 924 * s, 924 * s), radius=190 * s, fill=255)
    body.paste(grad, (0, 0), mask)
    img = Image.alpha_composite(img, body)

    # мягкая тень под объективом
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse((c[0] - 335 * s, c[1] - 315 * s, c[0] + 335 * s, c[1] + 355 * s), fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(28 * s)))

    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    # корпус объектива (металл), тёмное кольцо, тонкий светлый ободок, стекло
    layer.paste(diagonal(size, (150, 156, 168), (42, 46, 54)).convert("RGBA"), (0, 0), circle_mask(size, c, 330 * s))
    layer.paste(Image.new("RGBA", (size, size), (16, 18, 22, 255)), (0, 0), circle_mask(size, c, 292 * s))
    layer.paste(diagonal(size, (96, 104, 118), (28, 31, 37)).convert("RGBA"), (0, 0), circle_mask(size, c, 268 * s))
    glass = radial(size, (c[0] - 40 * s, c[1] - 50 * s), 260 * s, (46, 86, 150), (6, 9, 18)).convert("RGBA")
    layer.paste(glass, (0, 0), circle_mask(size, c, 250 * s))
    # внутренние кольца стекла
    d = ImageDraw.Draw(layer)
    for r, col in ((200, (84, 120, 190, 90)), (150, (120, 90, 200, 70))):
        d.ellipse((c[0] - r * s, c[1] - r * s, c[0] + r * s, c[1] + r * s), outline=col, width=int(5 * s))
    # лепестки диафрагмы: шестиугольник и «звезда» из линий
    hexa = [(c[0] + 112 * s * math.cos(math.radians(60 * i + 30)), c[1] + 112 * s * math.sin(math.radians(60 * i + 30)))
            for i in range(6)]
    d.polygon(hexa, fill=(7, 12, 28, 255), outline=(140, 160, 205, 200), width=int(7 * s))
    for i in range(6):
        a = math.radians(60 * i + 30)
        d.line([(c[0] + 112 * s * math.cos(a), c[1] + 112 * s * math.sin(a)),
                (c[0] + 205 * s * math.cos(a + 0.42), c[1] + 205 * s * math.sin(a + 0.42))],
               fill=(150, 175, 220, 110), width=int(4 * s))
    img = Image.alpha_composite(img, layer)

    # блики
    hl = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hl)
    hd.ellipse((c[0] - 170 * s, c[1] - 190 * s, c[0] - 30 * s, c[1] - 90 * s), fill=(255, 255, 255, 80))
    hd.ellipse((c[0] + 95 * s, c[1] + 70 * s, c[0] + 140 * s, c[1] + 112 * s), fill=(255, 255, 255, 120))
    img = Image.alpha_composite(img, hl.filter(ImageFilter.GaussianBlur(5 * s)))

    # угловые рамки автофокуса (янтарные)
    af = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ad = ImageDraw.Draw(af)
    amber, half, arm, th = (255, 176, 32, 255), 352 * s, 105 * s, 26 * s
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cx, cy = c[0] + sx * half, c[1] + sy * half
        ad.rounded_rectangle((min(cx, cx - sx * arm), min(cy, cy) - th / 2, max(cx, cx - sx * arm), max(cy, cy) + th / 2),
                             radius=th / 2, fill=amber)
        ad.rounded_rectangle((min(cx, cx) - th / 2, min(cy, cy - sy * arm), max(cx, cx) + th / 2, max(cy, cy - sy * arm)),
                             radius=th / 2, fill=amber)
    img = Image.alpha_composite(img, af)

    # обрезать всё по форме тела иконки
    img.putalpha(ImageChops.multiply(img.getchannel("A"), Image.new("L", (size, size), 255)))
    img = img.resize((1024, 1024), Image.LANCZOS)
    img.save(path)
    return img


def make_icns(png_path, icns_path):
    iconset = os.path.join(ASSETS, "AppIcon.iconset")
    shutil.rmtree(iconset, ignore_errors=True)
    os.makedirs(iconset)
    src = Image.open(png_path)
    for base in (16, 32, 128, 256, 512):
        for scale, suffix in ((1, ""), (2, "@2x")):
            src.resize((base * scale, base * scale), Image.LANCZOS).save(
                os.path.join(iconset, f"icon_{base}x{base}{suffix}.png"))
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns_path], check=True)
    shutil.rmtree(iconset, ignore_errors=True)


# ---------- приложения ----------

OPENER = '''#!/bin/bash
# Значок в Dock: запускает фоновую службу, если она ещё не работает, и открывает окно интерфейса.
DIR="__DIR__"
URL="__URL__"
if ! /usr/bin/curl -s -m 1 -o /dev/null "$URL/"; then
  /usr/bin/open -g "$DIR/__SERVICE__.app"
  for i in $(seq 1 40); do
    /usr/bin/curl -s -m 1 -o /dev/null "$URL/" && break
    sleep 0.5
  done
fi
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  "$CHROME" --app="$URL" >/dev/null 2>&1 &
else
  /usr/bin/open "$URL"
fi
'''

SERVICE = '''#!/bin/bash
# Фоновая служба: держит дашборд запущенным и перезапускает его, если он упал.
DIR="__DIR__"
LOG="$HOME/Library/Logs/__APP__.log"
[ -f "$LOG" ] && [ "$(/usr/bin/stat -f%z "$LOG")" -gt 5000000 ] && : > "$LOG"
cd "$DIR" || exit 1
while true; do
  echo "--- $(date '+%F %T') запуск" >> "$LOG"
  "$DIR/.venv/bin/python" dashboard.py --no-browser >> "$LOG" 2>&1
  sleep 5
done
'''


# Запускатель: настоящий исполняемый файл приложения. Скрипт сам по себе не годится: macOS тогда считает
# «ответственным» за доступ к Bluetooth/камере bash, а не наше приложение, и запросов доступа не показывает.
STUB_C = r'''
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
static pid_t child = 0;
static void forward(int sig) { if (child > 0) kill(-child, sig); }
int main(void) {
    char path[4096]; uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) return 1;
    char script[4200]; snprintf(script, sizeof(script), "%s.sh", path);
    posix_spawnattr_t attr; posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP); posix_spawnattr_setpgroup(&attr, 0);
    char *args[] = {"/bin/bash", script, NULL};
    if (posix_spawn(&child, "/bin/bash", NULL, &attr, args, environ) != 0) return 1;
    signal(SIGTERM, forward); signal(SIGINT, forward); signal(SIGHUP, forward);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) { }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
'''


def make_app(name, script, icns, url_scheme=None):
    app = os.path.join(HERE, f"{name}.app")
    shutil.rmtree(app, ignore_errors=True)
    os.makedirs(os.path.join(app, "Contents", "MacOS"))
    os.makedirs(os.path.join(app, "Contents", "Resources"))
    shutil.copy(icns, os.path.join(app, "Contents", "Resources", "AppIcon.icns"))
    exe = os.path.join(app, "Contents", "MacOS", name)
    with open(exe + ".sh", "w") as f:
        f.write(script.replace("__DIR__", HERE).replace("__URL__", URL)
                .replace("__SERVICE__", SERVICE_NAME).replace("__APP__", APP_NAME))
    os.chmod(exe + ".sh", 0o755)
    stub_c = os.path.join(ASSETS, "stub.c")
    with open(stub_c, "w") as f:
        f.write(STUB_C)
    subprocess.run(["clang", "-O2", "-o", exe, stub_c], check=True)
    ident = BUNDLE_ID + (".service" if name == SERVICE_NAME else "")
    info = {
        "CFBundleName": name, "CFBundleDisplayName": name, "CFBundleIdentifier": ident,
        "CFBundleExecutable": name, "CFBundleIconFile": "AppIcon", "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0", "CFBundleVersion": "1", "LSMinimumSystemVersion": "11.0",
        "LSUIElement": True,  # без значка в Dock во время работы
        "NSBluetoothAlwaysUsageDescription": "BM4K Autofocus управляет камерой Blackmagic по Bluetooth.",
        "NSCameraUsageDescription": "BM4K Autofocus смотрит на картинку камеры, чтобы следить за лицом и наводить фокус.",
    }
    if url_scheme:  # ссылка bm4k://start со страницы пульта запускает это приложение
        info["CFBundleURLTypes"] = [{"CFBundleURLName": name, "CFBundleURLSchemes": [url_scheme]}]
    with open(os.path.join(app, "Contents", "Info.plist"), "wb") as f:
        plistlib.dump(info, f)
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", app], check=True, capture_output=True)
    return app


def main():
    os.makedirs(ASSETS, exist_ok=True)
    png, icns = os.path.join(ASSETS, "icon.png"), os.path.join(ASSETS, "AppIcon.icns")
    if "--opener-only" in sys.argv:  # службу не трогаем: пересборка сбросила бы выданные ей доступы
        app = make_app(APP_NAME, OPENER, icns, url_scheme="bm4k")
        subprocess.run(["/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework"
                        "/Support/lsregister", "-f", app], capture_output=True)
        print(f"Обновлено приложение для Dock:\n  {app}")
        return
    draw_icon(png)
    make_icns(png, icns)
    app = make_app(APP_NAME, OPENER, icns, url_scheme="bm4k")
    service = make_app(SERVICE_NAME, SERVICE, icns)
    print(f"Собрано:\n  {app}\n  {service}")
    if "--autostart" in sys.argv:
        autostart.set_enabled(True)
        print(f"Автозапуск включён: {autostart.PLIST}")
    elif "--no-autostart" in sys.argv:
        autostart.set_enabled(False)
        print("Автозапуск выключен.")


if __name__ == "__main__":
    main()
