"""Дашборд управления BMPCC 4K: Bluetooth-управление камерой, следящий автофокус и превью в браузере.

Запуск из Terminal.app:  python dashboard.py [--cam 0] [--port 8765]
  --sim   симулятор вместо камеры (проверка интерфейса без железа)
  --no-browser   не открывать страницу (режим фоновой службы)
Страница открывается на http://127.0.0.1:8765 (доступна только с этого Mac).
"""
import argparse
import asyncio
import json
import math
import os
import statistics
import signal
import subprocess
import sys
import webbrowser

import cv2
from aiohttp import WSMsgType, web

import autostart
from autofocus import PROFILES, FaceAF
from camera import CameraLink, LinkError, SimLink
from protocol import packet
from video import VideoHub

HERE = os.path.dirname(os.path.abspath(__file__))
AUTO_IRIS_TARGET = 118   # целевая средняя яркость лица (0–255) в картинке камеры
AUTO_IRIS_GAMMA = 2.2    # яркость изображения ~ свет^(1/гамма)
SETTINGS = os.path.join(HERE, "settings.json")
DEFAULTS = {"af_enabled": True, "af_profile": "normal", "af_mode": "precise"}


def load_settings():
    try:
        with open(SETTINGS) as f:
            return {**DEFAULTS, **json.load(f)}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def build(action, v):
    """Команда интерфейса → пакет Camera Control (или None, если действие особое)."""
    table = {
        "iso": lambda: packet(1, 14, 3, [int(v)]),
        "shutter_angle": lambda: packet(1, 11, 3, [round(float(v) * 100)]),
        "wb": lambda: packet(1, 2, 2, [int(v["kelvin"]), int(v["tint"])]),
        "auto_wb": lambda: packet(1, 3, 0),
        "aperture_av": lambda: packet(0, 2, 128, [float(v)]),
        "focus_assign": lambda: packet(0, 0, 128, [float(v)]),  # служебное: абсолютный фокус 0..1
        "mic_level": lambda: packet(2, 0, 128, [float(v)]),
        "mic_input": lambda: packet(2, 4, 1, [int(v)]),
    }
    return table[action]() if action in table else None


def stop_process():
    """Завершить дашборд. Как служба (родитель — наш bash-скрипт): убить всю группу, чтобы скрипт не перезапустил его."""
    parent = subprocess.run(["ps", "-o", "command=", "-p", str(os.getppid())], capture_output=True, text=True).stdout
    if "BM4K Autofocus Service" in parent:
        os.killpg(os.getpgid(0), signal.SIGTERM)
    else:
        os.kill(os.getpid(), signal.SIGTERM)


class App:
    def __init__(self, link, grab):
        self.link, self.grab = link, grab
        self.settings = load_settings()
        self.af = FaceAF(link, grab)
        self.af.set_profile(self.settings["af_profile"])
        self.af.set_mode(self.settings["af_mode"])
        self.clients = set()
        self.ai_state = ""
        self._ai_busy = False

    def start(self):
        """Применить сохранённые настройки (вызывать уже внутри работающего цикла событий)."""
        if self.settings["af_enabled"]:
            self.af.set_enabled(True)

    def save(self, **changes):
        self.settings.update(changes)
        try:
            with open(SETTINGS, "w") as f:
                json.dump(self.settings, f, indent=1)
        except OSError as e:
            self.link.add_log(f"настройки не сохранены: {e}")

    def snapshot(self):
        s = self.link.state

        def first(k):
            v = s.get(k)
            return v[0] if v else None

        av = first("0.2")
        angle = first("1.11")
        wb = s.get("1.2") or []
        transport = s.get("10.1")
        d = {
            "iso": first("1.14"),
            "shutter_angle": angle / 100 if angle is not None else None,
            "shutter_speed": first("1.12"),
            "kelvin": wb[0] if wb else None,
            "tint": wb[1] if len(wb) > 1 else None,
            "av": av,
            "f": round(math.sqrt(2 ** av), 1) if av is not None else None,
            "f_text": s.get("12.10") if isinstance(s.get("12.10"), str) else None,
            "lens": s.get("12.9") if isinstance(s.get("12.9"), str) else None,
            "zoom_text": s.get("12.11") if isinstance(s.get("12.11"), str) else None,
            "rec_format": s.get("1.9"),
            "rec": bool(transport and transport[0] == 2),
            "mic": first("2.0"),
            "mic_input": first("2.4"),
            "video_mode": s.get("1.0"),
            "codec": s.get("10.0"),
        }
        af = dict(self.af.status, enabled=self.af.enabled, profile=self.af.profile)
        return {"link": {"status": self.link.status, "detail": self.link.detail, "name": self.link.name},
                "d": d, "raw": s, "af": af, "log": self.link.log[-14:],
                "video": self.grab.available, "video_error": self.grab.error,
                "autostart": autostart.enabled(), "auto_iris": self.ai_state}

    async def handle(self, data):
        action, v = data.get("a"), data.get("v")
        if action == "af_enable":
            self.af.set_enabled(bool(v))
            self.save(af_enabled=bool(v))
        elif action == "af_profile":
            self.af.set_profile(v)
            self.save(af_profile=v)
        elif action == "af_mode":
            self.af.set_mode(v)
            self.save(af_mode=v)
        elif action == "af_now":
            await self.af.af_now()
        elif action == "aperture_step":
            # сдвиг по 1/3 стопа через абсолютное значение (как ползунок): камера не принимает «смещение»
            av = (self.link.state.get("0.2") or [None])[0]
            if av is None:
                raise ValueError("камера ещё не прислала диафрагму")
            await self.link.send(packet(0, 2, 128, [max(0.0, min(16.0, av + float(v)))]))
        elif action == "auto_iris":
            if not self._ai_busy:
                asyncio.ensure_future(self.auto_iris())
        elif action == "sweep":
            asyncio.ensure_future(self.run_sweep(v or {}))
        elif action == "backlash":
            asyncio.ensure_future(self.run_backlash(v or {}))
        elif action == "save_trace":  # запись по кадрам за последние минуты для разбора: trace.json рядом с программой
            rows = list(self.af.trace)
            with open(os.path.join(HERE, "trace.json"), "w") as f:
                json.dump({"rows": rows, "fields": ["t", "size", "sharp", "x", "state", "cmds"]}, f)
            self.link.add_log(f"запись по кадрам сохранена: {len(rows)} кадров")
        elif action == "autostart":
            await asyncio.get_running_loop().run_in_executor(None, autostart.set_enabled, bool(v))
            self.link.touch()
        elif action == "quit":
            self.link.add_log("остановка службы по кнопке")
            asyncio.get_running_loop().call_later(0.4, stop_process)
        elif action == "record":
            current = list(self.link.state.get("10.1") or [0, 0])
            current[0] = 2 if v else 0
            await self.link.send(packet(10, 1, 1, current))
        else:
            pkt = build(action, v)
            if pkt is None:
                raise ValueError(f"неизвестное действие {action}")
            await self.link.send(pkt)
        self.link.add_log(f"→ {action} {v if v is not None else ''}")

    async def run_backlash(self, p):
        was = self.af.enabled
        self.af.set_enabled(False)
        try:
            await self.af.backlash_test(float(p.get("x", 0.8)))
        except LinkError as e:
            self.link.add_log(f"проверка люфта не выполнена: {e}")
        finally:
            if was:
                self.af.set_enabled(True)
            self.link.touch()

    async def run_sweep(self, p):
        was = self.af.enabled
        self.af.set_enabled(False)
        try:
            await self.af.sweep(float(p.get("lo", 0.7)), float(p.get("hi", 0.95)), float(p.get("step", 0.01)))
        except LinkError as e:
            self.link.add_log(f"перебор не выполнен: {e}")
        finally:
            if was:
                self.af.set_enabled(True)
            self.link.touch()

    # ---- программная авто-диафрагма ----
    def region_luma(self):
        """Средняя яркость лица (или центра кадра, если лица нет) в картинке камеры, 0–255."""
        frame, _ = self.grab.latest()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        box = self.af.status.get("box") if self.af.enabled else None
        if box:
            x, y, bw, bh = box
            x0, y0, x1, y1 = x + bw * 0.15, y + bh * 0.15, x + bw * 0.85, y + bh * 0.85
        else:
            x0, y0, x1, y1 = 0.3, 0.25, 0.7, 0.75
        crop = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()) if crop.size else None

    def _set_ai(self, text):
        self.ai_state = text
        self.link.add_log(f"авто диафрагма: {text}")

    async def auto_iris(self):
        """Подобрать диафрагму так, чтобы лицо имело целевую яркость. Несколько шагов с проверкой по картинке."""
        self._ai_busy = True
        self._set_ai("подбираю…")
        try:
            last_asked = None
            for _ in range(8):
                samples = []
                for _ in range(3):
                    luma = self.region_luma()
                    if luma is not None:
                        samples.append(luma)
                    await asyncio.sleep(0.08)
                av = (self.link.state.get("0.2") or [None])[0]
                if not samples or av is None:
                    self._set_ai("нет картинки или данных о диафрагме")
                    return
                if last_asked is not None and abs(last_asked - av) > 0.15:
                    self._set_ai("диафрагма упёрлась в предел объектива")
                    return
                # недостаток света в стопах; 0.8 — запас, чтобы не раскачиваться
                ev = 0.8 * AUTO_IRIS_GAMMA * math.log2(AUTO_IRIS_TARGET / max(statistics.median(samples), 1.0))
                if abs(ev) < 0.2:
                    self._set_ai(f"готово, f/{math.sqrt(2 ** av):.1f}")
                    return
                last_asked = max(0.0, min(16.0, av - ev))  # свет +1 стоп = AV −1
                await self.link.send(packet(0, 2, 128, [last_asked]))
                await asyncio.sleep(1.0)
            self._set_ai("готово")
        except LinkError as e:
            self._set_ai(f"не отправлено: {e}")
        finally:
            self._ai_busy = False
            self.link.touch()

    # ---- HTTP ----
    async def index(self, request):
        return web.FileResponse(os.path.join(HERE, "web", "index.html"))

    async def ws(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.clients.add(ws)
        await ws.send_str(json.dumps(self.snapshot()))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        await self.handle(json.loads(msg.data))
                    except LinkError as e:
                        self.link.add_log(f"не отправлено: {e}")
                    except Exception as e:  # noqa: BLE001
                        self.link.add_log(f"ошибка команды: {type(e).__name__}: {e}")
        finally:
            self.clients.discard(ws)
        return ws

    async def autosave_trace(self):
        """Раз в 20 секунд сохраняет запись по кадрам в trace.json, чтобы перезапуск службы её не стирал."""
        loop = asyncio.get_running_loop()

        def dump():
            rows = list(self.af.trace)
            if rows:
                with open(os.path.join(HERE, "trace.json"), "w") as f:
                    json.dump({"rows": rows, "fields": ["t", "size", "sharp", "x", "state", "cmds"]}, f)

        while True:
            await asyncio.sleep(20)
            try:
                await loop.run_in_executor(None, dump)
            except OSError:
                pass

    async def broadcaster(self):
        while True:
            await self.link.changed.wait()
            self.link.changed.clear()
            payload = json.dumps(self.snapshot())
            for c in list(self.clients):
                try:
                    await c.send_str(payload)
                except Exception:  # noqa: BLE001
                    self.clients.discard(c)
            await asyncio.sleep(0.1)

    def _encode_preview(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (640, int(h * 640 / w)))
        box = self.af.status.get("box") if self.af.enabled else None
        if box:
            x, y, bw, bh = box
            H, W = small.shape[:2]
            cv2.rectangle(small, (int(x * W), int(y * H)), (int((x + bw) * W), int((y + bh) * H)), (80, 220, 120), 2)
        return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    async def preview(self, request):
        resp = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                           "Cache-Control": "no-cache"})
        await resp.prepare(request)
        loop = asyncio.get_running_loop()
        last = 0.0
        try:
            while True:
                frame, st = self.grab.latest()
                if frame is None or st == last:
                    await asyncio.sleep(0.03)
                    continue
                last = st
                jpg = await loop.run_in_executor(None, self._encode_preview, frame)
                await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg)
                                 + jpg + b"\r\n")
                await asyncio.sleep(0.07)
        except Exception:  # noqa: BLE001 — клиент закрыл страницу
            pass
        return resp


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=None,
                    help="номер Blackmagic в OpenCV; другие марки отклоняются")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--sim", action="store_true", help="симулятор камеры и картинки")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # журнал службы без задержки

    link = SimLink() if args.sim else CameraLink()
    hub = VideoHub(sim=args.sim, index=args.cam, on_change=link.touch)
    prev = os.path.join(HERE, "trace.json")
    if os.path.exists(prev):  # запись прошлого запуска не теряем: перезапуск службы иначе стёр бы то, что нужно разобрать
        try:
            os.replace(prev, os.path.join(HERE, "trace_prev.json"))
        except OSError:
            pass
    app_state = App(link, hub)
    app = web.Application()
    app.add_routes([web.get("/", app_state.index), web.get("/ws", app_state.ws),
                    web.get("/preview.mjpg", app_state.preview)])
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", args.port).start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Дашборд: {url}  (Ctrl+C — остановить)")
    if not args.no_browser:
        webbrowser.open(url)
    app_state.start()
    await asyncio.gather(link.run(), hub.watch(), app_state.broadcaster(), app_state.autosave_trace())


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nОстановлено.")
