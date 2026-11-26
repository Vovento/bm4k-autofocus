"""Видео с камеры по USB (UVC): сам находит Pocket 4K, подхватывает при включении и отпускает при выключении."""
import asyncio
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from bmcc import camera_names, open_camera


def is_blackmagic_camera(name):
    """Разрешённый источник видео: только камеры Blackmagic Cinema/Pocket Cinema."""
    value = (name or "").casefold()
    return "pocket cinema camera" in value or ("blackmagic" in value and "camera" in value)


def blackmagic_camera_index(names, requested=None):
    """Индекс совместимой Blackmagic; явный индекс не может обойти белый список."""
    if requested is not None:
        if 0 <= requested < len(names) and is_blackmagic_camera(names[requested]):
            return requested
        return None
    return next((i for i, name in enumerate(names) if is_blackmagic_camera(name)), None)


class Grabber(threading.Thread):
    """Читает кадры непрерывно, чтобы потребители всегда получали свежий кадр без очереди."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap, self.frame, self.stamp = cap, None, 0.0
        self.lock = threading.Lock()
        self.running = True
        self.started = time.monotonic()

    def run(self):
        try:
            while self.running:
                ok, f = self.cap.read()
                if ok:
                    with self.lock:
                        self.frame, self.stamp = f, time.monotonic()
                else:
                    time.sleep(0.05)
        finally:
            self.cap.release()

    def stop(self):
        self.running = False

    def latest(self):
        with self.lock:
            return self.frame, self.stamp


class SimGrabber(Grabber):
    """Синтетическая картинка: градиент и движущийся круг, ~25 кадров/с."""

    def __init__(self):
        super().__init__(None)

    def run(self):
        h, w = 720, 1280
        base = np.zeros((h, w, 3), np.uint8)
        base[:] = np.linspace(40, 120, w, dtype=np.uint8)[None, :, None]
        t0 = time.monotonic()
        while self.running:
            t = time.monotonic() - t0
            f = base.copy()
            cx, cy = int(w / 2 + 300 * np.sin(t * 0.7)), int(h / 2 + 120 * np.cos(t * 0.9))
            cv2.circle(f, (cx, cy), 110, (80, 160, 230), -1)
            cv2.putText(f, "SIM", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4)
            with self.lock:
                self.frame, self.stamp = f, time.monotonic()
            time.sleep(0.04)


def usb_present():
    """Подключена ли Pocket 4K по USB. ioreg отвечает за ~15 мс и всегда видит актуальное состояние."""
    try:
        out = subprocess.run(["/usr/sbin/ioreg", "-p", "IOUSB", "-w0"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "pocket cinema" in out.lower()


class VideoHub:
    """Источник кадров для автофокуса и превью.

    Камеру ищем по имени среди видеоустройств macOS (номера OpenCV меняются при подключении iPhone-камеры и т. п.),
    поэтому по умолчанию никогда не схватим встроенную камеру MacBook вместо Pocket.

    Список видеоустройств macOS в долгоживущем процессе не обновляется (камеру, включённую позже, он не видит, а у
    выключенной остаётся мёртвый видеопоток). Поэтому включение и выключение камеры отслеживаем по USB (ioreg) и в этот
    момент перезапускаем процесс: свежий процесс видит актуальный список.
    """
    STALE_AFTER = 3.0     # столько секунд без кадров — видео считается недоступным
    SETTLE = 10.0         # первые секунды после старта на смену состояния камеры не реагируем
    CONFIRM = 2.0         # состояние USB должно продержаться столько секунд (камера успевает «подняться»)
    NO_LIST_AFTER = 6.0   # камера есть по USB, но macOS не показывает её в списке видеоустройств
    OPEN_DELAY = 4.0      # после старта процесса ждём столько секунд, прежде чем открывать камеру: она ещё загружается
    NO_FRAMES_AFTER = 8.0  # поток открыт, но кадры не идут: чтение зависло, помогает только перезапуск процесса

    def __init__(self, sim=False, index=None, on_change=None):
        self.sim, self.index, self.on_change = sim, index, on_change
        self.grabber = None
        self.error = ""
        self._last_names = None

    @property
    def available(self):
        g = self.grabber
        if g is None:
            return False
        if g.stamp == 0.0:  # кадров ещё не было: даём камере время на запуск
            return time.monotonic() - g.started < 5.0
        return time.monotonic() - g.stamp < self.STALE_AFTER

    def latest(self):
        g = self.grabber
        return g.latest() if g else (None, 0.0)

    def _notify(self):
        if self.on_change:
            self.on_change()

    def _blackmagic_index(self):
        names = camera_names()
        if names != self._last_names:
            allowed = [(i, name) for i, name in enumerate(names) if is_blackmagic_camera(name)]
            print(f"Камеры Blackmagic: {allowed or 'не найдены'}", flush=True)
            self._last_names = names
        index = blackmagic_camera_index(names, self.index)
        if self.index is not None and index is None:
            print(f"Камера #{self.index} отклонена: BM4K Autofocus принимает только Blackmagic Cinema Camera", flush=True)
        return index

    def _open(self, index):
        g = Grabber(open_camera(index))
        g.start()
        return g

    def _restart(self, reason):
        print(f"Перезапуск процесса: {reason}", flush=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    async def watch(self):
        if self.sim:
            self.grabber = SimGrabber()
            self.grabber.start()
            self._notify()
            return
        loop = asyncio.get_running_loop()
        t_start = time.monotonic()
        last_present = await loop.run_in_executor(None, usb_present)
        changed_since = no_list_since = None
        while True:
            try:
                present = await loop.run_in_executor(None, usb_present)
                now = time.monotonic()
                settled = now - t_start > self.SETTLE

                if settled and present != last_present:
                    changed_since = changed_since or now
                    if now - changed_since >= self.CONFIRM:
                        self._restart("камера включена" if present else "камера выключена")
                else:
                    changed_since = None

                stale_tries = int(os.environ.get("BM4K_STALE", "0"))
                if present:
                    idx = await loop.run_in_executor(None, self._blackmagic_index)
                    # после неудач ждём дольше: камера могла быть не готова
                    if idx is not None and self.grabber is None and now - t_start >= self.OPEN_DELAY + 3.0 * stale_tries:
                        print(f"Видео: открываю камеру #{idx}", flush=True)
                        try:
                            self.grabber = await asyncio.wait_for(loop.run_in_executor(None, self._open, idx), 15)
                        except asyncio.TimeoutError:  # открытие камеры зависло: поток не убить, только перезапуск
                            if stale_tries < 4:
                                os.environ["BM4K_STALE"] = str(stale_tries + 1)
                                self._restart(f"открытие камеры зависло (попытка {stale_tries + 1})")
                            raise
                        self.error = ""
                        os.environ.pop("BM4K_NOLIST", None)
                        self._notify()
                    if idx is None and self.grabber is None:
                        no_list_since = no_list_since or now
                        if settled and now - no_list_since >= self.NO_LIST_AFTER:
                            tries = int(os.environ.get("BM4K_NOLIST", "0"))
                            if tries < 3:  # предохранитель от бесконечных перезапусков
                                os.environ["BM4K_NOLIST"] = str(tries + 1)
                                self._restart("камера есть по USB, но не видна в списке видеоустройств")
                            elif self.error != "no-list":
                                self.error = "no-list"
                                print("Камера видна по USB, но не как веб-камера: проверьте режим USB на камере", flush=True)
                                self._notify()
                    else:
                        no_list_since = None
                    g = self.grabber
                    if g is not None:
                        age = now - (g.stamp or g.started)
                        if age > self.NO_FRAMES_AFTER:
                            if stale_tries < 4:
                                os.environ["BM4K_STALE"] = str(stale_tries + 1)
                                self._restart(f"видео: кадров нет {age:.0f} с (попытка {stale_tries + 1})")
                            elif self.error != "no-frames":
                                self.error = "no-frames"
                                print("Видео: кадры так и не пошли после нескольких перезапусков", flush=True)
                                self._notify()
                        elif g.stamp and now - g.started > 5:
                            os.environ.pop("BM4K_STALE", None)  # видео пошло: счётчик неудач сбрасываем
                if self.grabber is not None and not present and self.grabber.running:
                    self.grabber.stop()  # камера выключена: поток мёртв (процесс перезапустится сам)
            except SystemExit as e:  # нет доступа к камере и т. п.
                if str(e) != self.error:
                    self.error = str(e)
                    self._notify()
            except Exception as e:  # noqa: BLE001
                if repr(e) != self.error:
                    self.error = repr(e)
                    self._notify()
            await asyncio.sleep(1.0)
