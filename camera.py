"""Связь с камерой по BLE: подключение с авто-переподключением, текущее состояние параметров, отправка команд.

CameraLink — настоящая камера; SimLink — симулятор для проверки интерфейса без железа.
"""
import asyncio
import time

from bleak import BleakClient, BleakScanner

from protocol import ASSIGN, OFFSET, decode, key

BM_SERVICE = "291d567a-6d75-11e6-8b77-86f30ca893d3"
OUTGOING = "5dd3465f-1aee-4299-8493-d2eca2f8e1bb"
INCOMING = "b864e140-76a0-416a-bf30-5876504537d9"


class LinkError(Exception):
    pass


class CameraLink:
    def __init__(self):
        self.state = {}          # "категория.параметр" → список значений, как их прислала камера
        self.status = "search"   # search | connecting | connected | error
        self.detail = ""
        self.name = ""
        self.client = None
        self.changed = asyncio.Event()  # поднимается при любом изменении состояния/статуса
        self.log = []            # последние события для отладочной панели
        self._lock = asyncio.Lock()

    # ---- состояние ----
    def touch(self):
        self.changed.set()

    def add_log(self, text):
        line = f"{time.strftime('%H:%M:%S')}  {text}"
        self.log.append(line)
        del self.log[:-40]
        print(line, flush=True)  # и в постоянный журнал службы (~/Library/Logs/BM4K Autofocus.log): виден после перезапуска
        self.touch()

    def set_status(self, status, detail=""):
        if (status, detail) != (self.status, self.detail):
            self.status, self.detail = status, detail
            self.add_log(f"связь: {status} {detail}".strip())

    @property
    def connected(self):
        return self.status == "connected"

    # ---- подключение ----
    def _on_incoming(self, _, data):
        for cat, par, _dtype, op, values in decode(data):
            self.state[key(cat, par)] = values
        self.touch()

    async def run(self):
        """Бесконечный цикл: искать камеру, подключаться, ждать обрыва, повторять."""
        while True:
            try:
                if self.status != "error":  # ошибку (например, нет доступа к Bluetooth) показываем, пока не пройдёт
                    self.set_status("search")
                dev = await BleakScanner.find_device_by_filter(
                    lambda d, adv: BM_SERVICE in [u.lower() for u in adv.service_uuids], timeout=10.0)
                if dev is None:
                    self.set_status("search", "камера не найдена: включите Bluetooth на камере, отключите телефон")
                    continue
                self.set_status("connecting", dev.name or "")
                dropped = asyncio.Event()
                async with BleakClient(dev, disconnected_callback=lambda _c: dropped.set()) as client:
                    self.client, self.name = client, dev.name or "камера"
                    await client.start_notify(INCOMING, self._on_incoming)
                    self.set_status("connected")
                    await dropped.wait()
                self.add_log("связь потеряна")
            except Exception as e:  # noqa: BLE001 — любая ошибка BLE: показать и переподключиться
                self.set_status("error", f"{type(e).__name__}: {e}")
            finally:
                self.client = None
            await asyncio.sleep(3)

    async def send(self, data):
        if not self.connected or self.client is None:
            raise LinkError("камера не подключена")
        async with self._lock:
            await self.client.write_gatt_char(OUTGOING, data, response=True)


class SimLink(CameraLink):
    """Притворяется камерой: хранит параметры, отвечает на команды."""

    def __init__(self):
        super().__init__()
        self.name = "Симулятор BMPCC 4K"
        self.state = {
            "0.2": [1.0], "0.3": [0.5], "0.4": [3], "0.7": [18],
            "1.0": [24, 0, 9, 0, 0], "1.2": [5600, 0], "1.11": [18000], "1.12": [48], "1.14": [400],
            "2.0": [0.7], "2.4": [0], "10.0": [3, 5], "10.1": [0, 0, 0, 0, 0],
        }

    async def run(self):
        self.set_status("connected", "симулятор")
        await asyncio.Event().wait()

    async def send(self, data):
        for cat, par, _dtype, op, values in decode(data):
            k = key(cat, par)
            self.add_log(f"sim ← {k} {'+' if op == OFFSET else '='} {values}")
            if op == OFFSET and k in self.state:
                self.state[k] = [a + b for a, b in zip(self.state[k], values)]
            elif op == ASSIGN and values:
                self.state[k] = values
            elif k == "1.3":  # auto WB
                self.state["1.2"] = [5200, 4]
        self.touch()
