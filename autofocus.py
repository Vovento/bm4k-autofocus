"""Следящий автофокус как управляемый модуль: включается и выключается из дашборда.

Логика та же, что в tracker.py: резкость и размер лица → при расфокусе или сдвиге запускаем one-shot AF камеры.
"""
import asyncio
import collections
import math
import os
import statistics
import time

import cv2
import numpy as np

from bmcc import detect_face_landmarks
from camera import LinkError
from debug_sweep import tenengrad
from focusmodel import (INFINITY_MM, N_REF, SIGMA_LN, FocusModel, curve_from_sweep, lens_slug, load_profile, parse_mm,
                        save_profile, set_curve, x_from_distance)
from protocol import packet

AF_PACKET = packet(0, 1, 0)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus_model.json")

# точный режим
FOLLOW_DEAD = 0.01       # не двигать фокус, если целевая позиция сдвинулась меньше чем на это (примерно ступень объектива)
FOLLOW_EVERY = 0.15      # в покое: не чаще одной команды за столько секунд
# плавность: пока лицо движется, цель подтягивается экспоненциально, команды идут мелкими шагами и чаще
MOVE_V = 0.22            # скорость изменения ln размера лица (1/с), выше которой считаем движением (шум размера 2-3% даёт ложные 0.06-0.15)
MOVE_HOLD = 0.6          # после последнего признака движения режим движения держится столько секунд
MOVE_JUMP = 0.025        # или цель ушла от текущей позиции больше чем на это
MOVE_DEAD = 0.004        # в движении порог команды (в покое — FOLLOW_DEAD, чтобы фокус не дрожал на шуме)
MOVE_EVERY = 0.08        # в движении команды не чаще, чем раз в столько секунд
SMOOTH_TAU = 0.12        # постоянная времени сглаживания цели, с
PROBE_STEP = 0.012       # шаг уточнения по резкости (примерно одна ступень объектива)
PROBE_SETTLE = 0.3       # ждать после команды, пока объектив доедет
REFINE_BELOW = 0.55      # резкость стабильно ниже 55% эталона (с поправкой на размер лица): уточняем
REFINE_COOLDOWN = 4.0
REFINE_HOLD = 8.0        # если уточнение показало, что позиция уже лучшая, не повторять столько секунд
GAIN = 1.12              # позиция меняется, только если резкость выросла хотя бы на 12% (шум замера)
LATENCY = 0.2            # упреждение при движении, с (задержка команды и хода объектива)
PREDICT_MIN_V = 0.25     # скорость изменения ln размера лица (1/с), ниже которой упреждения нет
PREDICT_MAX = 0.12       # предельное упреждение по ln размера
SIZE_SOURCE = "eyes"     # размер лица: "eyes" — среднее геометрическое рамки и расстояния между глазами, "box" — только рамка
SHARP_SOURCE = "face"    # где мерить резкость: "face" — всё лицо (глаза оказались шумнее: моргание), "eyes" — область глаз
SIGMA = 0.022           # ширина колокола резкости по позиции фокуса (по картам резкости: половина пика около ±0.025)
GOOD = 0.85              # картинка не ниже 85% лучшей: перепроверка после перемещения и по таймеру не нужна
WIDE = 0.03             # широкий шаг первого поиска (когда картинка мягкая или лицо только вернулось)
QLOW = 0.6               # резкость×размер ниже 60% лучшего виденного — картинка мягкая, нужно уточнить
ZOOM_QUIET = 1.3         # зум считается остановленным, если фокусное не менялось столько секунд
MOVED_SINCE = 0.08       # размер лица изменился больше чем на столько (по ln) с последней проверенной позиции: перепроверить
PERIODIC = 90.0          # даже в покое перепроверять позицию не реже, чем раз в столько секунд (шкала фокуса со временем уходит)
VERIFIED_TTL = 900.0     # проверенные по резкости позиции пиков хранятся столько секунд (шкала фокуса со временем уходит)
W_BASE = 0.02           # вес «базовой» оценки (последняя проверенная точка или модель) рядом с найденными пиками
VERIFIED_TAU = 60.0      # вес найденного пика убывает с возрастом (шкала фокуса уходит: старые пики тянут назад к прежней позиции)
VERIFIED_STALE = 0.03    # свежая проверка отличается от старой записи на похожем размере лица больше этого: старая запись устарела
VERIFIED_NEAR = 0.7      # использовать пик на расстоянии, ближайшем по ln(размер/зум), если разница не больше этого
SHIFT_STEP = 0.012      # сдвиг шкалы обновляется, только если оценка изменилась больше чем на это
X_MIN, X_MAX = 0.3, 0.95  # рабочий диапазон Focus (шкала со временем уходит на ~0.06–0.09, поэтому запас выше прежних 0.93)

# предмет в центре кадра (когда лицо закрыто)
ROI = (0.3, 0.25, 0.7, 0.75)  # центральная область кадра (левый, верхний, правый, нижний край в долях)
LOST_AFTER = 1.2         # лица нет столько секунд — переходим на предмет в центре
BACK_AFTER = 0.4         # лицо снова видно столько секунд — возвращаемся на лицо
MANUAL_HOLD = 10.0       # «Навести сейчас» удерживает фокус на центре столько секунд, даже если лицо видно
OBJ_CHANGE = 0.7         # насколько должно измениться содержимое центра (0 — то же, ~1 — совсем другое): новый предмет
OBJ_STILL = 0.15         # изменение между кадрами ниже этого — предмет неподвижен

# sharp_drop: резкость < доли эталона = расфокус; size: изменение размера лица; need: сколько секунд держится условие
PROFILES = {
    "soft": {"sharp_drop": 0.45, "size": 0.18, "need": 0.7},
    "normal": {"sharp_drop": 0.55, "size": 0.12, "need": 0.35},
    "fast": {"sharp_drop": 0.65, "size": 0.08, "need": 0.25},
}
STILL_FOR, STILL_TOL = 0.4, 0.03   # лицо «остановилось»: размер стабилен столько секунд с допуском
FORCE_AFTER = 2.0                  # если движение не прекращается, запускать AF всё равно
COOLDOWN = 1.0
MIN_AF_TIME, MAX_AF_TIME = 0.7, 3.5
SETTLE_SAMPLES, SETTLE_TOL = 6, 0.15
MIN_GOOD, MAX_RETRIES = 30.0, 2
DETECT_WIDTH = 640


def measure(frame):
    """(резкость, ширина лица, рамка, расстояние между глазами, резкость лица, резкость глаз) или None, если лица нет.

    Резкость в первом поле берётся по области глаз (SHARP_SOURCE = "eyes"): она меньше зависит от мимики и разговора
    и совпадает с тем, на что реально нужен фокус. Расстояния и рамка — в долях кадра.
    """
    h, w = frame.shape[:2]
    scale = DETECT_WIDTH / w
    found = detect_face_landmarks(cv2.resize(frame, (DETECT_WIDTH, int(h * scale))))
    if found is None:
        return None
    (x, y, fw, fh), iod, eyes, eye_l, eye_r, yaw, roll = found
    x, y, fw, fh = (int(v / scale) for v in (x, y, fw, fh))
    crop = frame[max(y, 0):y + fh, max(x, 0):x + fw]
    if crop.size == 0:
        return None
    sharp_face = tenengrad(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    sharp_eyes = None
    if eyes is not None:
        ex, ey, ew, eh = (int(v / scale) for v in eyes)
        ecrop = frame[max(ey, 0):ey + eh, max(ex, 0):ex + ew]
        if ecrop.shape[0] >= 12 and ecrop.shape[1] >= 24:  # слишком мелкая область: шум, берём лицо
            sharp_eyes = tenengrad(cv2.cvtColor(ecrop, cv2.COLOR_BGR2GRAY))
    # центральная часть лица (от глаз до рта): фона и волос в ней почти нет, на дальнем плане рамка лица им «загрязнена»
    cx0, cx1 = x + int(fw * 0.25), x + int(fw * 0.75)
    cy0, cy1 = y + int(fh * 0.30), y + int(fh * 0.75)
    ccrop = frame[max(cy0, 0):cy1, max(cx0, 0):cx1]
    sharp_core = tenengrad(cv2.cvtColor(ccrop, cv2.COLOR_BGR2GRAY)) if ccrop.shape[0] >= 12 and ccrop.shape[1] >= 12 else None
    if SHARP_SOURCE == "eyes" and sharp_eyes is not None:
        sharp = sharp_eyes
    elif SHARP_SOURCE == "core" and sharp_core is not None:
        sharp = sharp_core
    else:
        sharp = sharp_face
    def eye_sharp(box):  # резкость одного глаза (None, если область слишком мелкая)
        if box is None:
            return None
        ex, ey, ew, eh = (int(v / scale) for v in box)
        c = frame[max(ey, 0):ey + eh, max(ex, 0):ex + ew]
        return tenengrad(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)) if c.shape[0] >= 8 and c.shape[1] >= 12 else None
    return (sharp, fw / w, (max(x, 0) / w, max(y, 0) / h, fw / w, fh / h), (iod / DETECT_WIDTH) if iod else None,
            sharp_face, sharp_eyes, sharp_core, eye_sharp(eye_l), eye_sharp(eye_r), yaw, roll)


def roi_metrics(frame):
    """Резкость центральной области и её «отпечаток» (24×18, без яркости и контраста): по нему видно, что в центре другой предмет."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame[int(ROI[1] * h):int(ROI[3] * h), int(ROI[0] * w):int(ROI[2] * w)], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (24, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
    return tenengrad(gray), (small - small.mean()) / (small.std() + 1e-3)


class LensChanged(Exception):
    """На камере сменили объектив: нужно загрузить его профиль."""


class FaceAF:
    def __init__(self, link, grab):
        self.link, self.grab = link, grab
        self.profile = "normal"
        self.enabled = False
        self.task = None
        self.mode = "precise"  # precise (основной): фокус по размеру лица + уточнение по резкости; camera: one-shot AF камеры
        self.status = {"state": "выкл", "sharp": None, "base": None, "width": None, "mode": "precise", "x": None,
                       "C": None, "af_count": 0, "last_af_s": None, "fps": 0, "box": None}
        self._last_stamp = 0.0
        self._samples = 0
        self._rebase = False
        self._roi = None           # (резкость центра, отпечаток) последнего кадра (точный режим)
        self._manual_until = 0.0   # до этого момента фокус удерживается на центре («Навести сейчас»)
        self.trace = collections.deque(maxlen=14000)  # по кадру: (время, размер, резкость, позиция фокуса, состояние, число команд) — для разбора
        self._cmds = 0
        self._verified_at = 0.0  # когда встроенный AF последний раз дал проверенный результат (учитывает цикл)
        self._fallback = False   # у объектива камера не присылает расстояние: работаем встроенным AF камеры
        self._eye_ratio = 0.395  # отношение «расстояние между глазами / ширина рамки лица», уточняется по ходу

    # ---- управление ----
    def _publish(self, **kw):
        self.status.update(kw)
        self.link.touch()

    def set_enabled(self, on):
        if on and not self.enabled:
            self.enabled = True
            self.task = asyncio.ensure_future(self._run())
        elif not on and self.enabled:
            self.enabled = False
            if self.task:
                self.task.cancel()
            self._publish(state="выкл", box=None, sharp=None, base=None)

    def set_mode(self, name):
        if name in ("camera", "precise") and name != self.mode:
            was = self.enabled
            self.set_enabled(False)
            self.mode = name
            self._publish(mode=name)
            if was:
                self.set_enabled(True)

    def set_profile(self, name):
        if name in PROFILES:
            self.profile = name
            self.link.touch()

    async def af_now(self):
        """Ручной запуск AF; если слежение включено, эталон будет снят заново."""
        await self.link.send(AF_PACKET)
        if self.mode == "precise":  # встроенный AF целится в центр: на время оставляем фокус на предмете в центре
            self._manual_until = time.monotonic() + MANUAL_HOLD
        else:
            self._rebase = True
        self._publish(af_count=self.status["af_count"] + 1)

    async def sweep(self, lo=0.7, hi=0.95, step=0.01, settle=0.7):
        """Перебор позиций фокуса: резкость лица и показание камеры о расстоянии (12.12). Для калибровки и отладки.

        Слежение на это время нужно выключить (это делает вызывающий).
        """
        rows = []
        self._publish(sweep=None, state="перебор…")
        await self.link.send(packet(0, 0, 128, [max(0.0, lo - 0.05)]))  # подход всегда снизу
        await asyncio.sleep(1.2)
        n = int(round((hi - lo) / step))
        for i in range(n + 1):
            x = lo + i * step
            await self.link.send(packet(0, 0, 128, [x]))
            await asyncio.sleep(settle)
            sh, ws, sf, se, sc, sl, sr, yw = [], [], [], [], [], [], [], []
            for _ in range(5):
                m = await self._next_sample()
                if m:
                    sh.append(m[0])
                    ws.append(m[1])
                    if len(m) > 6:
                        sf.append(m[4])
                        for src, dst in ((5, se), (6, sc), (7, sl), (8, sr), (9, yw)):
                            if len(m) > src and m[src] is not None:
                                dst.append(m[src])
            med = lambda v: round(statistics.median(v), 1) if v else None  # noqa: E731
            rows.append({"x": round(x, 3), "sharp": med(sh), "face": med(sf), "eyes": med(se), "core": med(sc),
                         "left": med(sl), "right": med(sr), "yaw": round(statistics.median(yw), 3) if yw else None,
                         "w": round(statistics.median(ws), 3) if ws else None,
                         "D": self.link.state.get("12.12")})
        self._publish(sweep=rows, state="выкл")

    async def backlash_test(self, x, delta=0.06, reps=3):
        """Резкость в одной и той же позиции x при подходе снизу и сверху (проверка люфта). Слежение должно быть выключено."""
        out = {"snizu": [], "sverhu": []}
        self._publish(backlash=None, state="проверка люфта…")
        for _ in range(reps):
            for name, start in (("snizu", x - delta), ("sverhu", x + delta)):
                await self.link.send(packet(0, 0, 128, [start]))
                await asyncio.sleep(1.2)
                await self.link.send(packet(0, 0, 128, [x]))
                await asyncio.sleep(1.0)
                vals = []
                for _ in range(8):
                    m = await self._next_sample()
                    if m:
                        vals.append(m[0])
                if vals:
                    out[name].append(round(statistics.median(vals), 1))
        self._publish(backlash=out, state="выкл")

    # ---- внутренняя логика ----
    def _analyze(self, frame):
        m = measure(frame)
        if self.mode == "precise":
            self._roi = roi_metrics(frame)
        return m

    async def _next_sample(self):
        loop = asyncio.get_running_loop()
        while True:
            if not self.grab.available:  # камеру выключили: пусть главный цикл обновит статус
                await asyncio.sleep(0.2)
                return None
            frame, st = self.grab.latest()
            if frame is not None and st != self._last_stamp:
                self._last_stamp = st
                self._samples += 1
                return await loop.run_in_executor(None, self._analyze, frame)
            await asyncio.sleep(0.01)

    async def _settle(self):
        await asyncio.sleep(MIN_AF_TIME)
        hist = collections.deque(maxlen=SETTLE_SAMPLES)
        t0 = time.monotonic()
        while time.monotonic() - t0 < MAX_AF_TIME:
            m = await self._next_sample()
            if m is None:
                continue
            hist.append(m)
            if len(hist) == SETTLE_SAMPLES:
                sh = [h[0] for h in hist]
                ws = [self._size_of(h) for h in hist]
                med = statistics.median(sh)
                if max(abs(s - med) for s in sh) < SETTLE_TOL * med and max(ws) - min(ws) < STILL_TOL * statistics.median(ws):
                    return med, statistics.median(ws)
        if not hist:
            return None
        return statistics.median(h[0] for h in hist), statistics.median(self._size_of(h) for h in hist)

    async def _run_af(self, reason):
        base = None
        for attempt in range(MAX_RETRIES + 1):
            self._publish(state="наводка…" if not attempt else "повтор AF…")
            t0 = time.monotonic()
            await self.link.send(AF_PACKET)
            self.status["af_count"] += 1
            base = await self._settle()
            if base is None:
                return None
            self._publish(last_af_s=round(time.monotonic() - t0, 1))
            self.link.add_log(f"AF ({reason}) за {time.monotonic() - t0:.1f} с, резкость {base[0]:.0f}")
            if base[0] >= MIN_GOOD:
                return base
        return base

    # ---- точный режим ----
    def _focal_mm(self):
        return parse_mm(self.link.state.get("12.11")) or 16.0

    async def _send_focus(self, x):
        x = min(max(x, X_MIN), X_MAX)
        await self.link.send(packet(0, 0, 128, [x]))
        self._cmds += 1
        return x

    async def _probe(self, x, size_ref):
        """Поставить фокус в x и измерить резкость лица. None — лицо двигалось, замер недействителен."""
        x = await self._send_focus(x)
        await asyncio.sleep(PROBE_SETTLE)
        await self._next_sample()  # кадр во время движения объектива отбрасываем
        sh, sz = [], []
        for _ in range(3):
            m = await self._next_sample()
            if m:
                sh.append(m[0])
                sz.append(self._size_of(m))
        if len(sh) < 2 or abs(statistics.median(sz) / size_ref - 1) > 0.08:
            return None
        self.trace.append((round(time.monotonic(), 3), round(statistics.median(sz), 4), round(statistics.median(sh), 1),
                           round(x, 4), "уточняю…", self._cmds))
        return x, statistics.median(sh)

    @staticmethod
    def _log_parabola_peak(pts):
        """Вершина параболы по ln(резкость) через три лучшие соседние точки; None, если не выпукла."""
        xs = sorted(pts)
        i = max(range(len(xs)), key=lambda k: pts[xs[k]])
        if i == 0 or i == len(xs) - 1:
            return None
        x1, x2, x3 = xs[i - 1], xs[i], xs[i + 1]
        y1, y2, y3 = (math.log(max(pts[x], 1e-6)) for x in (x1, x2, x3))
        den = (x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1)
        if den == 0:
            return None
        xv = x2 - 0.5 * ((x2 - x1) ** 2 * (y2 - y3) - (x2 - x3) ** 2 * (y2 - y1)) / den
        return xv if (y2 >= y1 and y2 >= y3 and x1 <= xv <= x3) else None

    @staticmethod
    def _gauss_peak(xa, sa, xb, sb, sigma):
        """Положение пика колокола ширины sigma по двум замерам резкости."""
        if sa <= 0 or sb <= 0 or xa == xb:
            return None
        return (xa + xb) / 2 + sigma ** 2 * math.log(sb / sa) / (xb - xa)

    async def _refine(self, x0, s0, size_ref, prefer, wide=False, sigma=SIGMA):
        """Найти пик резкости двумя-тремя пробами, двигаясь монотонно (без качки вперёд-назад на большие расстояния).

        Шаг первой пробы — доля ширины колокола sigma (не больше 0.03: большие пробы заметны на экране). Если она не лучше
        исходной, проверяется обратная сторона. По двум точкам с ростом вычисляется положение пика, туда едем одной пробой;
        итог — вершина параболы по ln(резкость) или лучшая точка. s0 — резкость в x0 (её уже измерило слежение).
        Возвращает (позиция, резкость, направление: +1/−1, 0 — осталась на месте) или None, если лицо двигалось.
        """
        d = min(max((1.0 if wide else 0.55) * sigma, 0.008), 0.03)
        pts = {round(x0, 4): s0}

        async def probe(x):
            r = await self._probe(x, size_ref)
            if r is None:
                return None
            pts[round(r[0], 4)] = r[1]
            return r

        r1 = await probe(x0 + prefer * d)
        if r1 is None:
            return None
        pair = None
        if r1[1] >= s0 * GAIN:
            pair = (x0, s0, r1[0], r1[1])
        else:
            r2 = await probe(x0 - prefer * d)  # первая проба не лучше: пик может быть с другой стороны
            if r2 is None:
                return None
            if r2[1] >= max(s0, r1[1]) * GAIN:
                pair = (x0, s0, r2[0], r2[1])
        for _ in range(2):
            if pair is None:
                break
            xp = self._gauss_peak(*pair, sigma)
            if xp is None:
                break
            xp = min(max(xp, x0 - 0.06), x0 + 0.06)
            nearest = min(pts, key=lambda k: abs(k - xp))
            if abs(nearest - xp) < 0.006:  # туда уже заходили
                break
            r = await probe(xp)
            if r is None:
                return None
            best_prev = pair[2] if pair[3] >= pair[1] else pair[0]
            pair = (best_prev, pts[round(best_prev, 4)], r[0], r[1]) if r[1] > pts[round(best_prev, 4)] else None
        best = max(pts, key=pts.get)
        v = self._log_parabola_peak(pts)
        x_star = v if v is not None else best
        x_star = min(max(x_star, min(pts) - d / 2), max(pts) + d / 2)
        moved_dir = 0 if abs(x_star - x0) < 0.006 else (1 if x_star > x0 else -1)
        return x_star, pts[best], moved_dir

    async def _seed(self, model, f):
        """Наводка встроенным AF камеры и запись результата в модель. Возвращает (резкость, размер) или None."""
        try:
            base = await self._run_af("обучение модели")
        except LinkError:
            return None
        if base and base[0] >= MIN_GOOD:
            await asyncio.sleep(0.3)
            if model.add(parse_mm(self.link.state.get("12.12")), base[1], f):
                model.q_verified(f, base[0] * base[1])
                self._verified_at = time.monotonic()
                return base
        await asyncio.sleep(1.0)
        return None

    def _size_of(self, m):
        """Размер лица в долях кадра: расстояние между глазами, приведённое к ширине рамки (или сама ширина рамки)."""
        if SIZE_SOURCE == "eyes" and self.mode == "precise" and not self._fallback and len(m) > 3 and m[3]:
            return math.sqrt(m[1] * m[3] / self._eye_ratio)  # ошибки рамки и глаз независимы: среднее геометрическое точнее
        return m[1]

    @staticmethod
    def _slope(points):
        """Наклон ln размера по времени (1/с) методом наименьших квадратов; None, если данных мало."""
        if len(points) < 8 or points[-1][0] - points[0][0] < 0.25:
            return None
        t0 = points[0][0]
        xs = [t - t0 for t, _ in points]
        ys = [v for _, v in points]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None

    async def _settle_roi(self):
        """Ждать, пока резкость центра после AF стабилизируется. Возвращает (резкость, отпечаток) или None."""
        await asyncio.sleep(MIN_AF_TIME)
        hist = collections.deque(maxlen=SETTLE_SAMPLES)
        t0 = time.monotonic()
        while time.monotonic() - t0 < MAX_AF_TIME:
            await self._next_sample()
            if self._roi is None:
                continue
            hist.append(self._roi)
            if len(hist) == SETTLE_SAMPLES:
                sh = [h[0] for h in hist]
                med = statistics.median(sh)
                if max(abs(s - med) for s in sh) < SETTLE_TOL * med:
                    return med, hist[-1][1]
        return (statistics.median(h[0] for h in hist), hist[-1][1]) if hist else None

    async def _object_step(self, o, now):
        """Один кадр в режиме «предмет в центре»: следим за резкостью центра и запускаем AF камеры, когда он сменился или расфокусировался."""
        sharp, sig = self._roi
        if o["manual"] and o["sig"] is None:  # AF по кнопке уже отправлен: текущая картинка — исходная, второго AF не нужно
            o["sig"], o["last_af"] = sig, now
        motion = float(np.abs(sig - o["prev"]).mean()) if o["prev"] is not None else 1.0
        o["prev"] = sig
        if motion < OBJ_STILL:
            o["still"] = o["still"] or now
            o["hist"].append(sharp)
        else:
            o["still"] = None
            o["hist"].clear()
        still = o["still"] is not None and now - o["still"] >= 0.4
        if o["manual"] and still:  # AF по кнопке уже отправлен: берём картинку после него за эталон, второй AF не нужен
            o["ref"], o["sig"], o["manual"] = statistics.median(o["hist"]), sig, False
            o["last_af"] = now
        changed = o["sig"] is None or float(np.abs(sig - o["sig"]).mean()) > OBJ_CHANGE
        blur = o["ref"] is not None and len(o["hist"]) >= 20 and statistics.median(o["hist"]) < REFINE_BELOW * o["ref"]
        self._publish(state="предмет в центре", sharp=round(sharp, 1), base=round(o["ref"], 1) if o["ref"] else None,
                      box=None, x=None)
        self.trace.append((round(now, 3), 0.0, round(sharp, 1), None, "предмет в центре", self._cmds))
        if still and (changed or blur) and now - o["last_af"] >= (4.0 if changed else 5.0):
            self._publish(state="наводка на предмет…")
            t0 = time.monotonic()
            await self.link.send(AF_PACKET)
            self.status["af_count"] += 1
            res = await self._settle_roi()
            o["last_af"] = time.monotonic()
            o["hist"].clear()
            if res:
                o["ref"], o["sig"] = res
                self._publish(last_af_s=round(time.monotonic() - t0, 1))
                self.link.add_log(f"предмет в центре: AF за {time.monotonic() - t0:.1f} с, резкость {res[0]:.0f}")

    # ---- объектив: профиль, автокалибровка ----
    def _lens_name(self):
        v = self.link.state.get("12.9")
        return v if isinstance(v, str) else ""

    def _fnumber(self):
        return parse_mm(self.link.state.get("12.10"))

    async def calibrate_lens(self, slug, name):
        """Автокалибровка нового объектива: перебор Focus 0…0.97, по показаниям расстояния (12.12) строится кривая.

        Лицо не нужно, фокус при этом проезжает весь ход (около минуты), поэтому выполняется один раз для нового объектива.
        Если камера для этого объектива расстояния не присылает, профиль помечается no_distance и слежение идёт встроенным AF камеры.
        """
        self._publish(state="калибровка нового объектива (около минуты)…", box=None)
        self.link.add_log(f"новый объектив «{name}»: калибровка (перебор фокуса 0…1)")
        await self.link.send(packet(0, 0, 128, [0.0]))
        await asyncio.sleep(2.0)
        pts, steps = [], 62
        for i in range(steps + 1):
            x = i / steps * 0.97
            await self.link.send(packet(0, 0, 128, [x]))
            await asyncio.sleep(0.7)
            pts.append((x, parse_mm(self.link.state.get("12.12"))))
        curve = curve_from_sweep(pts)
        profile = {"lens": name, "sigma_ln": SIGMA_LN, "n_ref": N_REF, "created": time.strftime("%Y-%m-%d"),
                   "notes": "снято автокалибровкой (перебор Focus 0…0.97)"}
        if len(curve) < 4:
            profile["no_distance"] = True
            self.link.add_log("камера не присылает расстояние для этого объектива: слежение встроенным AF камеры")
        else:
            profile["curve"] = [[round(x, 4), d] for x, d in curve]
            self.link.add_log(f"калибровка готова: {len(curve)} точек кривой")
        save_profile(slug, profile)
        return profile

    async def _prepare_lens(self):
        """Дождаться камеры и названия объектива, загрузить профиль (при необходимости откалибровать). Возвращает (slug, профиль)."""
        for _ in range(40):  # до 20 секунд ждём подключения и названия объектива
            if self.link.connected and self._lens_name():
                break
            await asyncio.sleep(0.5)
        name = self._lens_name()
        slug = lens_slug(name)
        profile = load_profile(slug)
        if profile is None:
            profile = await self.calibrate_lens(slug, name or "неизвестный объектив")
        if profile.get("curve"):
            set_curve(profile["curve"])
        self._publish(lens=name or "неизвестный объектив")
        return slug, profile

    async def _loop_precise(self):
        slug, profile = await self._prepare_lens()
        if profile.get("no_distance"):  # расстояния нет: точный режим невозможен, работает встроенный AF камеры
            self._fallback = True
            self._publish(state="слежу (AF камеры: объектив без расстояния)")
            await self._loop()
            return
        self._fallback = False
        model = FocusModel(MODEL_PATH, lens=slug)
        model.sigma_ln, model.n_ref = profile.get("sigma_ln", SIGMA_LN), profile.get("n_ref", N_REF)
        x_cmd = ref = anchor = ref_hi = None   # команда фокуса; эталон резкости (резкость, размер); якорь (позиция, размер)
        last_cmd = 0.0
        last_refine = time.monotonic()  # первое уточнение не сразу после включения
        prefer = 1                      # в какую сторону первой пробовать при уточнении
        recent = collections.deque(maxlen=5)
        slow = collections.deque(maxlen=30)    # резкость за последние ~1,5 с: на один кадр с мимикой не реагируем
        widths = collections.deque()           # (время, размер) за STILL_FOR секунд: «замер ли»
        trend = collections.deque()            # (время, ln размера) за 0,5 с: скорость движения
        ratios = collections.deque(maxlen=300)  # отношение «глаза / рамка» этого человека
        self._eye_ratio = 0.395
        noise_box, noise_eyes = collections.deque(maxlen=40), collections.deque(maxlen=40)
        noise_sf, noise_se = collections.deque(maxlen=40), collections.deque(maxlen=40)  # шум резкости: лицо / глаза
        win_box, win_eyes = collections.deque(maxlen=5), collections.deque(maxlen=5)    # окна для сглаживания сравниваемых сигналов
        win_sf, win_se = collections.deque(maxlen=10), collections.deque(maxlen=10)
        quality = collections.deque()          # (время, в фокусе?) за 60 с
        cmd_times, refine_spans = collections.deque(), collections.deque()  # времена команд и интервалы уточнений
        t_start = time.monotonic()
        t_prev = x_s = None   # время прошлого кадра и сглаженная цель фокуса
        moving_until = 0.0    # до этого момента лицо считается движущимся
        nochange = 0       # подряд уточнений «без сдвига» на этом месте: пауза растёт 8, 16, 32 с
        verified = []   # (время, размер, зум, позиция) пиков, найденных по резкости: ставим фокус по ближайшему при возврате на это расстояние
        last_seed = -1e9   # когда последний раз пробовали наводку камеры вместо поиска по резкости
        hold_until = 0.0   # после «без сдвига» повторять проверку на том же месте рано (но не после перемещения)
        last_shift_key = None
        last_verified = 0.0   # когда позиция фокуса последний раз проверена по резкости
        f_prev, f_changed_at, zoom_dirty = None, -1e9, False   # зум менялся недавно: фокус не гоняем, потом перепроверяем
        shift_hist = collections.deque(maxlen=8)  # последние оценки сдвига шкалы
        face_sig = None   # отпечаток центра кадра, пока было видно лицо
        obj = face_back = None   # состояние режима «предмет в центре» (None — слежение за лицом)
        face_seen = t_start
        last_fps_t, last_fps_n, last_q = t_start, 0, t_start
        self._publish(state="ищу лицо", C=round(model.C, 3) if model.C else None)

        def reset_windows():
            recent.clear()
            widths.clear()
            slow.clear()
            trend.clear()

        while True:
            if not self.link.connected:
                self._publish(state="ждём камеру", box=None)
                await asyncio.sleep(0.5)
                x_cmd = None
                continue
            if not self.grab.available:
                self._publish(state="нет видео с камеры", box=None)
                await asyncio.sleep(0.5)
                x_cmd = None
                continue
            m = await self._next_sample()
            now = time.monotonic()
            if self._lens_name() and lens_slug(self._lens_name()) != slug:
                raise LensChanged()  # объектив сменили на ходу: перечитать профиль
            if now - last_fps_t > 1.0:
                self.status["fps"] = round((self._samples - last_fps_n) / (now - last_fps_t))
                last_fps_t, last_fps_n = now, self._samples
            # лицо ↔ предмет в центре: лицо скрылось (или нажата «Навести сейчас») — фокус на центр; лицо вернулось — обратно
            if m is not None:
                face_seen = now
                if self._roi is not None:
                    face_sig = self._roi[1]  # как выглядел центр кадра с лицом: от него отсчитываем «появился другой предмет»
            manual = now < self._manual_until
            lost = m is None and now - face_seen >= LOST_AFTER
            if m is not None and obj is not None and not manual:
                face_back = face_back or now
            else:
                face_back = None
            if (manual or lost or (obj is not None and (m is None or (face_back is not None and now - face_back < BACK_AFTER)))) \
                    and self._roi is not None:
                if obj is None:
                    obj = {"prev": None, "still": None, "hist": collections.deque(maxlen=30), "ref": None,
                           "sig": None if manual else face_sig, "last_af": 0.0, "manual": manual}
                    self.link.add_log("удержание на центре: фокус на предмет в центре" if manual
                                      else "лицо закрыто: фокус на предмет в центре")
                try:
                    await self._object_step(obj, now)
                except LinkError:
                    await asyncio.sleep(0.5)
                continue
            if obj is not None:  # лицо вернулось: модель сразу ставит фокус на лицо, без поиска
                obj = None
                x_cmd = None
                ref = ref_hi = None  # эталон резкости был до ухода: перепроверим позицию, как только лицо замрёт
                last_refine = now - REFINE_COOLDOWN + 1.0
                reset_windows()
                self.link.add_log("лицо снова в кадре: фокус на лицо")
            if m is None:
                self._publish(state="ищу лицо", box=None)
                continue
            if len(m) > 3 and m[3]:
                ratios.append(m[3] / m[1])
                self._eye_ratio = statistics.median(ratios)
            size_now = self._size_of(m)
            recent.append((m[0], size_now))
            widths.append((now, size_now))
            trend.append((now, math.log(size_now)))
            while widths and now - widths[0][0] > STILL_FOR:
                widths.popleft()
            while trend and now - trend[0][0] > 0.5:
                trend.popleft()
            sharp = statistics.median(r[0] for r in recent)
            size = statistics.median(r[1] for r in recent)
            f = self._focal_mm()
            if f_prev is None:
                f_prev = f
            if abs(f - f_prev) >= 0.5:  # зум меняется: размер лица растёт не потому, что вы приблизились
                f_prev, f_changed_at, zoom_dirty = f, now, True
                slow.clear()
            zooming = now - f_changed_at < ZOOM_QUIET  # кольцо крутят с паузами: «остановился» — когда тихо дольше
            ws = [w for _, w in widths]
            still = len(widths) > 3 and now - widths[0][0] >= STILL_FOR * 0.8 \
                and (max(ws) - min(ws)) < STILL_TOL * statistics.median(ws)
            if still:
                slow.append(m[0])
            scaled = ref[0] * ref[1] / size if ref else None  # эталон резкости пересчитан на текущий размер лица
            win_box.append(m[1])
            if len(m) > 3 and m[3]:
                win_eyes.append(m[3] / self._eye_ratio)
            if len(m) > 5 and m[4] and m[5]:
                win_sf.append(m[4])
                win_se.append(m[5])
            if still:  # шум двумя способами, по сглаженным сигналам, как их видит программа (в покое: чем меньше, тем лучше)
                if len(win_box) == 5:
                    noise_box.append(statistics.median(win_box))
                if len(win_eyes) == 5:
                    noise_eyes.append(statistics.median(win_eyes))
                if len(win_sf) == 10 and len(win_se) == 10:
                    noise_sf.append(statistics.median(win_sf))
                    noise_se.append(statistics.median(win_se))

            # метрики качества за последние 60 секунд
            quality.append((now, scaled is not None and sharp >= 0.7 * scaled))  # как выглядит картинка, включая размытие в движении
            while quality and now - quality[0][0] > 60:
                quality.popleft()
            if now - last_q > 1.0:
                last_q = now
                span = min(60.0, now - t_start)
                for dq in (cmd_times,):
                    while dq and now - dq[0] > 60:
                        dq.popleft()
                while refine_spans and now - refine_spans[0][1] > 60:
                    refine_spans.popleft()
                blink = sum(min(e, now) - max(s, now - 60) for s, e in refine_spans)
                rel = lambda d: round(100 * statistics.pstdev(d) / statistics.mean(d), 1) if len(d) >= 15 else None  # noqa: E731
                self.status["quality"] = {
                    "in_focus": round(100 * sum(1 for _, ok in quality if ok) / len(quality)) if quality else None,
                    "blink": round(100 * blink / span, 1),
                    "cmds": round(len(cmd_times) * 60 / span),
                    "refines": round(len(refine_spans) * 60 / span, 1),
                    "noise_box": rel(noise_box), "noise_eyes": rel(noise_eyes),
                    "noise_sf": rel(noise_sf), "noise_se": rel(noise_se)}
            self.trace.append((round(now, 3), round(size, 4), round(sharp, 1), None if x_cmd is None else round(x_cmd, 4),
                               self.status["state"], self._cmds))
            self._publish(sharp=round(sharp, 1), width=round(size, 3), box=m[2],
                          base=round(scaled, 1) if scaled else None,
                          C=round(model.C, 3) if model.C else None,
                          x=round(x_cmd, 3) if x_cmd is not None else None)

            try:
                if self._rebase:  # ручной запуск AF: он уже отправлен, ждём результат и учимся на нём
                    self._rebase = False
                    base = await self._settle()
                    if base and base[0] >= MIN_GOOD and model.add(parse_mm(self.link.state.get("12.12")), base[1], f):
                        ref, x_cmd = base, model.target_x(base[1], f)
                        anchor = (x_cmd, base[1])
                        ref_hi = ref
                    reset_windows()
                    continue

                if model.C is None:  # модели ещё нет: один раз наводим встроенным AF и учимся
                    self._publish(state="обучаю модель (AF камеры)…")
                    base = await self._seed(model, f)
                    if base:
                        ref, x_cmd = base, model.target_x(base[1], f)
                        anchor = (x_cmd, base[1])
                        ref_hi = ref
                    reset_windows()
                    continue

                # зум остановился: у этого объектива фокус при зуме смещается. Если лицо ближе к центру, встроенный AF камеры
                # наводит за ~1 с (мелкими шагами по резкости на длинном конце выходило 15–30 с), и модель заодно учится
                if (zoom_dirty and not zooming) or (ref is None and now - last_seed >= 8.0 and not zooming):
                    zoom_dirty = False
                    last_seed = now
                    bx, by, bw, bh = m[2]
                    if 0.3 <= bx + bw / 2 <= 0.7 and 0.2 <= by + bh / 2 <= 0.7:
                        self._publish(state="зум остановился: наводка…")
                        base = await self._seed(model, f)
                        if base:
                            ref, x_cmd = base, model.target_x(base[1], f)
                            anchor = (x_cmd, base[1])
                            ref_hi = ref
                            last_cmd = last_refine = time.monotonic()
                        reset_windows()
                        continue
                    zoom_dirty = True  # лицо не по центру: AF камеры не годится, пусть работает широкая перепроверка

                # следование: целевая позиция по размеру лица с упреждением на движение
                v = self._slope(trend)
                lead = 0.0
                if v is not None and abs(v) > PREDICT_MIN_V:
                    lead = max(-PREDICT_MAX, min(PREDICT_MAX, v * LATENCY))
                size_pred = size * math.exp(lead)
                # от проверенной по резкости точки: сдвиг шкалы сокращается, и ответы камеры (ступенями) не раскачивают позицию.
                # Пики, найденные раньше на похожем размере лица (с похожим зумом, не старше VERIFIED_TTL), усредняются с весом,
                # плавно убывающим с различием размера. Выбор «ближайшего» одного пика прыгал при дрожании размера на 1% и давал хантинг.
                verified[:] = [v for v in verified if now - v[0] < VERIFIED_TTL]
                x_cur = model.target_x(size_pred, f)
                if anchor is not None:
                    x_base = anchor[0] + x_cur - model.target_x(anchor[1], f)
                else:
                    x_base = x_cur
                num, den = W_BASE * x_base, W_BASE
                for pk in verified:  # (имя не v: там скорость движения лица)
                    if abs(pk[2] - f) >= 1.5:
                        continue
                    dl = math.log(size / pk[1])
                    if abs(dl) >= VERIFIED_NEAR:
                        continue
                    w = math.exp(-dl * dl / (2 * 0.1 ** 2)) * math.exp(-(now - pk[0]) / VERIFIED_TAU)
                    num += w * (pk[3] + x_cur - model.target_x(pk[1], f))
                    den += w
                x_t = min(max(num / den, X_MIN), X_MAX)
                # плавное следование: в покое — большой порог (без дрожания), в движении — цель подтягивается экспоненциально
                # и фокус едет мелкими шагами, а не «ждём порога — прыжок»
                dt = (now - t_prev) if t_prev else 0.04
                t_prev = now
                if v is not None and abs(v) > MOVE_V:
                    moving_until = now + MOVE_HOLD
                moving = now < moving_until or (x_cmd is not None and abs(x_t - x_cmd) > MOVE_JUMP)
                if x_cmd is None or x_s is None or not moving:
                    x_s = x_t if x_cmd is None else x_cmd
                else:
                    x_s += (x_t - x_s) * (1 - math.exp(-dt / SMOOTH_TAU))
                target = x_s if (moving and x_cmd is not None) else x_t
                dead = MOVE_DEAD if moving else FOLLOW_DEAD
                every = MOVE_EVERY if moving else FOLLOW_EVERY
                if not zooming and (x_cmd is None or abs(target - x_cmd) > dead) and now - last_cmd >= every:
                    x_cmd = await self._send_focus(target)
                    last_cmd = now
                    cmd_times.append(now)

                # уточнение по резкости, когда лицо замерло, а резкость стабильно ниже эталона
                slow_med = statistics.median(slow) if len(slow) >= 24 else None
                qb = model.q_get(f)
                soft = slow_med is not None and qb is not None and slow_med * size < QLOW * qb
                after_zoom = zoom_dirty and not zooming  # зум остановился: у этого объектива фокус при зуме чуть смещается
                last_verified = max(last_verified, self._verified_at)
                # перепроверка по резкости не зависит от «нормы»: после перемещения (размер лица изменился), и раз в минуту
                far = anchor is not None and abs(math.log(size / anchor[1])) > MOVED_SINCE
                periodic = now - last_verified >= PERIODIC
                below = slow_med is None or qb is None or slow_med * size < GOOD * qb  # картинка ниже 85% лучшей (или данных нет)
                need = ref is None or soft or after_zoom or ((far or periodic) and below) \
                    or (slow_med is not None and slow_med < REFINE_BELOW * scaled)
                changed = after_zoom or (far and below) or ref is None  # реальная перемена: пауза «без сдвига» её не задерживает
                if still and need and not zooming and x_cmd is not None \
                        and (changed or now >= hold_until) \
                        and now - last_refine >= (1.0 if changed else REFINE_COOLDOWN):
                    self._publish(state="уточняю…")
                    t_ref = time.monotonic()
                    s0 = slow_med if slow_med is not None else sharp
                    res = await self._refine(x_cmd, s0, size, prefer, wide=after_zoom or (soft and ref is not None),
                                            sigma=model.sigma_x(x_cmd, self._fnumber()))
                    last_refine = time.monotonic()
                    refine_spans.append((t_ref, last_refine))
                    zoom_dirty = False
                    reset_windows()
                    if res is None:  # лицо двигалось во время замеров: повторим скоро
                        x_cmd = await self._send_focus(x_cmd)
                        last_refine -= REFINE_COOLDOWN - 1.0
                        continue
                    x_star, s_star, moved = res
                    x_cmd = await self._send_focus(x_star)
                    last_cmd = time.monotonic()
                    cmd_times.append(last_cmd)
                    await asyncio.sleep(PROBE_SETTLE)
                    if s_star >= MIN_GOOD and (qb is None or s_star * size >= 0.45 * qb):
                        ref = (0.5 * (s_star + s0) if moved else s0, size)  # не максимум из проб: он завышен шумом
                        anchor = (x_cmd, size)
                        ref_hi = ref
                        last_verified = time.monotonic()
                        keep = []
                        for pk in verified:  # старые записи рядом по размеру, но с другой позицией пика: шкала ушла, они устарели
                            if abs(pk[2] - f) < 1.5 and abs(math.log(size / pk[1])) < 0.15 \
                                    and abs(pk[3] + model.target_x(size, f) - model.target_x(pk[1], f) - x_cmd) > VERIFIED_STALE:
                                continue
                            keep.append(pk)
                        verified[:] = keep
                        for j, pk in enumerate(verified):
                            if abs(pk[2] - f) < 1.5 and abs(math.log(size / pk[1])) < 0.05:  # тот же размер лица: уточняем запись
                                x_adj = pk[3] + model.target_x(size, f) - model.target_x(pk[1], f)
                                verified[j] = (last_verified, size, f, 0.5 * x_adj + 0.5 * x_cmd)
                                break
                        else:
                            verified.append((last_verified, size, f, x_cmd))
                        del verified[:-30]
                        model.q_verified(f, ref[0] * ref[1])  # эталон «картинка резкая» — только по проверенным результатам
                        model.add(parse_mm(self.link.state.get("12.12")), size, f)
                        d_obs = parse_mm(self.link.state.get("12.12"))
                        model.learn_c(d_obs, size, f)
                        if d_obs and d_obs < INFINITY_MM:  # сдвиг шкалы — только по проверенной по резкости позиции, медленно
                            s_new = x_cmd - x_from_distance(d_obs)
                            if abs(s_new - model.shift) < 0.15:
                                model.shift = 0.7 * model.shift + 0.3 * s_new
                                model.save()
                        if moved:
                            prefer = moved
                            nochange = 0
                        else:  # лучше не нашли: резкость упала по другим причинам (свет, мимика) либо пик уже рядом;
                            nochange += 1  # повторять на этом месте всё реже, чтобы не получилось периодичного хантинга
                            hold_until = time.monotonic() + REFINE_HOLD * 2 ** min(nochange - 1, 3)
                        self.link.add_log(f"точная наводка: x={x_cmd:.3f}, резкость {s_star:.0f}, "
                                          f"{'шаг ' + ('+' if moved > 0 else '−') if moved else 'без сдвига'}, "
                                          f"C={model.C:.2f}")
                    else:
                        self.link.add_log("точная наводка не нашла резкость: помогает AF камеры")
                        base = await self._seed(model, f)
                        if base:
                            ref, x_cmd = base, model.target_x(base[1], f)
                            anchor = (x_cmd, base[1])
                        ref_hi = ref
                    continue
                self.status["dbg"] = {"still": still, "slow_n": len(slow), "qbest": round(qb, 1) if qb else None,
                                      "q": round((slow_med or 0) * size, 1), "soft": bool(soft),
                                      "since_refine": round(now - last_refine, 1), "spread": round(100 * (max(ws) - min(ws)) / statistics.median(ws), 1) if ws else None}
                self._publish(state="слежу")
                if ref and still and sharp >= REFINE_BELOW * scaled:
                    # эталон медленно следует за обычным уровнем резкости в обе стороны (а не только за пиками шума);
                    # пока картинка заметно хуже порога, он не меняется, чтобы не «привыкнуть» к расфокусу
                    new0 = ref[0] + 0.02 * (sharp * size / ref[1] - ref[0])
                    if ref_hi:  # вниз не дальше 80% проверенного уровня: иначе мягкую картинку примем за норму
                        new0 = max(new0, 0.8 * ref_hi[0] * ref_hi[1] / ref[1])
                    ref = (new0, ref[1])
            except LinkError:
                x_cmd = None
                await asyncio.sleep(0.5)

    async def _run(self):
        try:
            while True:
                try:
                    await (self._loop_precise() if self.mode == "precise" else self._loop())
                    return
                except LensChanged:
                    self.link.add_log("объектив сменился: загружаю его профиль")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()  # в постоянный журнал службы
            self.enabled = False
            self._publish(state=f"ошибка: {e}", box=None)

    async def _loop(self):
        base = None
        last_af, bad_since = 0.0, None
        recent = collections.deque(maxlen=3)
        widths = collections.deque()
        last_fps_t, last_fps_n = time.monotonic(), 0
        self._publish(state="ищу лицо")
        while True:
            if not self.link.connected:
                self._publish(state="ждём камеру", box=None)
                await asyncio.sleep(0.5)
                base = None
                continue
            if not self.grab.available:
                self._publish(state="нет видео с камеры", box=None)
                await asyncio.sleep(0.5)
                base = None
                continue
            m = await self._next_sample()
            now = time.monotonic()
            if now - last_fps_t > 1.0:
                self.status["fps"] = round((self._samples - last_fps_n) / (now - last_fps_t))
                last_fps_t, last_fps_n = now, self._samples
            if m is None:
                bad_since = None
                self._publish(state="ищу лицо", box=None)
                continue
            recent.append(m[:2])
            widths.append((now, m[1]))
            while widths and now - widths[0][0] > STILL_FOR:
                widths.popleft()
            sharp = statistics.median(r[0] for r in recent)
            width = statistics.median(r[1] for r in recent)
            p = PROFILES[self.profile]

            if base is None or self._rebase:
                self._rebase = False
                try:
                    base = await self._run_af("первичная наводка") or base
                except LinkError:
                    base = None
                    continue
                last_af = time.monotonic()
                recent.clear()
                widths.clear()
                continue

            blur = sharp < p["sharp_drop"] * base[0]
            moved = abs(width / base[1] - 1) > p["size"]
            ws = [w for _, w in widths]
            still = len(widths) > 3 and now - widths[0][0] >= STILL_FOR * 0.8 \
                and (max(ws) - min(ws)) < STILL_TOL * statistics.median(ws)
            self._publish(state="слежу", sharp=round(sharp, 1), base=round(base[0], 1),
                          width=round(width, 3), box=m[2])

            if blur or moved:
                bad_since = bad_since or now
                held = now - bad_since
                if now - last_af >= COOLDOWN and ((held >= p["need"] and still) or held >= FORCE_AFTER):
                    try:
                        base = await self._run_af("расфокус" if blur else "лицо сменило размер") or base
                    except LinkError:
                        base = None
                        continue
                    last_af = time.monotonic()
                    bad_since = None
                    recent.clear()
                    widths.clear()
            else:
                bad_since = None
                if sharp > base[0]:
                    base = (0.9 * base[0] + 0.1 * sharp, base[1])
