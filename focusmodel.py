"""Модель фокуса: позиция Focus (0..1) ↔ расстояние, которое показывает камера (параметр 12.12), и связь с размером лица.

Для Pocket 4K + Sigma 18-35 через Metabones: камера повторяемо отвечает на абсолютную команду Focus и присылает
достигнутое расстояние строкой вида "178mm" ступенями. Расстояние до лица ≈ C · f / ширина лица в кадре,
где f — фокусное расстояние (12.11), а C учится по результатам наводки (медиана последних замеров).
"""
import bisect
import json
import math
import os
import re
import statistics
from collections import deque

# Измеренная кривая Focus → показание расстояния (центры «плато»; камера отдаёт показание ступенями).
# Снята перебором 0..1 на этом объективе; положение плато зависит от зума на ±0.01–0.02, это уточняется по резкости.
CURVE = [(0.0, 56.0), (0.2, 59.0), (0.3, 64.0), (0.4, 69.0), (0.5, 80.0), (0.6, 93.0), (0.7, 124.0), (0.72, 137.0),
         (0.755, 154.0), (0.79, 178.0), (0.815, 212.0), (0.835, 262.0), (0.87, 354.0), (0.905, 593.0),
         (0.935, 10389.0)]
_XS = [p[0] for p in CURVE]
_LN_D = [math.log(p[1]) for p in CURVE]
SIGMA_LN = 0.233      # ширина колокола резкости по ln(расстояния): в позициях фокуса она делится на наклон кривой
N_REF = 1.2           # число диафрагмы, при котором измерена SIGMA_LN (Sigma 18-35 с бустером показывает f/1.2); ширина ∝ числу диафрагмы
INFINITY_MM = 5000.0  # показания выше считаем «бесконечность»: для обучения не годятся
LEGACY_SLUG = "64x-sigma-dc-18-35-1-7-hsm"  # данные пользователя в старом плоском формате принадлежат этому объективу
PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")


def lens_slug(name):
    """Название объектива из камеры (параметр 12.9) → имя файла профиля."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "unknown"


def load_profile(slug):
    """Профиль объектива (кривая «позиция фокуса → расстояние» и пр.) из profiles/<slug>.json или None."""
    try:
        with open(os.path.join(PROFILES_DIR, slug + ".json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_profile(slug, data):
    os.makedirs(PROFILES_DIR, exist_ok=True)
    with open(os.path.join(PROFILES_DIR, slug + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def set_curve(points):
    """Сделать активной кривую объектива: [(позиция Focus, показание расстояния, мм), ...]."""
    global _XS, _LN_D
    pts = sorted((float(x), float(d)) for x, d in points)
    if len(pts) < 2:
        raise ValueError("в кривой должно быть минимум две точки")
    _XS = [p[0] for p in pts]
    ln, last = [], -1e9
    for _, d in pts:
        last = max(last, math.log(max(d, 1.0)))  # расстояние не уменьшается с ростом позиции
        ln.append(last)
    _LN_D = ln


def curve_from_sweep(pts):
    """Перебор позиций [(x, показание или None)] по возрастанию x → точки кривой в центрах «плато» (показание идёт ступенями)."""
    groups = []
    for x, d in pts:
        if not d:
            continue
        if groups and groups[-1][2] == d:
            groups[-1][1] = x
        else:
            groups.append([x, x, d])
    curve, last = [], 0.0
    for x0, x1, d in groups:
        if d < last:  # шум: показание не может уменьшаться при росте позиции
            continue
        last = d
        curve.append(((x0 + x1) / 2, d))
    return curve


def parse_mm(text):
    """"178mm" → 178.0; None, если разобрать нельзя."""
    if not isinstance(text, str):
        return None
    m = re.search(r"[0-9]+(?:\.[0-9]+)?", text)
    return float(m.group()) if m else None


def distance_from_x(x):
    """Показание расстояния для позиции фокуса (линейно по ln D между точками кривой)."""
    x = min(max(x, _XS[0]), _XS[-1])
    i = max(1, bisect.bisect_left(_XS, x))
    x0, x1 = _XS[i - 1], _XS[i]
    t = (x - x0) / (x1 - x0)
    return math.exp(_LN_D[i - 1] + t * (_LN_D[i] - _LN_D[i - 1]))


def x_from_distance(d):
    """Позиция фокуса для показания расстояния (обратная функция)."""
    ln = math.log(max(d, 1.0))
    ln = min(max(ln, _LN_D[0]), _LN_D[-1])
    i = max(1, bisect.bisect_left(_LN_D, ln))
    t = (ln - _LN_D[i - 1]) / (_LN_D[i] - _LN_D[i - 1])
    return _XS[i - 1] + t * (_XS[i] - _XS[i - 1])


def slope_ln_d(x):
    """Наклон ln(расстояние) по позиции фокуса на базовой кривой (в позиции x без сдвига шкалы)."""
    x = min(max(x, _XS[0] + 1e-6), _XS[-1] - 1e-6)
    i = max(1, bisect.bisect_left(_XS, x))
    return (_LN_D[i] - _LN_D[i - 1]) / (_XS[i] - _XS[i - 1])


class FocusModel:
    """Постоянная C (расстояние ≈ C · f / размер лица) и два уточнения; хранится в файле.

    • Таблица C по размеру лица: C зависит от расстояния, и после каждой наводки по резкости в ячейку записывается
      показание расстояния камеры в этой точке. Это физическая величина: она не зависит от дрейфа шкалы Focus.
    • Сдвиг шкалы `shift`: позиция Focus, дающая то же показание расстояния, со временем уходит (после многих наводок камеры
      на 0.09 за вечер). Его оценивает движок по обратной связи: позиция команды минус позиция по базовой кривой для показания.
    Размер лица — доли кадра (среднее геометрическое рамки и расстояния между глазами).
    """
    BIN = 0.06           # ширина ячейки таблицы по ln(размер/фокусное)
    REACH = 0.35         # дальше этого от последней ячейки (по ln) значение возвращается к общему C

    def __init__(self, path=None, maxlen=15, lens=None):
        self.path, self.lens = path, lens
        self.samples = deque(maxlen=maxlen)
        self.cbins = {}  # номер ячейки → C
        self.shift = 0.0
        self.qb = {}  # зум (фокусное, ступени по 3 мм) → лучшая виденная резкость лица × размер лица (мера «картинка резкая»)
        self.sigma_ln, self.n_ref = SIGMA_LN, N_REF  # из профиля объектива
        if path:
            data = self._read_all().get("lenses", {}).get(lens or "unknown", {})
            try:
                self.samples.extend(float(v) for v in data.get("samples", []))
                self.cbins = {int(k): float(v) for k, v in data.get("cbins", {}).items()}
                self.shift = float(data.get("shift", 0.0))
                self.qb = {str(k): float(v) for k, v in data.get("qb", {}).items()}
            except (ValueError, AttributeError):
                pass

    def _read_all(self):
        """Файл пользователя: {"lenses": {slug: данные}}. Старый плоский формат (один объектив) переносится в Sigma."""
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        if "lenses" not in data and "samples" in data:
            return {"lenses": {LEGACY_SLUG: data}}
        return data

    def save(self):
        if not self.path:
            return
        try:
            all_ = self._read_all()
            all_.setdefault("lenses", {})[self.lens or "unknown"] = {
                "samples": [round(v, 4) for v in self.samples],
                "cbins": {str(k): round(v, 4) for k, v in self.cbins.items()},
                "shift": round(self.shift, 4),
                "qb": {k: round(v, 3) for k, v in self.qb.items()}}
            with open(self.path, "w") as f:
                json.dump(all_, f)
        except OSError:
            pass

    @staticmethod
    def _qkey(focal_mm):
        return str(int(round(focal_mm / 3.0)))

    def q_get(self, focal_mm):
        """Лучшая виденная «резкость × размер» для этого зума; для нового зума — 80% ближайшего известного (или None)."""
        k = self._qkey(focal_mm)
        if k in self.qb:
            return self.qb[k]
        if not self.qb:
            return None
        near = min(self.qb, key=lambda kk: abs(int(kk) - int(k)))
        return 0.8 * self.qb[near]

    def q_verified(self, focal_mm, verified):
        """Обновить «лучшее» по проверенному результату (найденный пик или наводка камеры), а не по любому скачку резкости."""
        k = self._qkey(focal_mm)
        cur = self.qb.get(k)
        if cur is None:
            self.qb[k] = verified
        elif verified > cur:  # вверх быстро (но не больше ×1.3 за раз), вниз очень медленно: мягкий результат норму не обесценивает
            self.qb[k] = min(cur + (verified - cur) * 0.5, 1.3 * cur)
        else:
            self.qb[k] = cur + (verified - cur) * 0.05
        self.save()

    @property
    def C(self):
        return statistics.median(self.samples) if self.samples else None

    def add(self, d_mm, size, focal_mm):
        """Добавить замер после удачной наводки. Возвращает True, если замер принят."""
        if not (d_mm and size and focal_mm) or d_mm >= INFINITY_MM:
            return False
        self.samples.append(d_mm * size / focal_mm)
        self.save()
        return True

    def learn_c(self, d_mm, size, focal_mm):
        """Запомнить C для этого размера лица по показанию камеры в точке лучшей резкости."""
        if not (d_mm and size and focal_mm) or d_mm >= INFINITY_MM:
            return
        k = round(math.log(size / focal_mm) / self.BIN)
        c = d_mm * size / focal_mm
        self.cbins[k] = c if k not in self.cbins else 0.5 * self.cbins[k] + 0.5 * c
        self.save()

    def c_at(self, size, focal_mm):
        """C для данного размера лица: линейно между ближайшими заполненными ячейками, вдали — общий C."""
        base = self.C
        if not self.cbins:
            return base
        pos = math.log(size / focal_mm) / self.BIN
        keys = sorted(self.cbins)
        lo = [k for k in keys if k <= pos]
        hi = [k for k in keys if k >= pos]
        if lo and hi:
            a, b = lo[-1], hi[0]
            if a == b:
                return self.cbins[a]
            t = (pos - a) / (b - a)
            return self.cbins[a] * (1 - t) + self.cbins[b] * t
        near = lo[-1] if lo else hi[0]
        fade = max(0.0, 1.0 - max(0.0, abs(pos - near) - 0.5) * self.BIN / self.REACH)  # в пределах половины ячейки — целиком
        return base + (self.cbins[near] - base) * fade if base else self.cbins[near]

    def sigma_x(self, x, fnumber=None):
        """Ширина колокола резкости в позициях фокуса вокруг x. На близком плане шкала «растянута», колокол шире;
        при более закрытой диафрагме глубина резкости больше, колокол шире пропорционально числу диафрагмы."""
        scale = (fnumber / self.n_ref) if (fnumber and self.n_ref) else 1.0
        s = self.sigma_ln * scale / max(slope_ln_d(x - self.shift), 1e-3)
        return min(max(s, 0.015), 0.07 * max(scale, 1.0))

    def target_distance(self, size, focal_mm):
        return self.c_at(size, focal_mm) * focal_mm / size

    def target_x(self, size, focal_mm):
        """Позиция Focus для лица заданного размера: по базовой кривой для целевого расстояния плюс текущий сдвиг шкалы."""
        return x_from_distance(self.target_distance(size, focal_mm)) + self.shift
