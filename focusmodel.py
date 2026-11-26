"""Focus model: Focus position (0..1) ↔ distance shown by the camera (parameter 12.12), and relationship with face size.

For Pocket 4K + Sigma 18-35 via Metabones: the camera responds repeatably to the absolute Focus command and sends
the distance reached by a line of the form "178mm" in steps. Distance to face ≈ C f / width of face in frame,
where f is the focal length (12.11), and C is learned from the aiming results (median of the last measurements).
"""
import bisect
import json
import math
import os
import re
import statistics
from collections import deque

# Measured Focus curve → distance reading (plateau centers; camera gives readings in steps).
# Shot by searching 0..1 on this lens; the position of the plateau depends on the zoom by ±0.01–0.02, this is specified by sharpness.
CURVE = [(0.0, 56.0), (0.2, 59.0), (0.3, 64.0), (0.4, 69.0), (0.5, 80.0), (0.6, 93.0), (0.7, 124.0), (0.72, 137.0),
         (0.755, 154.0), (0.79, 178.0), (0.815, 212.0), (0.835, 262.0), (0.87, 354.0), (0.905, 593.0),
         (0.935, 10389.0)]
_XS = [p[0] for p in CURVE]
_LN_D = [math.log(p[1]) for p in CURVE]
SIGMA_LN = 0.233 # sharpness bell width by ln(distance): at focus positions it is divided by the slope of the curve
N_REF = 1.2 # aperture number at which SIGMA_LN is measured (Sigma 18-35 with booster shows f/1.2); width ∝ aperture number
INFINITY_MM = 5000.0 # We consider the readings above to be “infinity”: not suitable for training
LEGACY_SLUG = "64x-sigma-dc-18-35-1-7-hsm" # user data in old flat format belongs to this lens
PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")


def lens_slug(name):
    """Lens name from camera (parameter 12.9) → profile file name."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "unknown"


def load_profile(slug):
    """Lens profile (focus position → distance curve, etc.) from profiles/<slug>.json or None."""
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
    """Make active the lens curve: [(Focus position, distance reading, mm), ...]."""
    global _XS, _LN_D
    pts = sorted((float(x), float(d)) for x, d in points)
    if len(pts) < 2:
        raise ValueError("the curve must have at least two points")
    _XS = [p[0] for p in pts]
    ln, last = [], -1e9
    for _, d in pts:
        last = max(last, math.log(max(d, 1.0))) # distance does not decrease with increasing position
        ln.append(last)
    _LN_D = ln


def curve_from_sweep(pts):
    """Looping through positions [(x, reading or None)] in ascending order x → curve points at the centers of the "plateau" (reading in steps)."""
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
        if d < last: # noise: reading cannot decrease as position increases
            continue
        last = d
        curve.append(((x0 + x1) / 2, d))
    return curve


def parse_mm(text):
    """"178mm" → 178.0; None, if it is impossible to make out."""
    if not isinstance(text, str):
        return None
    m = re.search(r"[0-9]+(?:\.[0-9]+)?", text)
    return float(m.group()) if m else None


def distance_from_x(x):
    """Distance display for focus position (linear in ln D between curve points)."""
    x = min(max(x, _XS[0]), _XS[-1])
    i = max(1, bisect.bisect_left(_XS, x))
    x0, x1 = _XS[i - 1], _XS[i]
    t = (x - x0) / (x1 - x0)
    return math.exp(_LN_D[i - 1] + t * (_LN_D[i] - _LN_D[i - 1]))


def x_from_distance(d):
    """Focus position for distance indication (inverse function)."""
    ln = math.log(max(d, 1.0))
    ln = min(max(ln, _LN_D[0]), _LN_D[-1])
    i = max(1, bisect.bisect_left(_LN_D, ln))
    t = (ln - _LN_D[i - 1]) / (_LN_D[i] - _LN_D[i - 1])
    return _XS[i - 1] + t * (_XS[i] - _XS[i - 1])


def slope_ln_d(x):
    """Slope ln(distance) at the focal position on the base curve (at position x without scale shift)."""
    x = min(max(x, _XS[0] + 1e-6), _XS[-1] - 1e-6)
    i = max(1, bisect.bisect_left(_XS, x))
    return (_LN_D[i] - _LN_D[i - 1]) / (_XS[i] - _XS[i - 1])


class FocusModel:
    """Constant C (distance ≈ C f / face size) and two refinements; stored in file.

    • Table C by face size: C depends on the distance, and after each focusing is recorded in the cell
      camera distance reading at that point. This is a physical quantity: it does not depend on the drift of the Focus scale.
    • Scale shift `shift`: the Focus position, giving the same distance reading, goes away over time (after many camera adjustments
      by 0.09 per evening). It is evaluated by the engine based on feedback: the position of the command minus the position along the base curve for the reading.
    The size of the face is a fraction of the frame (the geometric mean of the frame and the distance between the eyes).
    """
    BIN = 0.06 # table cell width by ln(size/focal)
    REACH = 0.35 # further than this from the last cell (by ln) the value returns to the general C

    def __init__(self, path=None, maxlen=15, lens=None):
        self.path, self.lens = path, lens
        self.samples = deque(maxlen=maxlen)
        self.cbins = {} # cell number → C
        self.shift = 0.0
        self.qb = {} # zoom (focal, 3 mm steps) → best seen face sharpness × face size (the “picture is sharp” measure)
        self.sigma_ln, self.n_ref = SIGMA_LN, N_REF # from lens profile
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
        """User file: {"lenses": {slug: data}}. The old flat format (single lens) is transferred to Sigma."""
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
        """Best seen "sharpness × size" for this zoom; for new zoom - 80% of nearest known (or None)."""
        k = self._qkey(focal_mm)
        if k in self.qb:
            return self.qb[k]
        if not self.qb:
            return None
        near = min(self.qb, key=lambda kk: abs(int(kk) - int(k)))
        return 0.8 * self.qb[near]

    def q_verified(self, focal_mm, verified):
        """Update the "best" based on the verified result (found peak or camera aim), and not on any sharpness jump."""
        k = self._qkey(focal_mm)
        cur = self.qb.get(k)
        if cur is None:
            self.qb[k] = verified
        elif verified > cur: # up quickly (but no more than ×1.3 at a time), down very slowly: a soft result does not devalue the norm
            self.qb[k] = min(cur + (verified - cur) * 0.5, 1.3 * cur)
        else:
            self.qb[k] = cur + (verified - cur) * 0.05
        self.save()

    @property
    def C(self):
        return statistics.median(self.samples) if self.samples else None

    def add(self, d_mm, size, focal_mm):
        """Add a measurement after a successful pointing. Returns True if the measurement is accepted."""
        if not (d_mm and size and focal_mm) or d_mm >= INFINITY_MM:
            return False
        self.samples.append(d_mm * size / focal_mm)
        self.save()
        return True

    def learn_c(self, d_mm, size, focal_mm):
        """Remember C for this face size according to the camera reading at the point of best sharpness."""
        if not (d_mm and size and focal_mm) or d_mm >= INFINITY_MM:
            return
        k = round(math.log(size / focal_mm) / self.BIN)
        c = d_mm * size / focal_mm
        self.cbins[k] = c if k not in self.cbins else 0.5 * self.cbins[k] + 0.5 * c
        self.save()

    def c_at(self, size, focal_mm):
        """C for a given face size: linear between the nearest filled cells, in the distance - common C."""
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
        fade = max(0.0, 1.0 - max(0.0, abs(pos - near) - 0.5) * self.BIN / self.REACH) # within half the cell - entirely
        return base + (self.cbins[near] - base) * fade if base else self.cbins[near]

    def sigma_x(self, x, fnumber=None):
        """The width of the sharpness bell at focus positions around x. In the close-up, the scale is “stretched”, the bell is wider;
        with a closed aperture, the depth of field is greater, the bell is wider in proportion to the aperture number."""
        scale = (fnumber / self.n_ref) if (fnumber and self.n_ref) else 1.0
        s = self.sigma_ln * scale / max(slope_ln_d(x - self.shift), 1e-3)
        return min(max(s, 0.015), 0.07 * max(scale, 1.0))

    def target_distance(self, size, focal_mm):
        return self.c_at(size, focal_mm) * focal_mm / size

    def target_x(self, size, focal_mm):
        """Focus position for a given face size: along the base curve for the target distance plus the current scale offset."""
        return x_from_distance(self.target_distance(size, focal_mm)) + self.shift
