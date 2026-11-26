"""Tracking autofocus as a controlled module: can be turned on and off from the dashboard.

The logic is the same as in tracker.py: sharpness and face size → when there is defocus or shift, we launch one-shot AF of the camera.
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

# precise mode
FOLLOW_DEAD = 0.01 # do not move focus if the target position has moved less than this (about a lens stop)
FOLLOW_EVERY = 0.15 # at rest: no more than one command in as many seconds
# smoothness: while the face moves, the target is pulled up exponentially, the commands move in small steps and more often
MOVE_V = 0.22 # rate of change of face size ln (1/s), above which we consider movement (size noise of 2-3% gives false 0.06-0.15)
MOVE_HOLD = 0.6 # after the last sign of movement, the movement mode is held for so many seconds
MOVE_JUMP = 0.025 # or the target has moved more than this from the current position
MOVE_DEAD = 0.004 # when moving, the command threshold (at rest - FOLLOW_DEAD, so that the focus does not shake due to noise)
MOVE_EVERY = 0.08 # in team movement no more often than once every so many seconds
SMOOTH_TAU = 0.12 # target smoothing time constant, s
PROBE_STEP = 0.012 # sharpening step (approximately one lens step)
PROBE_SETTLE = 0.3 # wait after the command until the lens arrives
REFINE_BELOW = 0.55 # sharpness is consistently below 55% of the standard (adjusted for face size): to clarify
REFINE_COOLDOWN = 4.0
REFINE_HOLD = 8.0 # if the refinement showed that the position is already the best, do not repeat for so many seconds
GAIN = 1.12 # position changes only if sharpness has increased by at least 12% (metering noise)
LATENCY = 0.2 # advance during movement, s (delay of command and lens travel)
PREDICT_MIN_V = 0.25 # rate of change of face size ln (1/s), below which there is no anticipation
PREDICT_MAX = 0.12 # maximum lead by ln size
SIZE_SOURCE = "eyes" # face size: "eyes" - geometric average of the frame and distance between the eyes, "box" - only the frame
SHARP_SOURCE = "face" # where to measure sharpness: "face" - the entire face (the eyes turned out to be noisier: blinking), "eyes" - the eye area
SIGMA = 0.022 # sharpness bell width at focus position (according to sharpness maps: half peak about ±0.025)
GOOD = 0.85 # picture is at least 85% the best: rechecking after moving and using a timer is not needed
WIDE = 0.03 # wide first search step (when the picture is soft or the face has just returned)
QLOW = 0.6 # sharpness×size below 60% of the best seen - the picture is soft, needs clarification
ZOOM_QUIET = 1.3 # zoom is considered stopped if the focal length has not changed for so many seconds
MOVED_SINCE = 0.08 # face size has changed by more than this amount (in ln) since the last checked position: recheck
PERIODIC = 90.0 # even at rest, recheck the position at least once every so many seconds (the focus scale goes away over time)
VERIFIED_TTL = 900.0 # sharpness-verified peak positions are stored for so many seconds (the focus scale goes away over time)
W_BASE = 0.02 # weight of the “base” estimate (last point or model checked) next to the peaks found
VERIFIED_TAU = 60.0 # the weight of the found peak decreases with age (the focus scale goes away: old peaks are pulled back to their previous position)
VERIFIED_STALE = 0.03 # fresh check differs from old entry on similar face size larger than this: old entry is out of date
VERIFIED_NEAR = 0.7 # use the peak at the closest distance in ln(size/zoom) if the difference is no more than that
SHIFT_STEP = 0.012 # scale shift is updated only if the score has changed by more than this
X_MIN, X_MAX = 0.3, 0.95 # Focus operating range (the scale goes by ~0.06–0.09 over time, so the margin is higher than the previous 0.93)

# subject in the center of the frame (when the face is covered)
ROI = (0.3, 0.25, 0.7, 0.75) # central area of ​​the frame (left, top, right, bottom edge in fractions)
LOST_AFTER = 1.2 # there is no face for so many seconds - go to the object in the center
BACK_AFTER = 0.4 # the face is visible again for so many seconds - return to the face
MANUAL_HOLD = 10.0 # "Point Now" keeps focus on the center for that many seconds, even if the face is visible
OBJ_CHANGE = 0.7 # how much the contents of the center should change (0 - the same, ~1 - completely different): new item
OBJ_STILL = 0.15 # change between frames below this - the object is stationary

# sharp_drop: sharpness < fraction of reference = defocus; size: change the size of the face; need: how many seconds the condition lasts
PROFILES = {
    "soft": {"sharp_drop": 0.45, "size": 0.18, "need": 0.7},
    "normal": {"sharp_drop": 0.55, "size": 0.12, "need": 0.35},
    "fast": {"sharp_drop": 0.65, "size": 0.08, "need": 0.25},
}
STILL_FOR, STILL_TOL = 0.4, 0.03 # face “stopped”: size is stable for so many seconds with tolerance
FORCE_AFTER = 2.0 # if the movement does not stop, start AF anyway
COOLDOWN = 1.0
MIN_AF_TIME, MAX_AF_TIME = 0.7, 3.5
SETTLE_SAMPLES, SETTLE_TOL = 6, 0.15
MIN_GOOD, MAX_RETRIES = 30.0, 2
DETECT_WIDTH = 640


def measure(frame):
    """(sharpness, face width, frame, distance between eyes, face sharpness, eye sharpness) or None if there is no face.

    Sharpness in the first field is taken from the eye area (SHARP_SOURCE = "eyes"): it depends less on facial expressions and conversation
    and coincides with what the focus really needs. Distances and frame are in fractions of a frame.
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
        if ecrop.shape[0] >= 12 and ecrop.shape[1] >= 24: # too small area: noise, take the face
            sharp_eyes = tenengrad(cv2.cvtColor(ecrop, cv2.COLOR_BGR2GRAY))
    # the central part of the face (from the eyes to the mouth): there is almost no background or hair in it, in the background the frame of the face is “contaminated” with it
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
    def eye_sharp(box): # sharpen one eye (None if the area is too shallow)
        if box is None:
            return None
        ex, ey, ew, eh = (int(v / scale) for v in box)
        c = frame[max(ey, 0):ey + eh, max(ex, 0):ex + ew]
        return tenengrad(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)) if c.shape[0] >= 8 and c.shape[1] >= 12 else None
    return (sharp, fw / w, (max(x, 0) / w, max(y, 0) / h, fw / w, fh / h), (iod / DETECT_WIDTH) if iod else None,
            sharp_face, sharp_eyes, sharp_core, eye_sharp(eye_l), eye_sharp(eye_r), yaw, roll)


def roi_metrics(frame):
    """The sharpness of the central area and its "imprint" (24x18, without brightness and contrast): it shows that there is another object in the center."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame[int(ROI[1] * h):int(ROI[3] * h), int(ROI[0] * w):int(ROI[2] * w)], cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (24, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
    return tenengrad(gray), (small - small.mean()) / (small.std() + 1e-3)


class LensChanged(Exception):
    """The lens on the camera has been changed: you need to load its profile."""


class FaceAF:
    def __init__(self, link, grab):
        self.link, self.grab = link, grab
        self.profile = "normal"
        self.enabled = False
        self.task = None
        self.mode = "precise" # precise (main): focus by face size + refinement by sharpness; camera: one-shot AF camera
        self.status = {"state": "off", "sharp": None, "base": None, "width": None, "mode": "precise", "x": None,
                       "C": None, "af_count": 0, "last_af_s": None, "fps": 0, "box": None}
        self._last_stamp = 0.0
        self._samples = 0
        self._rebase = False
        self._roi = None # (center sharpness, imprint) of last frame (fine mode)
        self._manual_until = 0.0 # until this point the focus is kept on the center (“Hover now”)
        self.trace = collections.deque(maxlen=14000) # by frame: (time, size, sharpness, focus position, state, number of commands) - for parsing
        self._cmds = 0
        self._verified_at = 0.0 # when the built-in AF last produced a verified result (counts loop)
        self._fallback = False # the camera does not send the distance to the lens: we use the camera’s built-in AF
        self._eye_ratio = 0.395 # ratio “distance between the eyes / width of the face frame”, refined along the way

    # ---- control ----
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
            self._publish(state="off", box=None, sharp=None, base=None)

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
        """Manual AF start; if tracking is enabled, the target will be captured again."""
        await self.link.send(AF_PACKET)
        if self.mode == "precise": # built-in AF aims at the center: leave focus on the subject in the center for a while
            self._manual_until = time.monotonic() + MANUAL_HOLD
        else:
            self._rebase = True
        self._publish(af_count=self.status["af_count"] + 1)

    async def sweep(self, lo=0.7, hi=0.95, step=0.01, settle=0.7):
        """Looping through focus positions: face sharpness and camera distance indication (12.12). For calibration and debugging.

        Tracking must be turned off for this time (this is done by the caller).
        """
        rows = []
        self._publish(sweep=None, state="scanning focus range...")
        await self.link.send(packet(0, 0, 128, [max(0.0, lo - 0.05)])) # approach is always from below
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
        self._publish(sweep=rows, state="off")

    async def backlash_test(self, x, delta=0.06, reps=3):
        """Sharpness at the same position x when approaching from below and from above (checking backlash). Tracking should be turned off."""
        out = {"snizu": [], "sverhu": []}
        self._publish(backlash=None, state="checking backlash...")
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
        self._publish(backlash=out, state="off")

    # ---- internal logic ----
    def _analyze(self, frame):
        m = measure(frame)
        if self.mode == "precise":
            self._roi = roi_metrics(frame)
        return m

    async def _next_sample(self):
        loop = asyncio.get_running_loop()
        while True:
            if not self.grab.available: # camera turned off: let the main loop update the status
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
            self._publish(state="focusing..." if not attempt else "retrying AF...")
            t0 = time.monotonic()
            await self.link.send(AF_PACKET)
            self.status["af_count"] += 1
            base = await self._settle()
            if base is None:
                return None
            self._publish(last_af_s=round(time.monotonic() - t0, 1))
            self.link.add_log(f"AF ({reason}) for {time.monotonic() - t0:.1f} s, sharpness {base[0]:.0f}")
            if base[0] >= MIN_GOOD:
                return base
        return base

    # ---- fine mode ----
    def _focal_mm(self):
        return parse_mm(self.link.state.get("12.11")) or 16.0

    async def _send_focus(self, x):
        x = min(max(x, X_MIN), X_MAX)
        await self.link.send(packet(0, 0, 128, [x]))
        self._cmds += 1
        return x

    async def _probe(self, x, size_ref):
        """Set focus at x and measure the sharpness of the face. None - the face moved, the metering is invalid."""
        x = await self._send_focus(x)
        await asyncio.sleep(PROBE_SETTLE)
        await self._next_sample() # discard the frame while the lens is moving
        sh, sz = [], []
        for _ in range(3):
            m = await self._next_sample()
            if m:
                sh.append(m[0])
                sz.append(self._size_of(m))
        if len(sh) < 2 or abs(statistics.median(sz) / size_ref - 1) > 0.08:
            return None
        self.trace.append((round(time.monotonic(), 3), round(statistics.median(sz), 4), round(statistics.median(sh), 1),
                           round(x, 4), "refining...", self._cmds))
        return x, statistics.median(sh)

    @staticmethod
    def _log_parabola_peak(pts):
        """The vertex of the parabola in ln (sharpness) through the three best neighboring points; None, if not convex."""
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
        """The position of the peak of the sigma width bell according to two sharpness measurements."""
        if sa <= 0 or sb <= 0 or xa == xb:
            return None
        return (xa + xb) / 2 + sigma ** 2 * math.log(sb / sa) / (xb - xa)

    async def _refine(self, x0, s0, size_ref, prefer, wide=False, sigma=SIGMA):
        """Find the peak of sharpness in two or three tests, moving monotonously (without rocking back and forth over long distances).

        The step of the first sample is a fraction of the width of the bell sigma (no more than 0.03: large samples are noticeable on the screen). If she ain't better
        original, the reverse side is checked. Based on two points with growth, the position of the peak is calculated, we go there with one test;
        the result is the top of the parabola in terms of ln (sharpness) or the best point. s0 - sharpness at x0 (it has already been measured by tracking).
        Returns (position, sharpness, direction: +1/−1, 0 - stayed in place) or None if the face moved.
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
            r2 = await probe(x0 - prefer * d) # the first probe is no better: the peak may be on the other side
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
            if abs(nearest - xp) < 0.006: # have already been there
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
        """Timing the camera's built-in AF and writing the result to the model. Returns (sharpness, size) or None."""
        try:
            base = await self._run_af("model training")
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
        """Face size in fractions of the frame: the distance between the eyes reduced to the width of the frame (or the width of the frame itself)."""
        if SIZE_SOURCE == "eyes" and self.mode == "precise" and not self._fallback and len(m) > 3 and m[3]:
            return math.sqrt(m[1] * m[3] / self._eye_ratio) # frame and eye errors are independent: the geometric mean is more accurate
        return m[1]

    @staticmethod
    def _slope(points):
        """Slope of ln size over time (1/s) by least squares; None if there is little data."""
        if len(points) < 8 or points[-1][0] - points[0][0] < 0.25:
            return None
        t0 = points[0][0]
        xs = [t - t0 for t, _ in points]
        ys = [v for _, v in points]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None

    async def _settle_roi(self):
        """Wait until center sharpness stabilizes after AF. Returns (sharpness, fingerprint) or None."""
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
        """One frame in the "subject in the center" mode: we monitor the sharpness of the center and trigger the camera's AF when it changes or becomes out of focus."""
        sharp, sig = self._roi
        if o["manual"] and o["sig"] is None: # AF on the button has already been sent: the current image is the original one, no need for a second AF
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
        if o["manual"] and still: # AF on the button has already been sent: we take the picture after it as a standard, a second AF is not needed
            o["ref"], o["sig"], o["manual"] = statistics.median(o["hist"]), sig, False
            o["last_af"] = now
        changed = o["sig"] is None or float(np.abs(sig - o["sig"]).mean()) > OBJ_CHANGE
        blur = o["ref"] is not None and len(o["hist"]) >= 20 and statistics.median(o["hist"]) < REFINE_BELOW * o["ref"]
        self._publish(state="subject in the center", sharp=round(sharp, 1), base=round(o["ref"], 1) if o["ref"] else None,
                      box=None, x=None)
        self.trace.append((round(now, 3), 0.0, round(sharp, 1), None, "subject in the center", self._cmds))
        if still and (changed or blur) and now - o["last_af"] >= (4.0 if changed else 5.0):
            self._publish(state="focusing on center subject...")
            t0 = time.monotonic()
            await self.link.send(AF_PACKET)
            self.status["af_count"] += 1
            res = await self._settle_roi()
            o["last_af"] = time.monotonic()
            o["hist"].clear()
            if res:
                o["ref"], o["sig"] = res
                self._publish(last_af_s=round(time.monotonic() - t0, 1))
                self.link.add_log(f"subject in center: AF for {time.monotonic() - t0:.1f} s, sharpness {res[0]:.0f}")

    # ---- lens: profile, auto-calibration ----
    def _lens_name(self):
        v = self.link.state.get("12.9")
        return v if isinstance(v, str) else ""

    def _fnumber(self):
        return parse_mm(self.link.state.get("12.10"))

    async def calibrate_lens(self, slug, name):
        """Auto-calibration of the new lens: search Focus 0...0.97, a curve is plotted based on the distance readings (12.12).

        The face is not needed; the focus travels the entire stroke (about a minute), so it is performed once for a new lens.
        If the camera does not send distances for this lens, the profile is marked no_distance and tracking is carried out by the camera’s built-in AF.
        """
        self._publish(state="calibrating a new lens (about a minute)...", box=None)
        self.link.add_log(f"new lens “{name}”: calibration (focus search 0…1)")
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
                   "notes": "taken by auto-calibration (searching Focus 0…0.97)"}
        if len(curve) < 4:
            profile["no_distance"] = True
            self.link.add_log("camera does not send distance for this lens: camera built-in AF tracking")
        else:
            profile["curve"] = [[round(x, 4), d] for x, d in curve]
            self.link.add_log(f"calibration ready: {len(curve)} curve points")
        save_profile(slug, profile)
        return profile

    async def _prepare_lens(self):
        """Wait for the camera and lens name, load the profile (calibrate if necessary). Returns (slug, profile)."""
        for _ in range(40): # wait up to 20 seconds for connection and lens name
            if self.link.connected and self._lens_name():
                break
            await asyncio.sleep(0.5)
        name = self._lens_name()
        slug = lens_slug(name)
        profile = load_profile(slug)
        if profile is None:
            profile = await self.calibrate_lens(slug, name or "unknown lens")
        if profile.get("curve"):
            set_curve(profile["curve"])
        self._publish(lens=name or "unknown lens")
        return slug, profile

    async def _loop_precise(self):
        slug, profile = await self._prepare_lens()
        if profile.get("no_distance"): # no distance: precise mode not possible, camera's built-in AF works
            self._fallback = True
            self._publish(state="following (AF camera: lens without distance)")
            await self._loop()
            return
        self._fallback = False
        model = FocusModel(MODEL_PATH, lens=slug)
        model.sigma_ln, model.n_ref = profile.get("sigma_ln", SIGMA_LN), profile.get("n_ref", N_REF)
        x_cmd = ref = anchor = ref_hi = None # focus command; standard of sharpness (sharpness, size); anchor (position, size)
        last_cmd = 0.0
        last_refine = time.monotonic() # first refinement not immediately after enabling
        prefer = 1 # which direction to try first when clarifying
        recent = collections.deque(maxlen=5)
        slow = collections.deque(maxlen=30) # sharpness for the last ~1.5 s: we do not react to one frame with facial expressions
        widths = collections.deque() # (time, size) for STILL_FOR seconds: "dead"
        trend = collections.deque() # (time, size ln) for 0.5 s: movement speed
        ratios = collections.deque(maxlen=300) # eye/frame ratio of this person
        self._eye_ratio = 0.395
        noise_box, noise_eyes = collections.deque(maxlen=40), collections.deque(maxlen=40)
        noise_sf, noise_se = collections.deque(maxlen=40), collections.deque(maxlen=40) # sharpening noise: face / eyes
        win_box, win_eyes = collections.deque(maxlen=5), collections.deque(maxlen=5) # windows for smoothing the compared signals
        win_sf, win_se = collections.deque(maxlen=10), collections.deque(maxlen=10)
        quality = collections.deque() # (time, in focus?) in 60s
        cmd_times, refine_spans = collections.deque(), collections.deque() # command times and refinement intervals
        t_start = time.monotonic()
        t_prev = x_s = None # last frame time and smoothed focus target
        moving_until = 0.0 # until this moment the face is considered moving
        nochange = 0 # consecutive clarifications “without shift” at this place: the pause grows 8, 16, 32 s
        verified = [] # (time, size, zoom, position) of peaks found by sharpness: set focus to the closest one when returning to this distance
        last_seed = -1e9 # when was the last time you tried aiming the camera instead of searching by sharpness
        hold_until = 0.0 # after “no shift”, repeat the check in the same place early (but not after moving)
        last_shift_key = None
        last_verified = 0.0 # when the focus position was last checked for sharpness
        f_prev, f_changed_at, zoom_dirty = None, -1e9, False # zoom was changed recently: we don’t change focus, then we check it again
        shift_hist = collections.deque(maxlen=8) # latest scale shift estimates
        face_sig = None # imprint of the center of the frame while the face was visible
        obj = face_back = None # state of the “object in the center” mode (None - face tracking)
        face_seen = t_start
        last_fps_t, last_fps_n, last_q = t_start, 0, t_start
        self._publish(state="looking for a face", C=round(model.C, 3) if model.C else None)

        def reset_windows():
            recent.clear()
            widths.clear()
            slow.clear()
            trend.clear()

        while True:
            if not self.link.connected:
                self._publish(state="waiting for camera", box=None)
                await asyncio.sleep(0.5)
                x_cmd = None
                continue
            if not self.grab.available:
                self._publish(state="no video from camera", box=None)
                await asyncio.sleep(0.5)
                x_cmd = None
                continue
            m = await self._next_sample()
            now = time.monotonic()
            if self._lens_name() and lens_slug(self._lens_name()) != slug:
                raise LensChanged() # lens changed on the fly: reread profile
            if now - last_fps_t > 1.0:
                self.status["fps"] = round((self._samples - last_fps_n) / (now - last_fps_t))
                last_fps_t, last_fps_n = now, self._samples
            # face ↔ object in the center: the face is hidden (or “Point now” is pressed) - focus on the center; the face came back - back
            if m is not None:
                face_seen = now
                if self._roi is not None:
                    face_sig = self._roi[1] # what the center of the frame with a face looked like: we count from it “another object has appeared”
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
                    self.link.add_log("hold on center: focus on object in center" if manual
                                      else "face obscured: focusing on center subject")
                try:
                    await self._object_step(obj, now)
                except LinkError:
                    await asyncio.sleep(0.5)
                continue
            if obj is not None: # the face is back: the model immediately focuses on the face, without searching
                obj = None
                x_cmd = None
                ref = ref_hi = None # the sharpness standard was before leaving: we’ll recheck the position as soon as the face freezes
                last_refine = now - REFINE_COOLDOWN + 1.0
                reset_windows()
                self.link.add_log("face back in frame: focus on face")
            if m is None:
                self._publish(state="looking for a face", box=None)
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
            if abs(f - f_prev) >= 0.5: # zoom changes: the size of the face does not grow because you are closer
                f_prev, f_changed_at, zoom_dirty = f, now, True
                slow.clear()
            zooming = now - f_changed_at < ZOOM_QUIET # the ring is rotated with pauses: “stopped” - when it’s quiet longer
            ws = [w for _, w in widths]
            still = len(widths) > 3 and now - widths[0][0] >= STILL_FOR * 0.8 \
                and (max(ws) - min(ws)) < STILL_TOL * statistics.median(ws)
            if still:
                slow.append(m[0])
            scaled = ref[0] * ref[1] / size if ref else None # sharpness standard is recalculated to the current face size
            win_box.append(m[1])
            if len(m) > 3 and m[3]:
                win_eyes.append(m[3] / self._eye_ratio)
            if len(m) > 5 and m[4] and m[5]:
                win_sf.append(m[4])
                win_se.append(m[5])
            if still: # noise in two ways, based on smoothed signals, as the program sees them (at rest: the less, the better)
                if len(win_box) == 5:
                    noise_box.append(statistics.median(win_box))
                if len(win_eyes) == 5:
                    noise_eyes.append(statistics.median(win_eyes))
                if len(win_sf) == 10 and len(win_se) == 10:
                    noise_sf.append(statistics.median(win_sf))
                    noise_se.append(statistics.median(win_se))

            # quality metrics for the last 60 seconds
            quality.append((now, scaled is not None and sharp >= 0.7 * scaled)) # what the picture looks like, including motion blur
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
                if self._rebase: # manual launch of AF: it has already been sent, we are waiting for the result and learning from it
                    self._rebase = False
                    base = await self._settle()
                    if base and base[0] >= MIN_GOOD and model.add(parse_mm(self.link.state.get("12.12")), base[1], f):
                        ref, x_cmd = base, model.target_x(base[1], f)
                        anchor = (x_cmd, base[1])
                        ref_hi = ref
                    reset_windows()
                    continue

                if model.C is None: # there is no model yet: aim once with the built-in AF and learn
                    self._publish(state="training a model (AF camera)…")
                    base = await self._seed(model, f)
                    if base:
                        ref, x_cmd = base, model.target_x(base[1], f)
                        anchor = (x_cmd, base[1])
                        ref_hi = ref
                    reset_windows()
                    continue

                # zoom stopped: this lens' focus shifts when zooming. If the face is closer to the center, the camera's built-in AF
                # aims in ~1 s (small steps in sharpness at the long end took 15–30 s), and the model learns at the same time
                if (zoom_dirty and not zooming) or (ref is None and now - last_seed >= 8.0 and not zooming):
                    zoom_dirty = False
                    last_seed = now
                    bx, by, bw, bh = m[2]
                    if 0.3 <= bx + bw / 2 <= 0.7 and 0.2 <= by + bh / 2 <= 0.7:
                        self._publish(state="zoom stopped: focusing...")
                        base = await self._seed(model, f)
                        if base:
                            ref, x_cmd = base, model.target_x(base[1], f)
                            anchor = (x_cmd, base[1])
                            ref_hi = ref
                            last_cmd = last_refine = time.monotonic()
                        reset_windows()
                        continue
                    zoom_dirty = True # face is not centered: camera AF is not good, let wide recheck work

                # follow: target position based on face size with movement anticipation
                v = self._slope(trend)
                lead = 0.0
                if v is not None and abs(v) > PREDICT_MIN_V:
                    lead = max(-PREDICT_MAX, min(PREDICT_MAX, v * LATENCY))
                size_pred = size * math.exp(lead)
                # from the sharpness-checked point: the scale shift is reduced and camera responses (in steps) do not wobble the position.
                # Peaks found earlier at a similar face size (with a similar zoom, not older than VERIFIED_TTL) are averaged with weight,
                # smoothly decreasing with size differences. The choice of the “nearest” single peak jumped when the size jittered by 1% and gave hunting.
                verified[:] = [v for v in verified if now - v[0] < VERIFIED_TTL]
                x_cur = model.target_x(size_pred, f)
                if anchor is not None:
                    x_base = anchor[0] + x_cur - model.target_x(anchor[1], f)
                else:
                    x_base = x_cur
                num, den = W_BASE * x_base, W_BASE
                for pk in verified: # (name not v: there is the speed of the face movement)
                    if abs(pk[2] - f) >= 1.5:
                        continue
                    dl = math.log(size / pk[1])
                    if abs(dl) >= VERIFIED_NEAR:
                        continue
                    w = math.exp(-dl * dl / (2 * 0.1 ** 2)) * math.exp(-(now - pk[0]) / VERIFIED_TAU)
                    num += w * (pk[3] + x_cur - model.target_x(pk[1], f))
                    den += w
                x_t = min(max(num / den, X_MIN), X_MAX)
                # smooth following: at rest - a large threshold (without shaking), in motion - the target is pulled up exponentially
                # and the focus moves in small steps, and not “waiting for the threshold - jump”
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

                # clarification of sharpness when the face is frozen and the sharpness is consistently below the standard
                slow_med = statistics.median(slow) if len(slow) >= 24 else None
                qb = model.q_get(f)
                soft = slow_med is not None and qb is not None and slow_med * size < QLOW * qb
                after_zoom = zoom_dirty and not zooming # zoom stopped: with this lens the focus shifts slightly when zooming
                last_verified = max(last_verified, self._verified_at)
                # rechecking for sharpness does not depend on the “norm”: after moving (the size of the face has changed), and once a minute
                far = anchor is not None and abs(math.log(size / anchor[1])) > MOVED_SINCE
                periodic = now - last_verified >= PERIODIC
                below = slow_med is None or qb is None or slow_med * size < GOOD * qb # picture below 85% best (or no data)
                need = ref is None or soft or after_zoom or ((far or periodic) and below) \
                    or (slow_med is not None and slow_med < REFINE_BELOW * scaled)
                changed = after_zoom or (far and below) or ref is None # real change: a pause “without shift” does not delay it
                if still and need and not zooming and x_cmd is not None \
                        and (changed or now >= hold_until) \
                        and now - last_refine >= (1.0 if changed else REFINE_COOLDOWN):
                    self._publish(state="refining focus...")
                    t_ref = time.monotonic()
                    s0 = slow_med if slow_med is not None else sharp
                    res = await self._refine(x_cmd, s0, size, prefer, wide=after_zoom or (soft and ref is not None),
                                            sigma=model.sigma_x(x_cmd, self._fnumber()))
                    last_refine = time.monotonic()
                    refine_spans.append((t_ref, last_refine))
                    zoom_dirty = False
                    reset_windows()
                    if res is None: # the face moved during measurements: will repeat soon
                        x_cmd = await self._send_focus(x_cmd)
                        last_refine -= REFINE_COOLDOWN - 1.0
                        continue
                    x_star, s_star, moved = res
                    x_cmd = await self._send_focus(x_star)
                    last_cmd = time.monotonic()
                    cmd_times.append(last_cmd)
                    await asyncio.sleep(PROBE_SETTLE)
                    if s_star >= MIN_GOOD and (qb is None or s_star * size >= 0.45 * qb):
                        ref = (0.5 * (s_star + s0) if moved else s0, size) # not the maximum of samples: it is overestimated by noise
                        anchor = (x_cmd, size)
                        ref_hi = ref
                        last_verified = time.monotonic()
                        keep = []
                        for pk in verified: # old records are close in size, but with a different peak position: the scale is gone, they are outdated
                            if abs(pk[2] - f) < 1.5 and abs(math.log(size / pk[1])) < 0.15 \
                                    and abs(pk[3] + model.target_x(size, f) - model.target_x(pk[1], f) - x_cmd) > VERIFIED_STALE:
                                continue
                            keep.append(pk)
                        verified[:] = keep
                        for j, pk in enumerate(verified):
                            if abs(pk[2] - f) < 1.5 and abs(math.log(size / pk[1])) < 0.05: # same face size: clarify the entry
                                x_adj = pk[3] + model.target_x(size, f) - model.target_x(pk[1], f)
                                verified[j] = (last_verified, size, f, 0.5 * x_adj + 0.5 * x_cmd)
                                break
                        else:
                            verified.append((last_verified, size, f, x_cmd))
                        del verified[:-30]
                        model.q_verified(f, ref[0] * ref[1]) # “sharp picture” standard - only based on verified results
                        model.add(parse_mm(self.link.state.get("12.12")), size, f)
                        d_obs = parse_mm(self.link.state.get("12.12"))
                        model.learn_c(d_obs, size, f)
                        if d_obs and d_obs < INFINITY_MM: # scale shift - only according to the position checked for sharpness, slowly
                            s_new = x_cmd - x_from_distance(d_obs)
                            if abs(s_new - model.shift) < 0.15:
                                model.shift = 0.7 * model.shift + 0.3 * s_new
                                model.save()
                        if moved:
                            prefer = moved
                            nochange = 0
                        else: # we couldn’t find a better one: the sharpness dropped due to other reasons (light, facial expressions) or the peak is already nearby;
                            nochange += 1 # repeat at this point less and less often, so that periodic hunting does not happen
                            hold_until = time.monotonic() + REFINE_HOLD * 2 ** min(nochange - 1, 3)
                        self.link.add_log(f"fine aiming: x={x_cmd:.3f}, sharpness {s_star:.0f}, "
                                          f"{'step ' + ('+' if moved > 0 else '−') if moved else 'no shift'}, "
                                          f"C={model.C:.2f}")
                    else:
                        self.link.add_log("precise aiming did not find sharpness: camera AF helps")
                        base = await self._seed(model, f)
                        if base:
                            ref, x_cmd = base, model.target_x(base[1], f)
                            anchor = (x_cmd, base[1])
                        ref_hi = ref
                    continue
                self.status["dbg"] = {"still": still, "slow_n": len(slow), "qbest": round(qb, 1) if qb else None,
                                      "q": round((slow_med or 0) * size, 1), "soft": bool(soft),
                                      "since_refine": round(now - last_refine, 1), "spread": round(100 * (max(ws) - min(ws)) / statistics.median(ws), 1) if ws else None}
                self._publish(state="following")
                if ref and still and sharp >= REFINE_BELOW * scaled:
                    # the reference slowly follows the normal level of sharpness in both directions (not just noise peaks);
                    # as long as the picture is noticeably worse than the threshold, it does not change so as not to “get used to” the out of focus
                    new0 = ref[0] + 0.02 * (sharp * size / ref[1] - ref[0])
                    if ref_hi: # down no further than 80% of the tested level: otherwise we will take a soft picture as the norm
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
                    self.link.add_log("lens changed: loading its profile")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc() # to permanent service log
            self.enabled = False
            self._publish(state=f"error: {e}", box=None)

    async def _loop(self):
        base = None
        last_af, bad_since = 0.0, None
        recent = collections.deque(maxlen=3)
        widths = collections.deque()
        last_fps_t, last_fps_n = time.monotonic(), 0
        self._publish(state="looking for a face")
        while True:
            if not self.link.connected:
                self._publish(state="waiting for camera", box=None)
                await asyncio.sleep(0.5)
                base = None
                continue
            if not self.grab.available:
                self._publish(state="no video from camera", box=None)
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
                self._publish(state="looking for a face", box=None)
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
                    base = await self._run_af("primary aiming") or base
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
            self._publish(state="following", sharp=round(sharp, 1), base=round(base[0], 1),
                          width=round(width, 3), box=m[2])

            if blur or moved:
                bad_since = bad_since or now
                held = now - bad_since
                if now - last_af >= COOLDOWN and ((held >= p["need"] and still) or held >= FORCE_AFTER):
                    try:
                        base = await self._run_af("defocus" if blur else "face changed size") or base
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
