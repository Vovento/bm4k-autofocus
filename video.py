"""Video from a camera via USB (UVC): it finds the Pocket 4K itself, picks it up when you turn it on and releases it when you turn it off."""
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
    """Allowed video source: Blackmagic Cinema/Pocket Cinema cameras only."""
    value = (name or "").casefold()
    return "pocket cinema camera" in value or ("blackmagic" in value and "camera" in value)


def blackmagic_camera_index(names, requested=None):
    """Blackmagic compatible index; explicit index cannot bypass whitelist."""
    if requested is not None:
        if 0 <= requested < len(names) and is_blackmagic_camera(names[requested]):
            return requested
        return None
    return next((i for i, name in enumerate(names) if is_blackmagic_camera(name)), None)


class Grabber(threading.Thread):
    """Reads frames continuously so that consumers always receive a fresh frame without queuing."""

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
    """Synthetic image: gradient and moving circle, ~25 frames/s."""

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
    """Is Pocket 4K connected via USB. ioreg responds in ~15 ms and always sees the current state."""
    try:
        out = subprocess.run(["/usr/sbin/ioreg", "-p", "IOUSB", "-w0"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "pocket cinema" in out.lower()


class VideoHub:
    """Source of frames for autofocus and preview.

    We look for the camera by name among macOS video devices (OpenCV numbers change when connecting an iPhone camera, etc.),
    so by default we will never grab the MacBook's built-in camera instead of the Pocket.

    The list of macOS video devices in a long-lived process is not updated (it does not see a camera turned on later, but
    the dead video stream remains turned off). Therefore, we monitor the camera’s switching on and off via USB (ioreg) and during this
    moment we restart the process: the new process sees the current list.
    """
    STALE_AFTER = 3.0 # so many seconds without frames - the video is considered unavailable
    SETTLE = 10.0 # the first seconds after the start we do not react to changes in the camera state
    CONFIRM = 2.0 # USB state should last this many seconds (the camera has time to “rise”)
    NO_LIST_AFTER = 6.0 # the camera is available via USB, but macOS does not show it in the list of video devices
    OPEN_DELAY = 4.0 # after the process starts, we wait so many seconds before opening the camera: it is still loading
    NO_FRAMES_AFTER = 8.0 # the stream is open, but no frames are flowing: reading is stuck, only restarting the process helps

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
        if g.stamp == 0.0: # there have been no frames yet: give the camera time to start
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
            print(f"Blackmagic cameras: {allowed or 'not found'}", flush=True)
            self._last_names = names
        index = blackmagic_camera_index(names, self.index)
        if self.index is not None and index is None:
            print(f"Camera #{self.index} rejected: BM4K Autofocus only accepts Blackmagic Cinema Camera", flush=True)
        return index

    def _open(self, index):
        g = Grabber(open_camera(index))
        g.start()
        return g

    def _restart(self, reason):
        print(f"Restarting process: {reason}", flush=True)
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
                        self._restart("camera on" if present else "camera off")
                else:
                    changed_since = None

                stale_tries = int(os.environ.get("BM4K_STALE", "0"))
                if present:
                    idx = await loop.run_in_executor(None, self._blackmagic_index)
                    # after failures we wait longer: the camera might not be ready
                    if idx is not None and self.grabber is None and now - t_start >= self.OPEN_DELAY + 3.0 * stale_tries:
                        print(f"Video: opening camera #{idx}", flush=True)
                        try:
                            self.grabber = await asyncio.wait_for(loop.run_in_executor(None, self._open, idx), 15)
                        except asyncio.TimeoutError: # opening the camera is stuck: the thread cannot be killed, only restarted
                            if stale_tries < 4:
                                os.environ["BM4K_STALE"] = str(stale_tries + 1)
                                self._restart(f"camera opening stuck (attempt {stale_tries + 1})")
                            raise
                        self.error = ""
                        os.environ.pop("BM4K_NOLIST", None)
                        self._notify()
                    if idx is None and self.grabber is None:
                        no_list_since = no_list_since or now
                        if settled and now - no_list_since >= self.NO_LIST_AFTER:
                            tries = int(os.environ.get("BM4K_NOLIST", "0"))
                            if tries < 3: # prevent infinite restarts
                                os.environ["BM4K_NOLIST"] = str(tries + 1)
                                self._restart("camera is available via USB, but is not visible in the list of video devices")
                            elif self.error != "no-list":
                                self.error = "no-list"
                                print("The camera is visible via USB, but not as a webcam: check the USB mode on the camera", flush=True)
                                self._notify()
                    else:
                        no_list_since = None
                    g = self.grabber
                    if g is not None:
                        age = now - (g.stamp or g.started)
                        if age > self.NO_FRAMES_AFTER:
                            if stale_tries < 4:
                                os.environ["BM4K_STALE"] = str(stale_tries + 1)
                                self._restart(f"video: no frames for {age:.0f} s (attempt {stale_tries + 1})")
                            elif self.error != "no-frames":
                                self.error = "no-frames"
                                print("Video: still no frames after several restarts", flush=True)
                                self._notify()
                        elif g.stamp and now - g.started > 5:
                            os.environ.pop("BM4K_STALE", None) # video started: reset the failure counter
                if self.grabber is not None and not present and self.grabber.running:
                    self.grabber.stop() # camera is turned off: the thread is dead (the process will restart itself)
            except SystemExit as e: # no access to camera, etc.
                if str(e) != self.error:
                    self.error = str(e)
                    self._notify()
            except Exception as e:  # noqa: BLE001
                if repr(e) != self.error:
                    self.error = repr(e)
                    self._notify()
            await asyncio.sleep(1.0)
