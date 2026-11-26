"""Diagnostics: frequent focus pass 0 → 1, sharpness of the face and two background areas, frames every 0.1 in snapshots/.

Run from Terminal.app: python debug_sweep.py --cam 0 [--step 0.02] [--settle 1.0]
Hold an object with text (book, sheet) motionless in the center of the frame at a distance from your face.
"""
import argparse
import asyncio
import json
import os

import cv2
from bleak import BleakClient

from bmcc import camera_names, detect_face, find_camera, latest_frame, open_camera, set_focus
from video import blackmagic_camera_index


def tenengrad(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return float((gx ** 2 + gy ** 2).mean()) / (float(gray.mean()) + 1e-6) ** 2 * 1000


def region(frame, x0, y0, x1, y1):
    h, w = frame.shape[:2]
    return cv2.cvtColor(frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], cv2.COLOR_BGR2GRAY)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=None, help="Blackmagic camera index; auto-detected by default")
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--settle", type=float, default=1.0)
    args = ap.parse_args()
    names = camera_names()
    cam = blackmagic_camera_index(names, requested=args.cam)
    if cam is None:
        raise SystemExit(f"No permitted Blackmagic camera found (macOS devices: {names}).")
    print(f"Blackmagic camera: {names[cam]} (OpenCV #{cam})")
    cap = open_camera(cam)
    os.makedirs("snapshots", exist_ok=True)
    loop = asyncio.get_running_loop()
    print("\nSTEP 1. Switch the lens to MF and manually focus on your face.")
    await loop.run_in_executor(None, input, "Sit at your desk without moving. Press Enter when the face is sharp: ")
    ref_vals = []
    for _ in range(5):
        frame = latest_frame(cap, flush=4)
        face = detect_face(frame)
        if face:
            x, y, w, h = face
            ref_vals.append(tenengrad(cv2.cvtColor(frame[max(y, 0):y + h, max(x, 0):x + w], cv2.COLOR_BGR2GRAY)))
            cv2.imwrite("snapshots/ref_manual.jpg", frame)
    if not ref_vals:
        raise SystemExit("Face not found in reference frame. Repeat, sitting exactly in the frame.")
    ref = sum(ref_vals) / len(ref_vals)
    print(f"Reference (manual focus): face sharpness {ref:.2f}")
    await loop.run_in_executor(None, input, "STEP 2. Return the switch to AF, don't move, Enter: ")
    dev = await find_camera()
    rows = []
    async with BleakClient(dev) as client:
        await set_focus(client, 0.0)
        await asyncio.sleep(2.0)
        print("focus | face | frame center | background left | background upper right (Tenengrad/brightness²)")
        n = round(1.0 / args.step)
        for i in range(n + 1):
            pos = min(1.0, i * args.step)
            await set_focus(client, pos)
            await asyncio.sleep(args.settle)
            frame = latest_frame(cap, flush=6)
            face = detect_face(frame)
            left = tenengrad(region(frame, 0.0, 0.0, 0.25, 0.45))
            right = tenengrad(region(frame, 0.65, 0.0, 1.0, 0.15))
            if face:
                x, y, w, h = face
                fc = tenengrad(cv2.cvtColor(frame[max(y, 0):y + h, max(x, 0):x + w], cv2.COLOR_BGR2GRAY))
            else:
                fc = None
            center = tenengrad(region(frame, 0.35, 0.3, 0.65, 0.7))
            if round(pos * 100) % 10 == 0:
                cv2.imwrite(f"snapshots/sweep_cam{cam}_{pos:.1f}.jpg", frame)
            rows.append({"focus": round(pos, 3), "face": fc, "center": center, "left": left, "right": right})
            print(f"{pos:.2f}  {fc if fc is None else round(fc, 2):>6}  {center:7.2f}  {left:7.2f}  {right:7.2f}")
    cap.release()
    with open("snapshots/sweep.json", "w") as f:
        json.dump(rows, f, indent=1)
    close = [r["focus"] for r in rows if r["face"] is not None and r["face"] >= 0.8 * ref]
    print(f"\nReference {ref:.2f}; max. face sharpness via Bluetooth "
          f"{max((r['face'] for r in rows if r['face'] is not None), default=0):.2f}")
    print(f"Focus positions with face sharpness ≥ 80% reference: {close if close else 'no'}")
    for key in ("face", "center", "left", "right"):
        vals = [r for r in rows if r[key] is not None]
        best = max(vals, key=lambda r: r[key])
        print(f"peak {key}: focus {best['focus']} ({best[key]:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
