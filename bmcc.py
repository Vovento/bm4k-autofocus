"""Common code: BLE focus control BMPCC 4K and image processing (USB camera, face search, sharpness)."""
import struct

import cv2
import Foundation
import Vision
from bleak import BleakScanner

BM_SERVICE = "291d567a-6d75-11e6-8b77-86f30ca893d3"
OUTGOING = "5dd3465f-1aee-4299-8493-d2eca2f8e1bb"
INCOMING = "b864e140-76a0-416a-bf30-5876504537d9"


# ---- BLE: Camera Control packets ----

def packet(category, parameter, dtype, operation, data=b""):
    body = bytes([category, parameter, dtype, operation]) + data
    msg = bytes([255, len(body), 0, 0]) + body # 255 = all devices
    return msg + b"\x00" * (-len(msg) % 4)


def focus_assign(x):
    x = max(0.0, min(1.0, x))
    return packet(0, 0, 128, 0, struct.pack("<h", round(x * 2048)))


def autofocus():
    return packet(0, 1, 0, 0)


async def find_camera(timeout=15.0):
    dev = await BleakScanner.find_device_by_filter(
        lambda d, adv: BM_SERVICE in [u.lower() for u in adv.service_uuids], timeout=timeout)
    if dev is None:
        raise SystemExit("The camera was not found via BLE. Is Bluetooth enabled on the camera? Is the phone turned off?")
    return dev


async def set_focus(client, x):
    await client.write_gatt_char(OUTGOING, focus_assign(x), response=True)


# ---- Video: camera via USB ----

def _avfoundation_camera_devices(avfoundation):
    """Devices in the same order in which OpenCV AVFoundation numbers them.

    OpenCV combines Video and Muxed, then sorts devices by uniqueID.
    The order that AVFoundation directly returns may vary and
    change when connecting Pocket or Continuity Camera.
    """
    devices = list(avfoundation.AVCaptureDevice.devicesWithMediaType_(avfoundation.AVMediaTypeVideo))
    devices += list(avfoundation.AVCaptureDevice.devicesWithMediaType_(avfoundation.AVMediaTypeMuxed))
    return sorted(devices, key=lambda device: str(device.uniqueID()))


def camera_names():
    """macOS video device names in exact order of OpenCV AVFoundation indexes."""
    import AVFoundation
    return [device.localizedName() for device in _avfoundation_camera_devices(AVFoundation)]


def ensure_camera_access():
    """Request access to the camera from macOS and wait for a response (OpenCV itself does not wait for the dialogue)."""
    import threading

    import AVFoundation
    status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeVideo)
    if status == 3:  # authorized
        return
    if status in (1, 2):  # restricted / denied
        raise SystemExit("Access to the camera is denied: System settings → Privacy and security → Camera → "
                         "enable the application (Terminal or BM4K Autofocus Service) and restart it.")
    print("Requesting access to the camera: click Allow in the macOS dialog...")
    done, granted = threading.Event(), []

    def handler(ok):
        granted.append(ok)
        done.set()

    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVFoundation.AVMediaTypeVideo, handler)
    if not done.wait(120) or not granted[0]:
        raise SystemExit("Access to the camera was not obtained: System settings → Privacy and security → Camera.")


def open_camera(index):
    ensure_camera_access()
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera #{index}. Gave Terminal access to camera "
                         "(System Settings → Privacy → Camera)?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def latest_frame(cap, flush=4):
    """Reset accumulated frames and return fresh."""
    frame = None
    for _ in range(flush):
        ok, f = cap.read()
        if ok:
            frame = f
    return frame


def detect_face(frame):
    """Largest face: (x, y, w, h) in pixels or None. Apple Vision, no external models."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    data = Foundation.NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    handler.performRequests_error_([request], None)
    faces = request.results() or []
    if not faces:
        return None
    h_img, w_img = frame.shape[:2]
    best = max(faces, key=lambda f: f.boundingBox().size.width)
    b = best.boundingBox() # normalized, origin bottom left
    return (int(b.origin.x * w_img), int((1 - b.origin.y - b.size.height) * h_img),
            int(b.size.width * w_img), int(b.size.height * h_img))


def detect_face_landmarks(frame):
    """Largest face: ((x, y, w, h), pupillary distance or None, eye frame or None) in pixels, or None.

    One Apple Vision request returns both the frame and the face points; The distance between the eyes is more stable than the width of the frame.
    """
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    data = Foundation.NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
    handler.performRequests_error_([request], None)
    faces = request.results() or []
    if not faces:
        return None
    h_img, w_img = frame.shape[:2]
    best = max(faces, key=lambda f: f.boundingBox().size.width)
    b = best.boundingBox()
    box = (int(b.origin.x * w_img), int((1 - b.origin.y - b.size.height) * h_img),
           int(b.size.width * w_img), int(b.size.height * h_img))
    iod = eyes = eye_l = eye_r = None
    lm = best.landmarks()
    if lm is not None:
        lp, rp = lm.leftPupil(), lm.rightPupil()
        if lp is not None and rp is not None and lp.pointCount() and rp.pointCount():
            a, c = lp.normalizedPoints()[0], rp.normalizedPoints()[0]
            iod = abs(c.x - a.x) * b.size.width * w_img
        le, re_ = lm.leftEye(), lm.rightEye()
        if le is not None and re_ is not None and le.pointCount() and re_.pointCount():
            pts = [le.normalizedPoints()[i] for i in range(le.pointCount())] + \
                  [re_.normalizedPoints()[i] for i in range(re_.pointCount())]
            xs = [b.origin.x + p.x * b.size.width for p in pts]
            ys = [b.origin.y + p.y * b.size.height for p in pts]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            padx, pady = (x1 - x0) * 0.12, max((y1 - y0) * 0.9, 0.012) # margin around the eyes: eyebrows and eyelids
            eyes = (int((x0 - padx) * w_img), int((1 - (y1 + pady)) * h_img),
                    int((x1 - x0 + 2 * padx) * w_img), int((y1 - y0 + 2 * pady) * h_img))

            def one(region): # frame of one eye (in frame pixels)
                ps = [region.normalizedPoints()[i] for i in range(region.pointCount())]
                px = [b.origin.x + p.x * b.size.width for p in ps]
                py = [b.origin.y + p.y * b.size.height for p in ps]
                ax0, ax1, ay0, ay1 = min(px), max(px), min(py), max(py)
                pdx, pdy = (ax1 - ax0) * 0.3, max((ay1 - ay0) * 0.9, 0.01)
                return (int((ax0 - pdx) * w_img), int((1 - (ay1 + pdy)) * h_img),
                        int((ax1 - ax0 + 2 * pdx) * w_img), int((ay1 - ay0 + 2 * pdy) * h_img))
            eye_l, eye_r = one(le), one(re_) # “left” according to Vision - the one on the left in the frame
    yaw = best.yaw() if hasattr(best, "yaw") else None
    roll = best.roll() if hasattr(best, "roll") else None
    return box, iod, eyes, eye_l, eye_r, (float(yaw) if yaw is not None else None), (float(roll) if roll is not None else None)


def sharpness(frame, face):
    """Sharpness of the center of the face (Laplacian variance), normalized to a fixed size."""
    x, y, w, h = face
    # center of face: eyes-nose, no background around the edges
    x0, x1 = x + int(w * 0.2), x + int(w * 0.8)
    y0, y1 = y + int(h * 0.2), y + int(h * 0.75)
    crop = frame[max(y0, 0):y1, max(x0, 0):x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(cv2.resize(crop, (160, 120)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
