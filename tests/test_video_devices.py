"""Checking the correspondence of camera names with OpenCV AVFoundation indexes.

Run: .venv/bin/python tests/test_video_devices.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bmcc import _avfoundation_camera_devices  # noqa: E402
from video import blackmagic_camera_index, is_blackmagic_camera  # noqa: E402


class Device:
    def __init__(self, name, uid):
        self.name, self.uid = name, uid

    def localizedName(self):
        return self.name

    def uniqueID(self):
        return self.uid


class CaptureDevice:
    by_type = {}

    @classmethod
    def devicesWithMediaType_(cls, media_type):
        return cls.by_type.get(media_type, [])


class FakeAVFoundation:
    AVMediaTypeVideo = "video"
    AVMediaTypeMuxed = "muxed"
    AVCaptureDevice = CaptureDevice


def test_devices_match_opencv_unique_id_order():
    # AVFoundation can return MacBook before Pocket, and OpenCV sorts by uniqueID.
    CaptureDevice.by_type = {
        "video": [Device("MacBook", "6C70"), Device("Pocket Cinema Camera 4K", "0x111")],
        "muxed": [Device("Capture Card", "3A20")],
    }
    devices = _avfoundation_camera_devices(FakeAVFoundation)
    assert [device.localizedName() for device in devices] == [
        "Pocket Cinema Camera 4K", "Capture Card", "MacBook"
    ]


def test_only_blackmagic_cameras_are_allowed():
    names = ["Camera (MacBook Pro)", "Pocket Cinema Camera 6K Pro", "Camera (iPhone)"]
    assert blackmagic_camera_index(names) == 1
    assert blackmagic_camera_index(names, requested=1) == 1
    assert blackmagic_camera_index(names, requested=0) is None
    assert blackmagic_camera_index(["FaceTime HD Camera", "USB Capture Card"]) is None
    assert is_blackmagic_camera("Blackmagic Cinema Camera 6K")
    assert not is_blackmagic_camera("Blackmagic UltraStudio Recorder")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
