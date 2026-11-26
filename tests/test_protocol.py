"""Проверка сборки и разбора пакетов по примерам из PDF (раздел Example Protocol Packets).

Запуск: .venv/bin/python tests/test_protocol.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from protocol import OFFSET, decode, packet  # noqa: E402


def test_spec_examples():
    # trigger instantaneous auto focus on camera 4
    assert packet(0, 1, 0, dest=4) == bytes([4, 4, 0, 0, 0, 1, 0, 0])
    # turn on OIS on all cameras
    assert packet(0, 6, 0, [True]) == bytes([255, 5, 0, 0, 0, 6, 0, 0, 1, 0, 0, 0])
    # set exposure to 10 ms on camera 4
    assert packet(1, 5, 3, [10000], dest=4) == bytes([4, 8, 0, 0, 1, 5, 3, 0, 0x10, 0x27, 0, 0])
    # add 15% to zebra level
    assert packet(4, 2, 128, [0.15], op=OFFSET, dest=4) == bytes([4, 6, 0, 0, 4, 2, 128, 1, 0x33, 0x01, 0, 0])
    # subtract 0.3 from gamma adjust for green and blue
    assert packet(8, 1, 128, [0, 0, -0.3, -0.3], op=OFFSET, dest=4) == bytes(
        [4, 12, 0, 0, 8, 1, 128, 1, 0, 0, 0, 0, 0x9a, 0xfd, 0x9a, 0xfd])


def test_roundtrip():
    msgs = decode(packet(1, 2, 2, [5600, -3]) + packet(0, 3, 128, [0.5]) + packet(0, 1, 0))
    assert msgs[0] == (1, 2, 2, 0, [5600, -3])
    assert msgs[1] == (0, 3, 128, 0, [0.5])
    assert msgs[2][:2] == (0, 1)


def test_camera_message_with_padding():
    # реальное сообщение камеры: фокусное расстояние 12 мм, int16
    assert decode(bytes([0, 6, 0, 0, 0, 7, 2, 0, 0x0c, 0x00, 0, 0]))[0] == (0, 7, 2, 0, [12])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ок:", name)
