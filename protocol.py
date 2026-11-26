"""Протокол Blackmagic Camera Control: сборка команд и разбор сообщений камеры (см. Документация/)."""
import struct

SIZES = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 128: 2}
FORMATS = {1: "b", 2: "h", 3: "i", 4: "q"}

ASSIGN, OFFSET = 0, 1


def _encode(dtype, values):
    if dtype == 5:
        return str(values).encode("utf-8")
    if dtype == 128:  # fixed 5.11: значение × 2048
        return b"".join(struct.pack("<h", round(max(-16.0, min(15.9995, float(v))) * 2048)) for v in values)
    if dtype == 0:
        return bytes(1 if v else 0 for v in values)
    return b"".join(struct.pack("<" + FORMATS[dtype], int(v)) for v in values)


def packet(category, parameter, dtype, values=(), op=ASSIGN, dest=255):
    """Сообщение «change configuration»: dest, длина, команда 0, резерв + категория, параметр, тип, операция + данные."""
    body = bytes([category, parameter, dtype, op]) + _encode(dtype, values)
    msg = bytes([dest, len(body), 0, 0]) + body
    return msg + b"\x00" * (-len(msg) % 4)


def _decode(dtype, raw):
    if dtype == 5:
        return raw.decode("utf-8", "replace")
    size = SIZES.get(dtype)
    if not size:
        return list(raw)
    n = len(raw) // size
    if dtype == 0:
        return [bool(b) for b in raw]
    if dtype == 128:
        return [v / 2048 for v in struct.unpack(f"<{n}h", raw[:n * 2])]
    return list(struct.unpack("<" + str(n) + FORMATS[dtype], raw[:n * size]))


def decode(data):
    """Разобрать пакет с одним и более сообщений: [(категория, параметр, тип, операция, значения)]."""
    out, i = [], 0
    while i + 4 <= len(data):
        length = data[i + 1]
        body = bytes(data[i + 4:i + 4 + length])
        command = data[i + 2]
        i += 4 + length + (-length % 4)
        if command != 0 or len(body) < 4:
            continue
        category, parameter, dtype, op = body[:4]
        out.append((category, parameter, dtype, op, _decode(dtype, body[4:])))
    return out


def key(category, parameter):
    return f"{category}.{parameter}"
