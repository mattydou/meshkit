"""Minimal PNG writer and reader (stdlib zlib only): 8-bit gray / RGB / RGBA and 16-bit gray, no interlacing."""
import struct
import zlib

import numpy as np

_COLOR = {1: 0, 3: 2, 4: 6}  # channels -> PNG color type
_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def png_bytes(img):
    """Encode (H, W) or (H, W, 1|3|4) uint8 (or (H, W) uint16) as PNG bytes."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = a[..., None]
    H, W, C = a.shape
    if C not in _COLOR:
        raise ValueError(f"png: need 1, 3 or 4 channels, got {C}")
    if a.dtype == np.uint16:
        if C != 1:
            raise ValueError("png: 16-bit images must be grayscale")
        depth, rows = 16, np.ascontiguousarray(a.astype(">u2")).view(np.uint8).reshape(H, W * 2 * C)
    elif a.dtype == np.uint8:
        depth, rows = 8, a.reshape(H, W * C)
    else:
        raise ValueError(f"png: need uint8 or uint16 pixels, got {a.dtype}")
    raw = np.hstack([np.zeros((H, 1), dtype=np.uint8), rows.reshape(H, -1)]).tobytes()

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, depth, _COLOR[C], 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def write_png(path, img):
    """Write an image array as PNG (see png_bytes). Returns path."""
    with open(path, "wb") as fh:
        fh.write(png_bytes(img))
    return path


def read_png(src):
    """Decode a PNG (path or bytes) written by any tool: 8/16-bit gray, gray+alpha, RGB, RGBA, non-interlaced.
    Returns (H, W) for gray, else (H, W, C); uint8, or uint16 for 16-bit."""
    data = src if isinstance(src, (bytes, bytearray)) else open(src, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos, idat, head = 8, [], None
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            head = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        pos += 12 + n
    W, H, depth, ctype, _, _, interlace = head
    if ctype not in _CHANNELS or depth not in (8, 16) or interlace:
        raise ValueError(f"png: unsupported format (color type {ctype}, depth {depth}, interlace {interlace})")
    C = _CHANNELS[ctype]
    bpp = C * depth // 8
    stride = W * bpp
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8).reshape(H, 1 + stride)
    out = np.zeros((H, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    for y in range(H):
        ft, line = raw[y, 0], raw[y, 1:].astype(np.int32)
        if ft == 0:
            cur = line
        elif ft == 2:
            cur = (line + prev) & 255
        elif ft in (1, 3, 4):  # left-dependent filters: per byte
            cur = np.zeros(stride, dtype=np.int32)
            for x in range(stride):
                a = cur[x - bpp] if x >= bpp else 0
                b = prev[x]
                if ft == 1:
                    p = a
                elif ft == 3:
                    p = (a + b) // 2
                else:
                    c = prev[x - bpp] if x >= bpp else 0
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    p = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                cur[x] = (line[x] + p) & 255
        else:
            raise ValueError(f"png: bad filter type {ft}")
        out[y] = cur
        prev = cur
    img = out.reshape(H, W, bpp)
    if depth == 16:
        img = img.reshape(H, W, C, 2).astype(np.uint16)
        img = (img[..., 0] << 8) | img[..., 1]
    else:
        img = img.reshape(H, W, C)
    return img[..., 0] if C == 1 else img
