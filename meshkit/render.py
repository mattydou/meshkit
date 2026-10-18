"""Optional milestone render, never called automatically: render_views writes one PNG contact sheet, by default
front, side, top and 3/4 orthographic views (same scale), flat shaded; back faces are red, so flipped normals show.
Custom cameras (azimuth, elevation, distance, target) and an unlit textured mode for checking pattern placement.
Pure numpy z-buffer + the stdlib PNG writer; no extra dependencies."""
import math

import numpy as np

from .mesh import Sel, unit, verts_of
from .png import write_png

_FONT = {  # 5x7 bitmap font for panel labels
    "A": "01110 10001 10001 11111 10001 10001 10001", "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110", "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111", "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110", "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001", "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001", "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101", "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110", "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110", "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 10101 01010", "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100", "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11110 00001 00001 01110 00001 00001 11110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    "/": "00001 00010 00010 00100 01000 01000 10000", "-": "00000 00000 00000 11111 00000 00000 00000",
    ".": "00000 00000 00000 00000 00000 01100 01100", ":": "00000 01100 01100 00000 01100 01100 00000",
    "=": "00000 00000 11111 00000 11111 00000 00000", "+": "00000 00100 00100 11111 00100 00100 00000",
    "_": "00000 00000 00000 00000 00000 00000 11111", ",": "00000 00000 00000 00000 01100 00100 01000",
    " ": "00000 00000 00000 00000 00000 00000 00000",
}
_BG, _FRONT, _BACK, _LINE = (46, 48, 54), (205, 205, 200), (205, 70, 60), 0.55
_FOV = 40.0  # perspective cameras (views with a distance)
NAMED_VIEWS = {"front": (0, 0), "back": (180, 0), "side": (90, 0), "right": (90, 0), "left": (-90, 0),
               "top": (0, 90), "bottom": (0, -90), "3/4": (45, math.degrees(math.atan2(0.8, math.sqrt(2))))}


def _default_views():
    """(label, forward, right, up) per panel; Z up, front view looks along +Y."""
    out = []
    for label, f, up in (("FRONT", (0, 1, 0), (0, 0, 1)), ("SIDE", (-1, 0, 0), (0, 0, 1)),
                         ("TOP", (0, 0, -1), (0, 1, 0)), ("3/4", (-1, 1, -0.8), (0, 0, 1))):
        f = unit(f)
        r = unit(np.cross(f, up))
        out.append((label, f, r, np.cross(r, f)))
    return out


def _basis(az, el):
    """(forward, right, up) for a camera at azimuth az (0 = front, looking +Y; 90 = from +X) and elevation el."""
    a, e = math.radians(az), math.radians(el)
    f = -np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
    if abs(f[2]) > 0.999:
        up = np.array([-math.sin(a), math.cos(a), 0.0]) * (1 if el > 0 else -1)
    else:
        up = np.array([0.0, 0.0, 1.0])
    r = unit(np.cross(f, up))
    return f, r, np.cross(r, f)


def _spec(v):
    """Camera spec -> dict(az, el, distance, target, zoom, label). Accepts a named view ('front', '3/4', ...),
    (az, el) / (az, el, distance), or a dict with azimuth/az, elevation/el, distance, target, zoom, label."""
    if isinstance(v, str):
        if v.lower() not in NAMED_VIEWS:
            raise ValueError(f"render_views: unknown view {v!r}; named views: {' '.join(NAMED_VIEWS)}")
        az, el = NAMED_VIEWS[v.lower()]
        return dict(az=az, el=el, distance=None, target=None, zoom=None, label=v.upper())
    if isinstance(v, dict):
        bad = set(v) - {"azimuth", "az", "elevation", "el", "distance", "target", "zoom", "label", "view"}
        if bad:
            raise ValueError(f"render_views: unknown camera keys {sorted(bad)}")
        base = _spec(v["view"]) if "view" in v else dict(az=0.0, el=0.0, label=None)
        az = float(v.get("azimuth", v.get("az", base["az"])))
        el = float(v.get("elevation", v.get("el", base["el"])))
        label = v.get("label") or (base["label"] if "view" in v and "azimuth" not in v and "az" not in v else None)
        return dict(az=az, el=el, distance=v.get("distance"), target=v.get("target"), zoom=v.get("zoom"),
                    label=label)
    t = tuple(v)
    if len(t) not in (2, 3):
        raise ValueError("render_views: a view is a name, (azimuth, elevation[, distance]) or a dict")
    return dict(az=float(t[0]), el=float(t[1]), distance=t[2] if len(t) == 3 else None, target=None, zoom=None,
                label=None)


def _raster(X, Y, Z, tris, W, H):
    """Z-buffer: per pixel the index into tris nearest the camera (-1 = background). Small triangles are rasterized
    together in numpy (ties go to the lower triangle index); the few large ones one by one."""
    zbuf = np.full((H, W), np.inf)
    tid_img = np.full((H, W), -1, dtype=np.int64)
    if not len(tris):
        return tid_img
    xs, ys, zs = X[tris], Y[tris], Z[tris]
    x0 = np.maximum(np.floor(xs.min(1)), 0).astype(np.int64)
    x1 = np.minimum(np.ceil(xs.max(1)), W - 1).astype(np.int64)
    y0 = np.maximum(np.floor(ys.min(1)), 0).astype(np.int64)
    y1 = np.minimum(np.ceil(ys.max(1)), H - 1).astype(np.int64)
    area = (xs[:, 1] - xs[:, 0]) * (ys[:, 2] - ys[:, 0]) - (ys[:, 1] - ys[:, 0]) * (xs[:, 2] - xs[:, 0])
    ok = (x0 <= x1) & (y0 <= y1) & (np.abs(area) >= 1e-12)
    span = np.maximum(x1 - x0, y1 - y0) + 1
    pix, depth, tid = [], [], []
    lo = 0
    for c in (2, 4, 8, 16, 32):  # size classes: each triangle tests a c x c pixel block
        t = np.flatnonzero(ok & (span > lo) & (span <= c))
        lo = c
        if not len(t):
            continue
        oy, ox = np.divmod(np.arange(c * c), c)
        px, py = x0[t, None] + ox, y0[t, None] + oy
        fx, fy = px + 0.5, py + 0.5
        X0, X1, X2 = xs[t, 0, None], xs[t, 1, None], xs[t, 2, None]
        Y0, Y1, Y2 = ys[t, 0, None], ys[t, 1, None], ys[t, 2, None]
        a = area[t, None]
        w0 = ((X2 - X1) * (fy - Y1) - (Y2 - Y1) * (fx - X1)) / a
        w1 = ((X0 - X2) * (fy - Y2) - (Y0 - Y2) * (fx - X2)) / a
        w2 = 1 - w0 - w1
        m = (px <= x1[t, None]) & (py <= y1[t, None]) & (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        z = w0 * zs[t, 0, None] + w1 * zs[t, 1, None] + w2 * zs[t, 2, None]
        pix.append((py * W + px)[m])
        depth.append(z[m])
        tid.append(np.broadcast_to(t[:, None], m.shape)[m])
    if pix:
        pix, depth, tid = np.concatenate(pix), np.concatenate(depth), np.concatenate(tid)
        order = np.lexsort((tid, depth, pix))
        first = order[np.r_[True, pix[order][1:] != pix[order][:-1]]]
        zbuf.flat[pix[first]] = depth[first]
        tid_img.flat[pix[first]] = tid[first]
    for t in np.flatnonzero(ok & (span > 32)):
        i, j, k = tris[t]
        px, py = np.meshgrid(np.arange(x0[t], x1[t] + 1) + 0.5, np.arange(y0[t], y1[t] + 1) + 0.5)
        w0 = ((X[k] - X[j]) * (py - Y[j]) - (Y[k] - Y[j]) * (px - X[j])) / area[t]
        w1 = ((X[i] - X[k]) * (py - Y[k]) - (Y[i] - Y[k]) * (px - X[k])) / area[t]
        w2 = 1 - w0 - w1
        z = w0 * Z[i] + w1 * Z[j] + w2 * Z[k]
        zb = zbuf[y0[t]:y1[t] + 1, x0[t]:x1[t] + 1]
        m = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (z < zb)
        zb[m] = z[m]
        tid_img[y0[t]:y1[t] + 1, x0[t]:x1[t] + 1][m] = t
    return tid_img


def _label(img, text, x, y, scale=2):
    room = (img.shape[1] - x) // (6 * scale)
    for c in str(text).upper()[:max(room, 0)]:
        rows = _FONT.get(c, _FONT[" "]).split()
        for r, bits in enumerate(rows):
            for col, bit in enumerate(bits):
                if bit == "1":
                    img[y + r * scale:y + (r + 1) * scale, x + col * scale:x + (col + 1) * scale] = 255
        x += 6 * scale


def _frame(mesh, focus, zoom):
    """(center, view radius, clip) for the whole mesh, or around a focus group / Sel / point."""
    V = mesh.verts
    lo, hi = V.min(0), V.max(0)
    c, R0 = (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)) / 2, 1e-12)
    if focus is None:
        return c, R0 / (zoom or 1.0), False
    if isinstance(focus, (str, Sel)):
        P = V[sorted(verts_of(mesh, focus))]
        if not len(P):
            raise ValueError(f"render_views: focus {focus!r} has no verts")
    else:
        P = np.asarray(focus, dtype=float).reshape(-1, 3)
    flo, fhi = P.min(0), P.max(0)
    r = float(np.linalg.norm(fhi - flo)) / 2
    radius = R0 / zoom if zoom else (1.25 * r if r > 1e-6 * R0 else R0 / 4)
    return (flo + fhi) / 2, radius, True


def _cameras(mesh, views, focus, zoom):
    """[dict(label, f, r, u, c, radius, dist, clip)]: the four default panels, or one per custom view."""
    if views is None:
        c, radius, clip = _frame(mesh, focus, zoom)
        return [dict(label=lb, f=f, r=r, u=u, c=c, radius=radius, dist=None, clip=clip)
                for lb, f, r, u in _default_views()]
    if isinstance(views, (str, dict)):
        views = [views]
    cams = []
    for v in views:
        s = _spec(v)
        tgt = s["target"] if s["target"] is not None else focus
        c, radius, clip = _frame(mesh, tgt, s["zoom"] if s["zoom"] is not None else zoom)
        f, r, u = _basis(s["az"], s["el"])
        label = s["label"] or f"AZ{s['az']:g} EL{s['el']:g}" + (f" D{float(s['distance']):g}" if s["distance"] else "")
        cams.append(dict(label=label, f=f, r=r, u=u, c=c, radius=radius, clip=clip,
                         dist=None if s["distance"] is None else float(s["distance"])))
    return cams


def _project(V, cam, size):
    """Screen X, Y, depth Z per vert and the camera position (None for orthographic)."""
    Q = V - cam["c"]
    if cam["dist"] is None:
        s = 0.46 * size / cam["radius"]
        return size / 2 + s * (Q @ cam["r"]), size / 2 - s * (Q @ cam["u"]), Q @ cam["f"], None
    eye = cam["c"] - cam["f"] * cam["dist"]
    q = V - eye
    z = q @ cam["f"]
    focal = 0.5 * size / math.tan(math.radians(_FOV) / 2)
    zs = np.where(z > 1e-9, z, 1e-9)
    return size / 2 + focal * (q @ cam["r"]) / zs, size / 2 - focal * (q @ cam["u"]) / zs, z, eye


def _albedo(mesh, textured, tri, bary):
    """(k,3) colors 0..255 for unlit textured panels: a Texture evaluated in 3D, a bake / image sampled through the
    UVs, or (True) a UV checker tinted per island."""
    from .texture import ColorField, Field, Surface, Texture, albedo_of
    T, src = mesh.triangles
    if isinstance(textured, (Texture, ColorField, Field)):  # evaluated in 3D: no UVs needed
        return np.clip(albedo_of(textured, Surface(mesh, tri, bary)), 0, 1) * 255
    if mesh.uvs is None:
        raise ValueError("render_views(textured=...): the mesh has no UVs; unwrap(m) first or pass a Texture")
    from .uv import corner_tris
    C = corner_tris(mesh)[tri]  # (k,3) corner ids
    uv = (mesh.uvs[C] * bary[..., None]).sum(1)
    if textured is True:
        from .uv import uv_islands
        isl = uv_islands(mesh)[src[tri]]
        hue = (isl * 0.618034) % 1.0
        tint = 0.5 + 0.5 * np.stack([np.cos(2 * np.pi * (hue + k / 3)) for k in range(3)], 1)
        check = ((np.floor(uv[:, 0] * 8) + np.floor(uv[:, 1] * 8)) % 2)[:, None]
        out = (0.35 + 0.65 * tint) * (0.55 + 0.45 * check) * 255
        return np.where(np.isnan(uv).any(1)[:, None], (255, 0, 255), out)
    img = getattr(textured, "albedo", textured)
    img = np.asarray(img)
    H, W = img.shape[:2]
    x = np.clip((uv[:, 0] * W).astype(np.int64), 0, W - 1)
    y = np.clip(((1 - uv[:, 1]) * H).astype(np.int64), 0, H - 1)
    col = img[y, x][:, :3].astype(float)
    return np.where(np.isnan(uv).any(1)[:, None], (255, 0, 255), col)


def _panel(mesh, cam, size, edges, textured):
    V, N = mesh.verts, mesh.face_normals
    tris, tri_face = mesh.triangles
    X, Y, Z, eye = _project(V, cam, size)
    TX, TY, TZ = X[tris], Y[tris], Z[tris]
    keep = (TX.max(1) >= 0) & (TX.min(1) <= size) & (TY.max(1) >= 0) & (TY.min(1) <= size)  # on screen
    if eye is not None:
        keep &= TZ.min(1) > 1e-3 * cam["dist"]  # in front of the camera
    if cam["clip"]:
        front = cam["dist"] if eye is not None else 0.0
        keep &= TZ.max(1) >= front - 1.5 * cam["radius"]
    sub = np.flatnonzero(keep)
    draw_edges = len(np.unique(tri_face[sub])) <= 3000 if edges is None else edges
    tid = _raster(X, Y, Z, tris[sub], size, size)
    hit = tid >= 0
    t_img = np.where(hit, sub[np.maximum(tid, 0)], -1)
    fid = np.where(hit, tri_face[np.maximum(t_img, 0)], -1)
    f, r, u = cam["f"], cam["r"], cam["u"]
    facing = N @ -f if eye is None else ((eye - mesh.face_centers) * N).sum(1)
    if textured is None or textured is False:
        light = unit(-0.35 * r + 0.55 * u - 0.75 * f)
        lit = 0.42 + 0.58 * np.clip(np.where(facing >= 0, 1, -1)[:, None] * N @ light, 0, 1)
        colors = np.where((facing >= 0)[:, None], _FRONT, _BACK) * lit[:, None]
        img = np.where(hit[..., None], colors[np.maximum(fid, 0)], _BG)
    else:
        img = np.empty((size, size, 3))
        img[:] = _BG
        py, px = np.nonzero(hit)
        t = t_img[py, px]
        i, j, k = tris[t, 0], tris[t, 1], tris[t, 2]
        fx, fy = px + 0.5, py + 0.5
        a = (X[j] - X[i]) * (Y[k] - Y[i]) - (Y[j] - Y[i]) * (X[k] - X[i])
        w0 = ((X[k] - X[j]) * (fy - Y[j]) - (Y[k] - Y[j]) * (fx - X[j])) / a
        w1 = ((X[i] - X[k]) * (fy - Y[k]) - (Y[i] - Y[k]) * (fx - X[k])) / a
        bary = np.clip(np.stack([w0, w1, 1 - w0 - w1], 1), 0, 1)
        if eye is not None:  # perspective-correct interpolation
            bary = bary / np.stack([Z[i], Z[j], Z[k]], 1)
        bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
        col = _albedo(mesh, textured, t, bary)
        back = facing[tri_face[t]] < 0
        col[back] = 0.5 * col[back] + 0.5 * np.array(_BACK)
        img[py, px] = col
    line = np.zeros(fid.shape, dtype=bool)
    dx, dy = fid[:, 1:] != fid[:, :-1], fid[1:, :] != fid[:-1, :]
    if draw_edges and (textured is None or textured is False):
        line[:, 1:] |= dx
        line[1:, :] |= dy
    else:  # silhouette only
        line[:, 1:] |= dx & ((fid[:, 1:] < 0) | (fid[:, :-1] < 0))
        line[1:, :] |= dy & ((fid[1:, :] < 0) | (fid[:-1, :] < 0))
    img[line & hit] *= _LINE
    panel = np.clip(img, 0, 255).astype(np.uint8)
    _label(panel, cam["label"], 8, 8)
    return panel


def render_views(mesh, path, size=400, edges=None, focus=None, zoom=None, views=None, textured=None):
    """Write a PNG contact sheet of size x size panels. Default: 2x2 FRONT (looking +Y), SIDE (from +X), TOP, 3/4,
    orthographic at one scale, flat shaded, back faces red, face outlines when edges=True (default: only if <= 3000
    faces are drawn). focus = group name, Sel or point(s) to center on; zoom = magnification vs the whole-mesh view
    (default: fit the focused group, 4x for a point); geometry more than 1.5 view radii in front of a focus is cut
    away so it cannot hide it.
    views = [camera, ...] replaces the default panels: a name ('front' 'back' 'left' 'right' 'top' 'bottom' '3/4'),
    (azimuth, elevation[, distance]) in degrees (azimuth 0 = front, 90 = from +X; elevation 90 = top), or a dict
    with azimuth, elevation, distance (perspective camera that far from the target; omitted = orthographic fit),
    target (group / Sel / point; default focus or whole mesh), zoom, label.
    textured = a Texture (layers(...)), ramp or mask (evaluated in 3D), a Bake or (H, W, 3) image (sampled through
    the UVs), or True for a UV checker tinted per island: unlit flat albedo, only for checking pattern placement.
    Returns path."""
    if not len(mesh.faces):
        raise ValueError("render_views: mesh has no faces")
    cams = _cameras(mesh, views, focus, zoom)
    n = len(cams)
    cols = n if n <= 3 and views is not None else (2 if n <= 4 else 3)
    rows = -(-n // cols)
    sheet = np.full((rows * size, cols * size, 3), 90, dtype=np.uint8)
    for p, cam in enumerate(cams):
        y0, x0 = (p // cols) * size, (p % cols) * size
        sheet[y0:y0 + size, x0:x0 + size] = _panel(mesh, cam, size, edges, textured)
    for k in range(1, rows):
        sheet[k * size - 1:k * size + 1, :] = 90
    for k in range(1, cols):
        sheet[:, k * size - 1:k * size + 1] = 90
    return write_png(path, sheet)
