"""Procedural texturing, evaluated in 3D on the surface (so it never cares where UV seams are): masks from anatomy
(normal direction, height, groups, curvature, distance to a group, noise) combined with * | ~ and smoothstep;
patterns (stripes / bands along an axis or a curve, cellular spots / warts / cells, fbm via mask_noise) and color
ramps; a layer stack for albedo, roughness and height; bake() to PNG maps (albedo, roughness, tangent-space normal
from the height layers, height) with edge padding."""
import time
import weakref

import numpy as np

from .field import _hash, noise
from .mesh import unit, vec
from .png import write_png

_cache = weakref.WeakKeyDictionary()  # per-mesh derived data (vertex curvature, group distances); meshes are immutable


def _memo(mesh, key, fn):
    d = _cache.setdefault(mesh, {})
    if key not in d:
        d[key] = fn()
    return d[key]


# ---- surface samples --------------------------------------------------------------------
class Surface:
    """Points on a mesh: triangle ids (rows of mesh.triangles) + barycentrics. .points, .normals (interpolated vertex
    normals), .face (source face); interp() blends per-vertex values, so everything is continuous across UV seams."""

    def __init__(self, mesh, tri, bary):
        self.mesh = mesh
        self.tri = np.asarray(tri, dtype=np.int64)
        self.bary = np.asarray(bary, dtype=float).reshape(-1, 3)
        T, src = mesh.triangles
        self.T, self.face = T[self.tri], src[self.tri]
        self.points = self.interp(mesh.verts)
        n = self.interp(mesh.vert_normals)
        L = np.linalg.norm(n, axis=1, keepdims=True)
        self.normals = np.divide(n, L, out=np.zeros_like(n), where=L > 0)

    def __len__(self):
        return len(self.tri)

    def interp(self, vals):
        """Per-vertex values (n,) or (n,k) at the samples."""
        v = np.asarray(vals, dtype=float)[self.T]
        return (v * self.bary[..., None]).sum(1) if v.ndim == 3 else (v * self.bary).sum(1)

    def moved(self, d):
        """Samples moved by 3D offsets d (k,3) within their triangle's plane (barycentrics may leave 0..1)."""
        V = self.mesh.verts
        A, B, C = V[self.T[:, 0]], V[self.T[:, 1]], V[self.T[:, 2]]
        e1, e2 = B - A, C - A
        g11, g12, g22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
        r1, r2 = (d * e1).sum(1), (d * e2).sum(1)
        det = np.maximum(g11 * g22 - g12 * g12, 1e-300)
        b1, b2 = (g22 * r1 - g12 * r2) / det, (g11 * r2 - g12 * r1) / det
        return Surface(self.mesh, self.tri, self.bary + np.stack([-b1 - b2, b1, b2], 1))

    @classmethod
    def at_verts(cls, mesh):
        """One sample per vertex used by a face (on the first triangle using it); unused verts are skipped."""
        T, _ = mesh.triangles
        flat = T.reshape(-1)
        verts, first = np.unique(flat, return_index=True)
        tri, k = first // 3, first % 3
        bary = np.zeros((len(tri), 3))
        bary[np.arange(len(tri)), k] = 1.0
        s = cls(mesh, tri, bary)
        s.verts = verts
        return s

    @classmethod
    def at_faces(cls, mesh):
        """One sample per face, at the center of its first fan triangle."""
        T, src = mesh.triangles
        first = np.flatnonzero(np.r_[True, src[1:] != src[:-1]])
        return cls(mesh, first, np.full((len(first), 3), 1 / 3))


def surface_values(mesh, field, at="verts"):
    """Evaluate a mask / pattern / ramp / texture on the mesh: at='verts' (per vert, (V,...)) or 'faces'. A texture
    gives a dict of channels."""
    S = Surface.at_verts(mesh) if at == "verts" else Surface.at_faces(mesh)
    out = field.evaluate(S) if isinstance(field, Texture) else _value(field, S)
    if at == "verts" and len(S.verts) != len(mesh.verts):
        def full(a):
            x = np.zeros((len(mesh.verts),) + np.shape(a)[1:])
            x[S.verts] = a
            return x
        return {k: full(v) for k, v in out.items()} if isinstance(out, dict) else full(out)
    return out


# ---- fields: masks and patterns ----------------------------------------------------------------------
def _smooth(x, lo, hi):
    if hi == lo:
        return (x >= lo).astype(float)
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _value(o, S):
    if isinstance(o, (Field, ColorField)):
        return o(S)
    return np.full(len(S), float(o))


class Field:
    """A scalar on the surface (masks and patterns, mostly 0..1): f(Surface) -> (k,). Combine: a * b (multiply,
    also by numbers), a | b (max), a & b (min), ~a (1 - a), a.smoothstep(lo, hi) (remap)."""

    def __init__(self, fn, label):
        self.fn, self.label = fn, label

    def __call__(self, S):
        return np.asarray(self.fn(S), dtype=float).reshape(-1)

    def __repr__(self):
        return f"Field({self.label})"

    def __mul__(self, o):
        return Field(lambda S: self(S) * _value(o, S), f"{self.label} * {_label(o)}")

    __rmul__ = __mul__

    def __or__(self, o):
        return Field(lambda S: np.maximum(self(S), _value(o, S)), f"{self.label} | {_label(o)}")

    def __and__(self, o):
        return Field(lambda S: np.minimum(self(S), _value(o, S)), f"{self.label} & {_label(o)}")

    def __invert__(self):
        return Field(lambda S: 1.0 - self(S), f"~{self.label}")

    def smoothstep(self, lo, hi):
        """Remap: 0 below lo, 1 above hi, smooth between (lo > hi reverses)."""
        return Field(lambda S: _smooth(self(S), lo, hi), f"smoothstep({self.label}, {lo:g}, {hi:g})")

    def evaluate(self, S):
        return self(S)


def _label(o):
    return o.label if isinstance(o, (Field, ColorField)) else f"{o:g}"


def smoothstep(field, lo, hi):
    """field.smoothstep(lo, hi) as a function."""
    return field.smoothstep(lo, hi)


def mask_normal(direction="z", lo=0.0, hi=0.6):
    """1 where the surface faces direction (e.g. 'z' dorsal, '-z' ventral): smoothstep(lo, hi, normal . dir)."""
    d = unit(direction)
    return Field(lambda S: _smooth(S.normals @ d, lo, hi), f"normal({direction})")


def mask_height(lo, hi, axis="z"):
    """Ramp along an axis: 0 at lo, 1 at hi (world units; lo > hi ramps the other way)."""
    a = unit(axis)
    return Field(lambda S: _smooth(S.points @ a, lo, hi), f"height({lo:g}..{hi:g})")


def _group_names(mesh, name):
    """Group names from a name, a wildcard pattern ('claw*') or a list of either."""
    import fnmatch
    out = []
    for n in ([name] if isinstance(name, str) else list(name)):
        if any(c in n for c in "*?["):
            hits = sorted(g for g in mesh.groups if fnmatch.fnmatchcase(g, n))
            if not hits:
                raise KeyError(f"no group matches {n!r}; groups: {sorted(mesh.groups)}")
            out += hits
        else:
            out.append(n)
    return out


def _group_faces(mesh, name):
    from .mesh import faces_of
    return _memo(mesh, ("faces", _key(name)),
                 lambda: np.asarray(sorted(faces_of(mesh, _group_names(mesh, name))), dtype=np.int64))


def _key(name):
    return name if isinstance(name, str) else tuple(name)


def _nearest(Q, P):
    from .report import _nearest_dist
    return _nearest_dist(Q, P) if len(P) else np.full(len(Q), np.inf)


def _group_dist(mesh, name):
    """Per vertex: distance to the group's verts (0 on the group), and distance to the rest (0 off it)."""
    def make():
        fs = _group_faces(mesh, name)
        inside = np.zeros(len(mesh.verts), dtype=bool)
        inside[[v for f in fs for v in mesh.faces[f]]] = True
        full = np.zeros(len(mesh.verts), dtype=bool)  # verts whose faces are all in the group
        member = np.zeros(len(mesh.faces), dtype=bool)
        member[fs] = True
        full[:] = inside
        for f in np.flatnonzero(~member):
            full[list(mesh.faces[f])] = False
        P = mesh.verts
        d_in = np.where(inside, 0.0, _nearest(P, P[inside]))
        d_out = np.where(full, _nearest(P, P[~full]), 0.0)
        return d_in, d_out
    return _memo(mesh, ("group", _key(name)), make)


def mask_group(name, blur=0.0):
    """1 on the group's faces, 0 elsewhere. name = a group (e.g. an SDF part), a wildcard ('claw*') or a list; blur =
    width of a soft border centered on the group's edge (world units)."""
    def fn(S):
        if blur <= 0:
            return np.isin(S.face, _group_faces(S.mesh, name)).astype(float)
        d_in, d_out = _group_dist(S.mesh, name)
        return _smooth(S.interp(d_in - d_out), blur / 2, -blur / 2)
    return Field(fn, f"group({name})")


def mask_distance(name, radius, inner=0.0):
    """1 within `inner` of a group (name / wildcard / list; and on it), fading to 0 at `radius` from it (3D distance,
    world units)."""
    return Field(lambda S: _smooth(S.interp(_group_dist(S.mesh, name)[0]), radius, inner),
                 f"distance({name}, {radius:g})")


def _curvature(mesh, smooth):
    """Per vertex: normal . (vert - neighbor mean) / mean edge length, smoothed, scaled so its 95th percentile of |c|
    is 1 (convex > 0)."""
    def make():
        E = np.array(mesh.edges, dtype=np.int64).reshape(-1, 2)
        V, n = mesh.verts, len(mesh.verts)
        deg, acc, el = np.zeros(n), np.zeros((n, 3)), np.zeros(n)
        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        for a, b in ((0, 1), (1, 0)):
            np.add.at(deg, E[:, a], 1)
            np.add.at(acc, E[:, a], V[E[:, b]])
            np.add.at(el, E[:, a], L)
        ok = deg > 0
        c = np.zeros(n)
        c[ok] = ((V[ok] - acc[ok] / deg[ok, None]) * mesh.vert_normals[ok]).sum(1) / (el[ok] / deg[ok])
        for _ in range(int(smooth)):
            s = np.zeros(n)
            for a, b in ((0, 1), (1, 0)):
                np.add.at(s, E[:, a], c[E[:, b]])
            c = np.where(ok, 0.5 * c + 0.5 * s / np.maximum(deg, 1), c)
        scale = np.percentile(np.abs(c[ok]), 95) if ok.any() else 1.0
        return c / max(scale, 1e-12)
    return _memo(mesh, ("curv", int(smooth)), make)


def mask_curvature(kind="convex", lo=0.2, hi=1.0, smooth=2):
    """1 on ridges and bumps (kind='convex') or creases and pits ('concave'): per-vertex curvature (normal offset from
    the neighbor mean, smoothed `smooth` times) scaled so the strongest 5% read >= 1, then smoothstep(lo, hi)."""
    if kind not in ("convex", "concave"):
        raise ValueError("mask_curvature: kind must be 'convex' or 'concave'")
    sign = 1.0 if kind == "convex" else -1.0
    return Field(lambda S: _smooth(sign * S.interp(_curvature(S.mesh, smooth)), lo, hi), f"curvature({kind})")


def mask_noise(seed=0, scale=0.2, lo=-0.3, hi=0.3, octaves=4):
    """fbm noise (scale = feature size, world units) through smoothstep(lo, hi): blotches, variation."""
    f = noise(seed=seed, scale=scale, strength=1.0, octaves=octaves)
    return Field(lambda S: _smooth(f(S.points), lo, hi), f"noise({seed})")


def _curve_param(points, resolution=16):
    """Arc length of the closest point on the Catmull-Rom curve through points, for (k,3) queries."""
    from .sweep import catmull_rom
    C, _ = catmull_rom(np.asarray(points, dtype=float).reshape(-1, 3), resolution)
    A, B = C[:-1], C[1:]
    D = B - A
    L = np.linalg.norm(D, axis=1)
    s0 = np.r_[0, np.cumsum(L)][:-1]

    def fn(P):
        out = np.empty(len(P))
        step = max(1, int(2e6 // max(len(A), 1)))
        for i in range(0, len(P), step):
            Q = P[i:i + step, None, :]
            t = np.clip(((Q - A) * D).sum(-1) / np.maximum(L * L, 1e-300), 0, 1)
            d = np.linalg.norm(Q - (A + t[..., None] * D), axis=-1)
            j = np.argmin(d, axis=1)
            out[i:i + step] = s0[j] + t[np.arange(len(j)), j] * L[j]
        return out
    return fn


def stripes(axis="z", period=0.1, width=0.5, soft=0.15, along=None, offset=0.0, warp=0.0, seed=0):
    """Bands: 1 on stripes `width` (fraction of the period) wide every `period` world units along an axis, or along a
    curve (along = the points of a sweep / sweep_sdf: bands follow its length). soft = edge blur (fraction of the
    period); warp > 0 wobbles the bands with noise (in periods)."""
    coord = _curve_param(along) if along is not None else (lambda P, a=unit(axis): P @ a)
    wob = noise(seed=seed, scale=period * 1.5, strength=1.0, octaves=2) if warp else None

    def fn(S):
        t = coord(S.points) / period + offset
        if wob is not None:
            t = t + warp * wob(S.points)
        f = t % 1.0
        d = np.where(f < width, np.minimum(f, width - f), -np.minimum(f - width, 1.0 - f))
        return _smooth(d, -soft / 2, soft / 2) if soft > 0 else (d >= 0).astype(float)
    return Field(fn, f"stripes({period:g})")


def spots(scale=0.1, size=0.35, soft=0.3, seed=0, jitter=0.8, vary=0.3, shape="flat"):
    """Cellular (Voronoi) scatter: one feature per cube of edge `scale` (world units), jittered; spot radius = size x
    scale (each varied by up to `vary`). shape 'flat' = solid spots with soft edges, 'dome' = rounded bumps (warts, as
    height), 'cells' = the borders between cells (scales, cracks; soft = line width)."""
    if shape not in ("flat", "dome", "cells"):
        raise ValueError("spots: shape must be 'flat', 'dome' or 'cells'")

    def rnd(c, k):
        return _hash(c[..., 0], c[..., 1], c[..., 2], seed * 7 + k).astype(np.float64) / 4294967296.0

    off = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)])

    def fn(S):
        Q = S.points / scale
        base = np.floor(Q).astype(np.int64) + (1 << 20)
        keys, inv = np.unique((base[:, 0] << 42) | (base[:, 1] << 21) | base[:, 2], return_inverse=True)
        inv = inv.reshape(-1)
        cells = np.stack([keys >> 42, (keys >> 21) & ((1 << 21) - 1), keys & ((1 << 21) - 1)], 1) - (1 << 20)
        c = cells[:, None, :] + off  # features of each occupied cell's 27 neighbors, hashed once per cell
        feat = c + 0.5 + jitter * (np.stack([rnd(c, 1), rnd(c, 2), rnd(c, 3)], -1) - 0.5)
        rad = size * (1 - vary * rnd(c, 4))
        f1 = np.full(len(Q), np.inf)
        f2 = np.full(len(Q), np.inf)
        for k in range(27):
            D = Q - feat[inv, k]
            d = np.sqrt(np.einsum("ij,ij->i", D, D))
            if shape != "cells":
                d = d / rad[inv, k]
            f2 = np.where(d < f1, f1, np.minimum(f2, d))
            f1 = np.minimum(f1, d)
        if shape == "cells":
            return 1.0 - _smooth(f2 - f1, 0.0, max(soft, 1e-6))
        if shape == "dome":
            return np.sqrt(np.clip(1 - f1 * f1, 0, 1)) * _smooth(f1, 1.0, 1.0 - max(soft, 1e-6) * 0.5)
        return _smooth(f1, 1.0, 1.0 - max(soft, 1e-6))
    return Field(fn, f"spots({scale:g})")


# ---- color --------------------------------------------------------------------------------------
def color(c):
    """'#rrggbb', (r, g, b) in 0..1, or (r, g, b) in 0..255 -> float RGB array in 0..1."""
    if isinstance(c, str):
        h = c.lstrip("#")
        if len(h) != 6:
            raise ValueError(f"color: expected '#rrggbb', got {c!r}")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
    v = vec(c)
    return v / 255.0 if v.max() > 1.0 else v


class ColorField:
    """An RGB value on the surface: f(Surface) -> (k,3) in 0..1 (see ramp)."""

    def __init__(self, fn, label):
        self.fn, self.label = fn, label

    def __call__(self, S):
        return np.asarray(self.fn(S), dtype=float).reshape(-1, 3)

    def __repr__(self):
        return f"ColorField({self.label})"


def ramp(field, stops):
    """Color ramp driven by a field (mask or pattern): stops = colors spread evenly over 0..1, or [(t, color), ...]."""
    items = list(stops)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2 and np.ndim(items[0][0]) == 0 \
            and not isinstance(items[0][1], (int, float)):
        ts = np.array([float(t) for t, _ in items])
        cs = np.array([color(c) for _, c in items])
    else:
        ts = np.linspace(0, 1, len(items))
        cs = np.array([color(c) for c in items])
    order = np.argsort(ts, kind="stable")
    ts, cs = ts[order], cs[order]

    def fn(S):
        x = _value(field, S)
        return np.stack([np.interp(x, ts, cs[:, k]) for k in range(3)], 1)
    return ColorField(fn, f"ramp({_label(field)})")


# ---- layer stack -------------------------------------------------------------------------------------
def _entry(e):
    """(source, mask, amount) from a layer entry; a bare color or value is one layer everywhere."""
    plain_rgb = len(e) == 3 and all(isinstance(x, (int, float, np.number)) for x in e) if isinstance(e, tuple) else 0
    if isinstance(e, tuple) and len(e) in (2, 3) and not plain_rgb:
        return e[0], e[1], (e[2] if len(e) == 3 else 1.0)
    return e, None, 1.0


def albedo_of(src, S):
    """(k,3) colors 0..1 from a Texture, ramp, field (gray) or plain color at samples S."""
    if isinstance(src, Texture):
        return src.evaluate(S, ("albedo",))["albedo"]
    if isinstance(src, ColorField):
        return src(S)
    if isinstance(src, Field):
        return np.repeat(src(S)[:, None], 3, 1)
    return np.broadcast_to(color(src), (len(S), 3)).copy()


class Texture:
    """Layer stacks for albedo, roughness and height (see layers). evaluate(Surface) -> dict of channels."""

    def __init__(self, albedo, roughness, height):
        self.albedo, self.roughness, self.height = albedo, roughness, height

    def __repr__(self):
        return (f"Texture(albedo {len(self.albedo)} layers, roughness {len(self.roughness)}, "
                f"height {len(self.height)})")

    def _albedo(self, S):
        out = np.zeros((len(S), 3))
        for src, mask, op in self.albedo:
            col = albedo_of(src, S)
            w = op * (np.ones(len(S)) if mask is None else _value(mask, S))
            out += (col - out) * np.clip(w, 0, 1)[:, None]
        return out

    def _scalar(self, stack, S, blend):
        out = np.zeros(len(S))
        for src, mask, op in stack:
            v = _value(src, S)
            w = (np.ones(len(S)) if mask is None else _value(mask, S)) * op
            out = out + (v - out) * np.clip(w, 0, 1) if blend else out + v * w
        return out

    def evaluate(self, S, channels=("albedo", "roughness", "height")):
        out = {}
        if "albedo" in channels:
            out["albedo"] = self._albedo(S)
        if "roughness" in channels:
            out["roughness"] = np.clip(self._scalar(self.roughness, S, True), 0, 1)
        if "height" in channels:
            out["height"] = self._scalar(self.height, S, False)
        return out


def layers(albedo, roughness=0.6, height=None):
    """A texture as layer stacks, painted bottom to top like a paint program.
    albedo = [(color_or_pattern, mask, opacity), ...]: a color ('#rrggbb' or rgb), a ramp(...) or a field (gray);
    mask None = everywhere; opacity 0..1. Each layer blends over the ones below by mask x opacity.
    roughness = a number, or [(value_or_field, mask, opacity), ...] blended the same way.
    height = [(field_or_number, mask, depth), ...]: summed field x mask x depth (world units; negative = dents);
    bake() turns it into the normal map."""
    alb = [_entry(e) for e in (albedo if isinstance(albedo, list) else [albedo])]
    rough = [(float(roughness), None, 1.0)] if np.ndim(roughness) == 0 else [_entry(e) for e in roughness]
    hgt = [] if height is None else [_entry(e) for e in height]
    return Texture(alb, rough, hgt)


# ---- baking ---------------------------------------------------------------------------------------------
BAKE_PRESETS = {"draft": 512, "final": 1024}  # texture size follows the runner like to_mesh's res


def corner_tangents(mesh):
    """(C,4) per-corner tangent (xyz, handedness w) from the UVs: per-triangle dP/du accumulated over corners sharing
    a vert and a UV (smooth inside islands, split at seams), made orthogonal to the vertex normal."""
    from .uv import corner_tris
    C = corner_tris(mesh)
    T, _ = mesh.triangles
    U = np.where(np.isnan(mesh.uvs), 0, mesh.uvs)
    P = mesh.verts[T]
    W = U[C]
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    d1, d2 = W[:, 1] - W[:, 0], W[:, 2] - W[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-20, 1.0 / np.where(det == 0, 1, det), 0.0)
    tan = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    bit = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    area = np.linalg.norm(np.cross(e1, e2), axis=1)
    tn = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-30) * area[:, None]
    bn = bit / np.maximum(np.linalg.norm(bit, axis=1, keepdims=True), 1e-30) * area[:, None]
    key = _wedge_ids(mesh)
    nk = key.max() + 1 if len(key) else 0
    accT, accB = np.zeros((nk, 3)), np.zeros((nk, 3))
    for k in range(3):
        np.add.at(accT, key[C[:, k]], tn)
        np.add.at(accB, key[C[:, k]], bn)
    N = mesh.vert_normals[mesh.corner_verts]
    t = accT[key]
    t = t - N * (t * N).sum(1, keepdims=True)
    L = np.linalg.norm(t, axis=1, keepdims=True)
    fallback = np.cross(N, np.where(np.abs(N[:, 2:3]) < 0.9, [[0, 0, 1]], [[1, 0, 0]]))
    t = np.where(L > 1e-20, t / np.maximum(L, 1e-30), fallback / np.maximum(np.linalg.norm(fallback, axis=1,
                                                                                          keepdims=True), 1e-30))
    w = np.where((np.cross(N, t) * accB[key]).sum(1) < 0, -1.0, 1.0)
    return np.c_[t, w]


def _wedge_ids(mesh):
    """Per corner: an id shared by corners with the same vert and UV (the vertices of an export)."""
    U = np.where(np.isnan(mesh.uvs), 0, mesh.uvs) if mesh.uvs is not None else np.zeros((len(mesh.corner_verts), 2))
    q = np.round(U * 1e7).astype(np.int64)
    _, inv = np.unique(np.stack([mesh.corner_verts, q[:, 0], q[:, 1]], 1), axis=0, return_inverse=True)
    return inv.reshape(-1)


def _dilate(img, filled, steps):
    """Edge padding: texels within `steps` of an island take the nearest covered texel's value (scipy), or grow by
    averaging covered 8-neighbors `steps` times; the rest get the mean."""
    img = img.astype(float)
    try:
        from scipy.ndimage import distance_transform_edt
        dist, (iy, ix) = distance_transform_edt(~filled, return_indices=True)
        near = ~filled & (dist <= steps)
        img[near] = img[iy[near], ix[near]]
        rest = ~filled & ~near
        if rest.any():
            img[rest] = img[filled].mean(0) if filled.any() else 0
        return img
    except ImportError:
        pass
    filled = filled.copy()
    ch = img.shape[2]
    for _ in range(steps):
        if filled.all():
            break
        acc, cnt = np.zeros_like(img), np.zeros(filled.shape)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sy = slice(max(dy, 0), img.shape[0] + min(dy, 0))
                ty = slice(max(-dy, 0), img.shape[0] + min(-dy, 0))
                sx = slice(max(dx, 0), img.shape[1] + min(dx, 0))
                tx = slice(max(-dx, 0), img.shape[1] + min(-dx, 0))
                f = filled[sy, sx]
                acc[ty, tx] += img[sy, sx] * f[..., None]
                cnt[ty, tx] += f
        grow = ~filled & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow][:, None]
        filled |= grow
    if not filled.all():
        img[~filled] = img[filled].mean(0) if filled.any() else 0
    return img.reshape(img.shape[:2] + (ch,))


class Bake:
    """bake() result: albedo (H,W,3) uint8, roughness (H,W) uint8, normal (H,W,3) uint8 (tangent space, OpenGL +Y),
    height (H,W) float world units or None, size, coverage (texel share inside UV islands), paths {map: file},
    seconds."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        maps = " ".join(k for k in ("albedo", "roughness", "normal", "height") if getattr(self, k) is not None)
        return f"Bake({self.size}px {maps} coverage={self.coverage:.0%} {self.seconds:.2f}s)"


def bake(mesh, texture, size=None, out=None, name="texture", pad=16, normal_strength=1.0):
    """Evaluate the texture at every texel's surface point (through the mesh's UVs) and write PNG maps when `out`
    (a directory) is given: <name>_albedo.png, _roughness.png (gray), _normal.png (tangent space from the height
    layers, OpenGL / glTF convention) and _height.png (16-bit, range in Bake.height_range) when there are height
    layers. size: texels per side (default: the runner's preset, draft 512 / final 1024). Islands are padded by
    `pad` texels so seams don't bleed under filtering. Returns a Bake."""
    from .sdf import _defaults
    from .uv import _cover, uv_pixels
    t0 = time.perf_counter()
    if mesh.uvs is None:
        raise ValueError("bake: mesh has no UVs; unwrap(m) first")
    size = BAKE_PRESETS[_defaults["res"]] if size is None else int(size)
    T, src = mesh.triangles
    no_uv = np.zeros(len(mesh.faces), dtype=bool)
    no_uv[np.repeat(np.arange(len(mesh.faces)), np.diff(mesh.corner_start))[np.isnan(mesh.uvs).any(1)]] = True
    tri_ok = np.flatnonzero(~no_uv[src])  # faces without UVs are not baked
    pix, t, bary = _cover(uv_pixels(mesh, size)[tri_ok], size, size)
    t = tri_ok[t]
    first = np.unique(pix, return_index=True)[1]  # one triangle per texel (overlaps: the first)
    pix, t, bary = pix[first], t[first], bary[first]
    S = Surface(mesh, t, bary)
    has_h = bool(texture.height)
    vals = texture.evaluate(S, ("albedo", "roughness"))
    filled = np.zeros((size, size), dtype=bool)
    filled.flat[pix] = True

    def image(v, ch):
        img = np.zeros((size * size, ch))
        img[pix] = v.reshape(len(pix), ch)
        return _dilate(img.reshape(size, size, ch), filled, pad)

    both = image(np.c_[vals["albedo"], vals["roughness"]], 4)  # one padding pass for both
    albedo = np.clip(both[..., :3] * 255 + 0.5, 0, 255).astype(np.uint8)
    rough = np.clip(both[..., 3] * 255 + 0.5, 0, 255).astype(np.uint8)
    normal_img, height_img, hrange = None, None, None
    if has_h:
        from .uv import corner_tris
        tg = corner_tangents(mesh)[corner_tris(mesh)[t]]  # (k,3,4)
        tan = (tg[..., :3] * bary[..., None]).sum(1)
        tan -= S.normals * (tan * S.normals).sum(1, keepdims=True)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-30)
        w = np.where((tg[..., 3] * bary).sum(1) < 0, -1.0, 1.0)
        bit = np.cross(S.normals, tan) * w[:, None]
        UV = mesh.uvs[corner_tris(mesh)[t]]
        uv_area = np.abs((UV[:, 1, 0] - UV[:, 0, 0]) * (UV[:, 2, 1] - UV[:, 0, 1])
                         - (UV[:, 1, 1] - UV[:, 0, 1]) * (UV[:, 2, 0] - UV[:, 0, 0])) / 2
        P = mesh.verts[T[t]]
        a3 = np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1) / 2
        eps = 0.5 / size * np.sqrt(a3 / np.maximum(uv_area, 1e-30))  # half a texel, in world units
        h0 = texture.evaluate(S, ("height",))["height"]

        def h(d):
            return texture.evaluate(S.moved(d * eps[:, None]), ("height",))["height"]
        dt = (h(tan) - h0) / eps  # surface slope along the tangent and bitangent
        db = (h(bit) - h0) / eps
        n = np.stack([-dt * normal_strength, -db * normal_strength, np.ones(len(dt))], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        nh = image(np.c_[n * 0.5 + 0.5, h0], 4)
        normal_img = np.clip(nh[..., :3] * 255 + 0.5, 0, 255).astype(np.uint8)
        hrange = (float(h0.min()), float(h0.max())) if len(h0) else (0.0, 0.0)
        height_img = nh[..., 3]
    else:
        normal_img = np.empty((size, size, 3), dtype=np.uint8)
        normal_img[:] = (128, 128, 255)
    paths = {}
    if out is not None:
        import os
        os.makedirs(out, exist_ok=True)
        base = os.path.join(out, name)
        paths["albedo"] = write_png(base + "_albedo.png", albedo)
        paths["roughness"] = write_png(base + "_roughness.png", rough)
        paths["normal"] = write_png(base + "_normal.png", normal_img)
        if height_img is not None:
            lo, hi = hrange
            q = np.zeros_like(height_img) if hi <= lo else (height_img - lo) / (hi - lo)
            paths["height"] = write_png(base + "_height.png", np.clip(q * 65535 + 0.5, 0, 65535).astype(np.uint16))
    return Bake(albedo=albedo, roughness=rough, normal=normal_img, height=height_img, height_range=hrange,
                size=size, coverage=float(filled.mean()), paths=paths, seconds=time.perf_counter() - t0,
                name=name)
