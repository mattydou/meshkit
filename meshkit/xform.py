"""Transforms on any vertex subset: sel = None (whole mesh), indices, a group name or a query result.
pivot defaults to the selection's bbox center. Created: moved (vert Sel)."""
import math

import numpy as np

from .mesh import Created, Mesh, Sel, resolve, unit, vec


def _apply(mesh, sel, fn):
    idx = resolve(mesh, sel, "v")
    V = mesh.verts.copy()
    if idx:
        V[idx] = fn(V[idx])
    return Mesh(V, mesh.faces, mesh.loose_edges, mesh.groups, mesh.uvs), Created(moved=Sel("v", idx))


def _pivot(P, pivot):
    if pivot is not None:
        return vec(pivot)
    return (P.min(axis=0) + P.max(axis=0)) / 2 if len(P) else np.zeros(3)


def _cos_sin(deg):
    if float(deg) % 90 == 0:
        return {0: (1.0, 0.0), 1: (0.0, 1.0), 2: (-1.0, 0.0), 3: (0.0, -1.0)}[int(round(deg / 90)) % 4]
    r = math.radians(deg)
    return math.cos(r), math.sin(r)


def rotation_matrix(axis, degrees):
    """3x3 rotation about axis by degrees (right-hand rule); exact for multiples of 90."""
    k = unit(axis)
    c, s = _cos_sin(degrees)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return c * np.eye(3) + s * K + (1 - c) * np.outer(k, k)


def translate(mesh, offset, sel=None):
    """Move verts by offset."""
    o = vec(offset)
    return _apply(mesh, sel, lambda P: P + o)


def rotate(mesh, axis, degrees, sel=None, pivot=None):
    """Rotate verts about axis ('z', vector) through pivot by degrees (right-hand rule)."""
    R = rotation_matrix(axis, degrees)

    def fn(P):
        c = _pivot(P, pivot)
        return (P - c) @ R.T + c
    return _apply(mesh, sel, fn)


def scale(mesh, factor, sel=None, pivot=None):
    """Scale verts by a scalar or (sx, sy, sz) about pivot."""
    f = np.asarray(factor, dtype=float) * np.ones(3)

    def fn(P):
        c = _pivot(P, pivot)
        return (P - c) * f + c
    return _apply(mesh, sel, fn)


def map_verts(mesh, fn, sel=None):
    """Replace selected vert positions with fn(P), P an (k,3) array (vectorized; return (k,3))."""
    def wrapped(P):
        out = np.asarray(fn(P.copy()), dtype=float)
        if out.shape != P.shape:
            raise ValueError(f"map_verts: fn returned shape {out.shape}, expected {P.shape}")
        return out
    return _apply(mesh, sel, wrapped)


def displace(mesh, field, sel=None, along="normal"):
    """Move verts by field values: field = number or f(P) -> (k,). along = 'normal' (area-weighted vertex normals),
    an axis/vector, or None when f(P) returns (k,3) offsets."""
    idx = resolve(mesh, sel, "v")
    P = mesh.verts[idx]
    d = np.asarray(field(P.copy()) if callable(field) else np.full(len(idx), float(field)), dtype=float)
    if along is None:
        off = d.reshape(len(idx), 3)
    else:
        if d.shape != (len(idx),):
            raise ValueError(f"displace: field returned shape {d.shape}, expected ({len(idx)},)")
        dirs = mesh.vert_normals[idx] if isinstance(along, str) and along == "normal" else unit(along)
        off = d[:, None] * dirs
    return _apply(mesh, idx, lambda Q: Q + off)
