"""Core data: Mesh (verts, faces, loose edges, typed groups), Sel (typed selection) and Created (what an op made).

A Mesh is a value: ops never mutate it, they return a new one. Edges and adjacency are derived from
faces + loose edges on first use and cached on the (immutable) mesh."""
from collections import namedtuple
from functools import cached_property
from types import MappingProxyType

import numpy as np

Group = namedtuple("Group", "kind idx")  # kind 'v' or 'f'; idx frozenset of indices
_KIND_ALIASES = {"v": "v", "vert": "v", "verts": "v", "f": "f", "face": "f", "faces": "f"}
_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def edge_key(a, b):
    """Canonical (low, high) key for the edge between verts a and b."""
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def face_edges(f):
    """Directed edges (a, b) of face f in winding order."""
    return list(zip(f, f[1:] + f[:1]))


def vec(d):
    """'x', '-z', '+y', a scalar (broadcast to all 3 axes) or any 3-sequence -> float array (not normalized)."""
    if isinstance(d, str):
        s = d.strip().lower()
        sign = -1.0 if s.startswith("-") else 1.0
        s = s.lstrip("+-")
        if s not in _AXES:
            raise ValueError(f"axis must be x, y, z (optionally signed) or a 3-vector, got {d!r}")
        return sign * np.array(_AXES[s])
    v = np.asarray(d, dtype=float)
    if v.ndim == 0:
        return np.full(3, float(v))
    if v.size != 3:
        raise ValueError(f"expected a scalar or 3 values, got {v.size}")
    return v.reshape(3).copy()


def unit(d):
    """Normalized vec(d); raises on a zero vector."""
    v = vec(d)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length direction")
    return v / n


def _dedupe(items):
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


class Sel(tuple):
    """Typed, ordered selection: a tuple of vert/face indices or edge pairs with .kind ('v'|'e'|'f').
    Vertex loops also carry .closed. Combine same-kind Sels with + | & -."""

    def __new__(cls, kind, idx=(), closed=False):
        if kind not in ("v", "e", "f"):
            raise ValueError(f"Sel kind must be 'v', 'e' or 'f', got {kind!r}")
        items = tuple(edge_key(*e) for e in idx) if kind == "e" else tuple(int(i) for i in idx)
        s = super().__new__(cls, items)
        s.kind, s.closed = kind, bool(closed)
        return s

    def __reduce__(self):
        return (Sel, (self.kind, tuple(self), self.closed))

    def _same(self, other):
        if isinstance(other, Sel) and other.kind != self.kind:
            raise ValueError(f"cannot combine Sel kinds {self.kind!r} and {other.kind!r}; "
                             "convert with verts_of/faces_of/edges_of")
        return tuple(Sel(self.kind, other))

    def __add__(self, other):
        return Sel(self.kind, tuple(self) + self._same(other))

    def __or__(self, other):
        return Sel(self.kind, _dedupe(tuple(self) + self._same(other)))

    def __and__(self, other):
        o = set(self._same(other))
        return Sel(self.kind, [i for i in self if i in o])

    def __sub__(self, other):
        o = set(self._same(other))
        return Sel(self.kind, [i for i in self if i not in o])

    def __repr__(self):
        body = ", ".join(map(str, self[:8])) + (f", ... +{len(self) - 8}" if len(self) > 8 else "")
        return f"Sel({self.kind}{', closed' if self.closed else ''}, n={len(self)}: [{body}])"


class Created:
    """What an op made: .verts, .faces, .edges (Sel) plus op-specific extras (e.g. .top, .start, .vmap)."""

    def __init__(self, verts=(), faces=(), edges=(), **extra):
        self.verts, self.faces, self.edges = Sel("v", verts), Sel("f", faces), Sel("e", edges)
        self.__dict__.update(extra)

    def __repr__(self):
        def brief(v):
            return len(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
        parts = [f"{k}={brief(v)}" for k, v in self.__dict__.items()
                 if not (k in ("verts", "faces", "edges") and not v)]
        return "Created(" + ", ".join(parts) + ")"


def _parse_uvs(uvs, F):
    """Per-corner UVs -> read-only (C, 2) float array (NaN = missing) or None. Accepts a flat (C, 2) array in face
    order, or one entry per face: None (no UVs) or that face's (u, v) pairs."""
    if uvs is None:
        return None
    sizes = np.fromiter((len(f) for f in F), dtype=np.int64, count=len(F))
    C = int(sizes.sum())
    if isinstance(uvs, np.ndarray) and uvs.ndim == 2:
        U = np.array(uvs, dtype=float)
    else:
        items = list(uvs)
        if len(items) == len(F) and len(F) and C != len(F):
            parts = []
            for i, (f, u) in enumerate(zip(F, items)):
                a = np.full((len(f), 2), np.nan) if u is None else np.asarray(u, dtype=float).reshape(-1, 2)
                if len(a) != len(f):
                    raise ValueError(f"uvs for face {i} have {len(a)} corners; the face has {len(f)}")
                parts.append(a)
            U = np.concatenate(parts) if parts else np.zeros((0, 2))
        else:
            U = np.asarray(items, dtype=float).reshape(-1, 2)
    if U.shape != (C, 2):
        raise ValueError(f"uvs: need one (u, v) per face corner ({C}), got shape {U.shape}")
    U.flags.writeable = False
    return U


class Mesh:
    """verts (N,3) float, faces (tuples of vert indices), loose_edges (wire edges on no face), groups {name: Group},
    uvs (C,2) per face corner in face order (NaN = missing) or None. Immutable value; every op returns a new Mesh.
    Derived data (edges, adjacency, normals) is cached."""

    def __init__(self, verts=(), faces=(), loose_edges=(), groups=None, uvs=None):
        V = np.array(verts, dtype=float).reshape(-1, 3)
        V.flags.writeable = False
        n = len(V)
        if isinstance(faces, np.ndarray) and faces.ndim == 2 and faces.dtype.kind in "iu":  # fast path
            if len(faces) and faces.shape[1] < 3:
                raise ValueError(f"faces have {faces.shape[1]} verts; need >= 3")
            F = list(map(tuple, faces.tolist()))
        else:
            F = []
            for f in faces:
                f = tuple(int(i) for i in f)
                if len(f) < 3:
                    raise ValueError(f"face {len(F)} has {len(f)} verts; need >= 3")
                F.append(f)
        if F:
            flat = np.fromiter((i for f in F for i in f), dtype=np.int64)
            if flat.min() < 0 or flat.max() >= n:
                raise IndexError(f"face vertex index out of range (mesh has {n} verts)")
        self.verts, self.faces = V, tuple(F)
        loose = set()
        for e in loose_edges:
            k = edge_key(*e)
            if k[0] == k[1]:
                raise ValueError(f"loose edge {tuple(e)} connects a vert to itself")
            if k[0] < 0 or k[1] >= n:
                raise IndexError(f"loose edge {tuple(e)} out of range (mesh has {n} verts)")
            loose.add(k)
        self.loose_edges = frozenset(loose - self.edge_faces.keys()) if loose else frozenset()
        G = {}
        for name, g in (groups or {}).items():
            kind, idx = g
            if kind not in _KIND_ALIASES:
                raise ValueError(f"group {name!r}: kind must be 'v' or 'f', got {kind!r}")
            kind = _KIND_ALIASES[kind]
            idx = frozenset(int(i) for i in idx)
            lim = n if kind == "v" else len(F)
            if idx and (min(idx) < 0 or max(idx) >= lim):
                raise IndexError(f"group {name!r} has an index out of range")
            G[str(name)] = Group(kind, idx)
        self.groups = MappingProxyType(G)
        self.uvs = _parse_uvs(uvs, F)

    def __repr__(self):
        gs = ", ".join(f"{k}:{g.kind}{len(g.idx)}" for k, g in self.groups.items())
        uv = ", uvs" if self.uvs is not None else ""
        return (f"Mesh(V={len(self.verts)}, F={len(self.faces)}, E={len(self.edges)}, loose={len(self.loose_edges)}"
                f"{uv}, groups=[{gs}])")

    # ---- face corners (UV storage order) ------------------------------------------
    @cached_property
    def corner_start(self):
        """(F+1,) offsets: face f's corners (and uvs rows) are corner_start[f]:corner_start[f+1]."""
        return np.concatenate([[0], np.cumsum([len(f) for f in self.faces], dtype=np.int64)]).astype(np.int64)

    @cached_property
    def corner_verts(self):
        """(C,) vert of each face corner, in face order."""
        return np.fromiter((v for f in self.faces for v in f), dtype=np.int64, count=int(self.corner_start[-1]))

    def face_uvs(self, f):
        """(n,2) UVs of face f's corners (None when the mesh has no UVs)."""
        return None if self.uvs is None else self.uvs[self.corner_start[f]:self.corner_start[f + 1]]

    # ---- derived topology -------------------------------------------------
    @cached_property
    def edge_faces(self):
        """{edge_key: (face, ...)} for every face edge, faces in index order."""
        d = {}
        for fi, f in enumerate(self.faces):
            for a, b in face_edges(f):
                if a != b:
                    d.setdefault(edge_key(a, b), []).append(fi)
        return {k: tuple(v) for k, v in d.items()}

    @cached_property
    def edges(self):
        """All edges (face edges + loose), sorted."""
        return tuple(sorted(self.edge_faces.keys() | self.loose_edges))

    @cached_property
    def vert_faces(self):
        """Per vertex: faces using it."""
        vf = [[] for _ in range(len(self.verts))]
        for fi, f in enumerate(self.faces):
            for v in dict.fromkeys(f):
                vf[v].append(fi)
        return tuple(tuple(x) for x in vf)

    @cached_property
    def vert_neighbors(self):
        """Per vertex: sorted neighbor verts (via face or loose edges)."""
        nb = [set() for _ in range(len(self.verts))]
        for a, b in self.edges:
            nb[a].add(b)
            nb[b].add(a)
        return tuple(tuple(sorted(s)) for s in nb)

    # ---- derived geometry -------------------------------------------------
    @cached_property
    def _newell(self):
        N = np.zeros((len(self.faces), 3))
        C = np.zeros((len(self.faces), 3))
        by_size = {}
        for fi, f in enumerate(self.faces):
            by_size.setdefault(len(f), []).append(fi)
        for fis in by_size.values():
            P = self.verts[np.array([self.faces[i] for i in fis])]
            c = P.mean(axis=1)
            Q = P - c[:, None, :]
            N[fis] = np.cross(Q, np.roll(Q, -1, axis=1)).sum(axis=1)
            C[fis] = c
        return N, C

    @cached_property
    def face_normals(self):
        """Unit face normals (Newell); zero for degenerate faces."""
        N = self._newell[0]
        L = np.linalg.norm(N, axis=1)
        return np.divide(N, L[:, None], out=np.zeros_like(N), where=L[:, None] > 0)

    @cached_property
    def face_areas(self):
        return np.linalg.norm(self._newell[0], axis=1) / 2

    @cached_property
    def face_centers(self):
        """Vertex mean of each face."""
        return self._newell[1]

    @cached_property
    def vert_normals(self):
        """Area-weighted vertex normals."""
        acc = np.zeros((len(self.verts), 3))
        N = self._newell[0]
        for fi, f in enumerate(self.faces):
            acc[list(dict.fromkeys(f))] += N[fi]
        L = np.linalg.norm(acc, axis=1)
        return np.divide(acc, L[:, None], out=np.zeros_like(acc), where=L[:, None] > 0)

    @cached_property
    def triangles(self):
        """Fan triangulation: ((T,3) vert index array, (T,) source face of each triangle)."""
        tris, src = [], []
        for fi, f in enumerate(self.faces):
            for i in range(1, len(f) - 1):
                tris.append((f[0], f[i], f[i + 1]))
                src.append(fi)
        return np.array(tris, dtype=np.int64).reshape(-1, 3), np.array(src, dtype=np.int64)

    @cached_property
    def diag(self):
        """Bounding-box diagonal (1.0 for an empty or point mesh); used for relative tolerances."""
        if len(self.verts) == 0:
            return 1.0
        d = float(np.linalg.norm(self.verts.max(0) - self.verts.min(0)))
        return d if d > 0 else 1.0


def signed_volume(verts, faces):
    """Signed volume enclosed by faces (positive when normals point outward)."""
    V = np.asarray(verts, dtype=float)
    by_size = {}
    for f in faces:
        by_size.setdefault(len(f), []).append(f)
    vol = 0.0
    for fs in by_size.values():  # fan triangles per face size, vectorized
        P = V[np.array(fs)]
        vol += float(np.einsum("mj,mkj->", P[:, 0], np.cross(P[:, 1:-1], P[:, 2:])))
    return vol / 6.0


# ---- selection resolution ---------------------------------------------------
def _check_range(mesh, kind, idx):
    if kind == "e":
        for e in idx:
            if e not in mesh.edge_faces and e not in mesh.loose_edges:
                raise IndexError(f"edge {e} is not in the mesh")
        return
    lim = len(mesh.verts) if kind == "v" else len(mesh.faces)
    for i in idx:
        if not 0 <= i < lim:
            raise IndexError(f"{'vert' if kind == 'v' else 'face'} index {i} out of range (0..{lim - 1})")


def _convert(mesh, src, idx, kind):
    if src == kind:
        _check_range(mesh, kind, idx)
        return _dedupe(idx)
    _check_range(mesh, src, idx)
    s = set(idx)
    if src == "f":
        if kind == "v":
            return sorted({v for f in s for v in mesh.faces[f]})
        return sorted({edge_key(a, b) for f in s for a, b in face_edges(mesh.faces[f]) if a != b})
    if src == "e":
        if kind == "v":
            return sorted({v for e in s for v in e})
        cand = sorted({f for e in s for f in mesh.edge_faces.get(e, ())})
        return [f for f in cand if all(edge_key(a, b) in s for a, b in face_edges(mesh.faces[f]) if a != b)]
    if kind == "f":
        return [fi for fi, f in enumerate(mesh.faces) if all(v in s for v in f)]
    return [e for e in mesh.edges if e[0] in s and e[1] in s]


def resolve(mesh, sel, kind):
    """Indices of kind 'v'|'e'|'f' named by sel: None (all), group name(s), Sel (auto-converted), int,
    or a plain list (taken as `kind`; a list of pairs is edges). v->f keeps faces whose verts are all selected."""
    if sel is None:
        return list(range(len(mesh.verts))) if kind == "v" else (
            list(range(len(mesh.faces))) if kind == "f" else list(mesh.edges))
    if isinstance(sel, str):
        if sel not in mesh.groups:
            raise KeyError(f"no group {sel!r}; groups: {sorted(mesh.groups)}")
        g = mesh.groups[sel]
        return _convert(mesh, g.kind, sorted(g.idx), kind)
    if isinstance(sel, Sel):
        return _convert(mesh, sel.kind, list(sel), kind)
    if isinstance(sel, (int, np.integer)):
        return _convert(mesh, kind, [int(sel)], kind)
    items = list(sel)
    if items and all(isinstance(s, str) for s in items):
        return _dedupe(i for s in items for i in resolve(mesh, s, kind))
    if items and not isinstance(items[0], (int, np.integer)):
        return _convert(mesh, "e", [edge_key(*e) for e in items], kind)
    if kind == "e" and items:
        raise ValueError("a plain list of ints is ambiguous for edges; pass (a, b) pairs or a Sel")
    return _convert(mesh, kind, [int(i) for i in items], kind)


def verts_of(mesh, sel):
    """Sel of verts named by sel (faces/edges convert to their verts)."""
    return Sel("v", resolve(mesh, sel, "v"))


def faces_of(mesh, sel):
    """Sel of faces named by sel (verts/edges convert to faces fully inside them)."""
    return Sel("f", resolve(mesh, sel, "f"))


def edges_of(mesh, sel):
    """Sel of edges named by sel."""
    return Sel("e", resolve(mesh, sel, "e"))


def group(mesh, name):
    """The group as a typed Sel (sorted indices)."""
    if name not in mesh.groups:
        raise KeyError(f"no group {name!r}; groups: {sorted(mesh.groups)}")
    g = mesh.groups[name]
    return Sel(g.kind, sorted(g.idx))


# ---- internal working copy for ops ------------------------------------------
class Builder:
    """Mutable working copy used inside ops; build() freezes it into a new Mesh."""

    def __init__(self, mesh):
        self.V = mesh.verts.tolist()
        self.F = list(mesh.faces)
        self.L = set(mesh.loose_edges)
        self.G = {k: [g.kind, set(g.idx)] for k, g in mesh.groups.items()}
        if mesh.uvs is None:
            self.UV = [None] * len(self.F)
        else:  # one tuple of (u, v) per face corner, or None
            cs, U = mesh.corner_start.tolist(), mesh.uvs.tolist()
            self.UV = [tuple(map(tuple, U[cs[i]:cs[i + 1]])) for i in range(len(self.F))]
        self.F0 = mesh.faces  # faces as they were, for UV lookups after an op rewrites them
        self.vsrc = {}  # new vert -> the vert it was made from

    def vert(self, co, src=None):
        """Append a vert; src = vert whose group membership (and UVs, in faces made from src's faces) it inherits."""
        i = len(self.V)
        self.V.append([float(c) for c in co])
        if src is not None:
            self._inherit("v", i, src)
            self.vsrc[i] = int(src)
        return i

    def face(self, vs, src=None, uv=None):
        """Append a face; src = face whose groups it inherits. Corner UVs: uv (one (u, v) per vert) if given, else
        each vert's (or its source vert's) UV in src, NaN where it has none; no UVs without src."""
        i = len(self.F)
        vs = tuple(int(v) for v in vs)
        self.F.append(vs)
        if src is not None:
            self._inherit("f", i, src)
        if uv is not None:
            self.UV.append(tuple((float(a), float(b)) for a, b in uv))
        elif src is not None and self.UV[src] is not None:
            self.UV.append(self.trace_uv(vs, src))
        else:
            self.UV.append(None)
        return i

    def trace_uv(self, vs, src):
        """UVs for verts vs looked up in face src (as it was before the op), following new verts to their sources."""
        F = self.F0[src] if src < len(self.F0) else self.F[src]
        at = dict(zip(F, self.UV[src]))
        out = []
        for v in vs:
            while v not in at and v in self.vsrc:
                v = self.vsrc[v]
            out.append(at.get(v, (np.nan, np.nan)))
        return tuple(out)

    def uvs(self, faces=None):
        """UV argument for Mesh(): None when no face has UVs."""
        UV = self.UV if faces is None else [self.UV[i] for i in faces]
        return UV if any(u is not None for u in UV) else None

    def add_to_group(self, name, kind, idx):
        g = self.G.setdefault(name, [kind, set()])
        if g[0] != kind:
            raise ValueError(f"group {name!r} is a {'vert' if g[0] == 'v' else 'face'} group")
        g[1].update(int(i) for i in idx)

    def _inherit(self, kind, new, src):
        for g in self.G.values():
            if g[0] == kind and src in g[1]:
                g[1].add(new)

    def build(self):
        return Mesh(self.V, self.F, self.L, {k: (g[0], g[1]) for k, g in self.G.items()}, self.uvs())

    def build_compact(self, drop_v=(), drop_f=()):
        """Build without dropped verts/faces, renumbering everything (groups included). Returns (mesh, vmap, fmap)."""
        drop_v, drop_f = set(drop_v), set(drop_f)
        vmap = np.full(len(self.V), -1, dtype=np.int64)
        keep_v = [i for i in range(len(self.V)) if i not in drop_v]
        vmap[keep_v] = np.arange(len(keep_v))
        fmap = np.full(len(self.F), -1, dtype=np.int64)
        keep_f = [i for i in range(len(self.F)) if i not in drop_f]
        fmap[keep_f] = np.arange(len(keep_f))
        F = []
        for i in keep_f:
            f = tuple(int(vmap[v]) for v in self.F[i])
            if min(f) < 0:
                raise RuntimeError("internal: kept face references a dropped vert")
            F.append(f)
        L = {edge_key(vmap[a], vmap[b]) for a, b in self.L if vmap[a] >= 0 and vmap[b] >= 0}
        G = {}
        for k, (kind, idx) in self.G.items():
            m = vmap if kind == "v" else fmap
            G[k] = (kind, {int(m[i]) for i in idx if m[i] >= 0})
        V = np.array(self.V, dtype=float).reshape(-1, 3)[keep_v]
        return Mesh(V, F, L, G, self.uvs(keep_f)), vmap, fmap
