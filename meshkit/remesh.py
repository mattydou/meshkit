"""Remeshing fallbacks for shapes cage_from_sdf can't map (raw SDFs, intersections, scanned or imported meshes):
isotropic_remesh(m, edge_length) evens out a triangle surface in pure Python (split / collapse / flip / relax /
project, Botsch & Kobbelt), and quad_remesh(m, target_faces) hands the surface to an external quad remesher
(QuadriFlow or Instant Meshes binaries, or the QuadriFlow inside Blender) and reads the quads back. Both re-derive
topology, so face groups are carried over by nearest source face and UVs are dropped (unwrap again)."""
import glob
import os
import shutil
import subprocess
import tempfile

import numpy as np

from .mesh import Created, Mesh, edge_key
from .spatial import _closest_on_tris


# ---- nearest source faces ----------------------------------------------------------------------
class _Surface:
    """Closest points on a mesh's triangles: k nearest triangle centers (scipy's cKDTree when installed, else
    all triangles) are tested exactly."""

    def __init__(self, mesh, k=16):
        T, self.src = mesh.triangles
        V = mesh.verts
        self.A, self.B, self.C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        self.k = min(k, len(T))
        try:
            from scipy.spatial import cKDTree
            self.tree = cKDTree((self.A + self.B + self.C) / 3)
        except ImportError:  # pragma: no cover
            self.tree = None

    def closest(self, P):
        """(points, source face per point)."""
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        if self.tree is None:  # pragma: no cover
            from .spatial import closest_point
            n = len(self.A)
            got = closest_point(Mesh(np.vstack([self.A, self.B, self.C]), [(i, i + n, i + 2 * n) for i in range(n)]), P)
            return got.points, self.src[got.faces]
        _, idx = self.tree.query(P, k=self.k)
        idx = idx.reshape(len(P), -1)
        X = _closest_on_tris(P[:, None, :], self.A[idx], self.B[idx], self.C[idx])
        j = np.argmin(((X - P[:, None, :]) ** 2).sum(-1), axis=1)
        r = np.arange(len(P))
        return X[r, j], self.src[idx[r, j]]


def _carry_groups(src_mesh, out_verts, out_faces, surf=None):
    """Groups for a new surface: face groups by each new face's nearest source face, vert groups by each new
    vert's nearest source vert."""
    surf = surf or _Surface(src_mesh)
    V = np.asarray(out_verts, dtype=float)
    groups = {}
    fg = {n: g for n, g in src_mesh.groups.items() if g.kind == "f"}
    if fg and len(out_faces):
        ctr = np.array([V[list(f)].mean(0) for f in out_faces])
        _, near = surf.closest(ctr)
        for n, g in fg.items():
            groups[n] = ("f", np.flatnonzero(np.isin(near, list(g.idx))).tolist())
    vg = {n: g for n, g in src_mesh.groups.items() if g.kind == "v"}
    if vg and len(V):
        try:
            from scipy.spatial import cKDTree
            _, nv = cKDTree(src_mesh.verts).query(V)
        except ImportError:  # pragma: no cover
            nv = np.array([int(np.argmin(np.linalg.norm(src_mesh.verts - p, axis=1))) for p in V])
        for n, g in vg.items():
            groups[n] = ("v", np.flatnonzero(np.isin(nv, list(g.idx))).tolist())
    return groups


# ---- isotropic remeshing ------------------------------------------------------------------------
def _normal(a, b, c):
    return np.cross(b - a, c - a)


def isotropic_remesh(mesh, edge_length, iterations=5, relax=0.5):
    """Even triangles of about edge_length: each iteration splits edges longer than 4/3 L, collapses edges shorter
    than 4/5 L (never creating one over 4/3 L, folding a triangle or pinching the surface), flips edges toward
    valence 6, relaxes verts tangentially (relax = step toward the neighbours' centroid) and projects them back onto
    the input surface. Boundary verts stay on the boundary. A fallback for shapes cage_from_sdf can't map (and for
    cleaning up to_mesh output before quad_remesh); pure Python, seconds for ~20k triangles; faster with scipy.
    Returns (mesh, created) with a triangle mesh: face groups follow the nearest source face, vert groups the
    nearest source vert, UVs are dropped. Created: faces_before, faces_after."""
    L = float(edge_length)
    if L <= 0:
        raise ValueError("isotropic_remesh: edge_length must be > 0")
    hi, lo = 4 / 3 * L, 4 / 5 * L
    surf = _Surface(mesh)
    T0, _ = mesh.triangles
    V = [np.array(p, dtype=float) for p in mesh.verts]
    T = [list(t) for t in T0.tolist()]
    alive = [True] * len(T)
    vt = [set() for _ in V]
    for i, t in enumerate(T):
        for v in t:
            vt[v].add(i)
    ecount = {}
    for t in T:
        for k in range(3):
            key = edge_key(t[k], t[(k + 1) % 3])
            ecount[key] = ecount.get(key, 0) + 1
    border = [False] * len(V)
    for (a, b), c in ecount.items():
        if c == 1:
            border[a] = border[b] = True
    dead = [False] * len(V)

    def edges():
        out = set()
        for i, t in enumerate(T):
            if alive[i]:
                for k in range(3):
                    out.add(edge_key(t[k], t[(k + 1) % 3]))
        return out

    def length(e):
        return float(np.linalg.norm(V[e[0]] - V[e[1]]))

    def nbrs(v):
        return {w for t in vt[v] for w in T[t]} - {v}

    def split(a, b):
        ts = vt[a] & vt[b]
        m = len(V)
        V.append((V[a] + V[b]) / 2)
        vt.append(set())
        border.append(border[a] and border[b] and len(ts) == 1)
        dead.append(False)
        for t in list(ts):
            tri = T[t]
            i = tri.index(a)
            x, y = (a, b) if tri[(i + 1) % 3] == b else (b, a)
            c = next(v for v in tri if v not in (a, b))
            T[t] = [x, m, c]  # (x, y, c) in winding order -> (x, m, c) + (m, y, c)
            T.append([m, y, c])
            alive.append(True)
            t2 = len(T) - 1
            vt[y].discard(t)
            vt[y].add(t2)
            vt[c].add(t2)
            vt[m] |= {t, t2}

    def collapse(a, b):
        if border[a] or border[b]:
            return False
        shared = vt[a] & vt[b]
        if len(shared) != 2:
            return False
        opp = {v for t in shared for v in T[t]} - {a, b}
        if nbrs(a) & nbrs(b) != opp or any(len(nbrs(c)) <= 3 for c in opp):
            return False  # link condition / keep valence >= 3
        x = (V[a] + V[b]) / 2
        if any(np.linalg.norm(x - V[w]) > hi for w in nbrs(a) | nbrs(b)):
            return False
        for t in (vt[a] | vt[b]) - shared:
            old = [V[v] for v in T[t]]
            new = [x if v in (a, b) else V[v] for v in T[t]]
            n0, n1 = _normal(*old), _normal(*new)
            if n0 @ n1 <= 0.2 * np.linalg.norm(n0) * np.linalg.norm(n1) or np.linalg.norm(n1) < 1e-14:
                return False
        V[a] = x
        for t in shared:
            alive[t] = False
            for v in T[t]:
                vt[v].discard(t)
        for t in vt[b]:
            T[t] = [a if v == b else v for v in T[t]]
            vt[a].add(t)
        vt[b], dead[b] = set(), True
        return True

    def flip(a, b, deg):
        ts = list(vt[a] & vt[b])
        if len(ts) != 2:
            return False
        t1, t2 = ts
        tri = T[t1]
        i = tri.index(a)
        if tri[(i + 1) % 3] != b:
            t1, t2 = t2, t1
        c = next(v for v in T[t1] if v not in (a, b))
        d = next(v for v in T[t2] if v not in (a, b))
        if c == d or vt[c] & vt[d]:
            return False
        tgt = {v: 4 if border[v] else 6 for v in (a, b, c, d)}
        before = sum((deg[v] - tgt[v]) ** 2 for v in (a, b, c, d))
        after = sum((deg[v] + dv - tgt[v]) ** 2 for v, dv in ((a, -1), (b, -1), (c, 1), (d, 1)))
        if after >= before or deg[a] <= 3 or deg[b] <= 3:
            return False
        n_old = _normal(V[a], V[b], V[c]) + _normal(V[b], V[a], V[d])
        for tri in ((a, d, c), (d, b, c)):
            n = _normal(*(V[v] for v in tri))
            if n @ n_old <= 0.2 * np.linalg.norm(n) * np.linalg.norm(n_old):
                return False
        T[t1], T[t2] = [a, d, c], [d, b, c]
        vt[b].discard(t1)
        vt[d].add(t1)
        vt[a].discard(t2)
        vt[c].add(t2)
        for v, dv in ((a, -1), (b, -1), (c, 1), (d, 1)):
            deg[v] += dv
        return True

    for _ in range(int(iterations)):
        for e in sorted(edges(), key=length, reverse=True):
            if length(e) > hi and vt[e[0]] & vt[e[1]]:
                split(*e)
        for e in sorted(edges(), key=length):
            if not (dead[e[0]] or dead[e[1]]) and length(e) < lo and vt[e[0]] & vt[e[1]]:
                collapse(*e)
        deg = [len(nbrs(v)) if not dead[v] else 0 for v in range(len(V))]
        for e in list(edges()):
            if vt[e[0]] & vt[e[1]]:
                flip(e[0], e[1], deg)
        P = np.array(V)
        tris = np.array([t for t, ok in zip(T, alive) if ok])
        E = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        S = np.zeros_like(P)
        np.add.at(S, E[:, 0], P[E[:, 1]])
        np.add.at(S, E[:, 1], P[E[:, 0]])
        cnt = np.bincount(E.ravel(), minlength=len(P)).astype(float)
        N = np.zeros_like(P)
        n = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
        for k in range(3):
            np.add.at(N, tris[:, k], n)
        N /= np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
        move = (cnt > 0) & ~np.array(border)
        D = np.zeros_like(P)
        D[move] = S[move] / cnt[move, None] - P[move]
        D -= (D * N).sum(1, keepdims=True) * N
        P = P + relax * D
        P[move] = surf.closest(P[move])[0]
        V = list(P)
    used = sorted({v for t, ok in zip(T, alive) if ok for v in t})
    remap = {v: i for i, v in enumerate(used)}
    faces = [tuple(remap[v] for v in t) for t, ok in zip(T, alive) if ok]
    verts = np.array([V[v] for v in used])
    out = Mesh(verts, faces, groups=_carry_groups(mesh, verts, faces, surf))
    return out, Created(faces=range(len(faces)), faces_before=len(mesh.faces), faces_after=len(faces))


# ---- quad remeshing (external) -----------------------------------------------------------------------
_BLENDER_SCRIPT = '''import sys
import bpy
argv = sys.argv[sys.argv.index("--") + 1:]
src, dst, faces = argv[0], argv[1], int(argv[2])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=src, forward_axis="Y", up_axis="Z")
ob = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = ob
bpy.ops.object.quadriflow_remesh(target_faces=faces, use_mesh_symmetry=False, use_preserve_sharp=False,
                                 use_preserve_boundary=False, smooth_normals=False, mode="FACES", seed=0)
bpy.ops.wm.obj_export(filepath=dst, export_selected_objects=True, export_materials=False, export_uv=False,
                      export_normals=False, forward_axis="Y", up_axis="Z")
'''
BACKENDS = ("quadriflow", "instant-meshes", "blender")


def _blender_paths():
    pats = [r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
            "/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/snap/bin/blender"]
    return sorted((p for pat in pats for p in glob.glob(pat)), reverse=True)


def find_quad_remesher(backend=None):
    """(backend, command) for the first quad remesher found, or None: QuadriFlow ($QUADRIFLOW or `quadriflow` on
    PATH), Instant Meshes ($INSTANT_MESHES, or 'Instant Meshes' / instant-meshes / InstantMeshes on PATH), then
    Blender's built-in QuadriFlow ($BLENDER, `blender` on PATH, or a standard install)."""
    found = {
        "quadriflow": os.environ.get("QUADRIFLOW") or shutil.which("quadriflow"),
        "instant-meshes": os.environ.get("INSTANT_MESHES") or next(
            (p for p in map(shutil.which, ("Instant Meshes", "instant-meshes", "InstantMeshes")) if p), None),
        "blender": os.environ.get("BLENDER") or shutil.which("blender") or next(iter(_blender_paths()), None),
    }
    for name in ([backend] if backend else BACKENDS):
        if name not in found:
            raise ValueError(f"quad_remesh: backend must be one of {', '.join(BACKENDS)}")
        if found[name]:
            return name, [found[name]]
    return None


def quad_remesh(mesh, target_faces, backend=None, command=None, timeout=600):
    """All-quad remesh of a closed surface with about target_faces quads by an external field-aligned remesher:
    QuadriFlow or Instant Meshes binaries, or the QuadriFlow built into Blender (run headless); see
    find_quad_remesher for where each is looked up, or pass backend= and command= (the executable, or a command
    prefix list). A fallback for shapes cage_from_sdf can't map: edge flow follows curvature, not anatomy, and
    poles land where the remesher puts them. Face groups follow the nearest source face; UVs are dropped.
    Returns (mesh, created). Created: faces_before, faces_after, backend. Raises RuntimeError with install hints
    when no remesher is found or it fails."""
    if command is not None:
        name = backend or "quadriflow"
        cmd = list(command) if isinstance(command, (list, tuple)) else [command]
    else:
        got = find_quad_remesher(backend)
        if got is None:
            raise RuntimeError("quad_remesh: no quad remesher found. Install Blender (its QuadriFlow is used "
                               "headless), or put a QuadriFlow / Instant Meshes binary on PATH (or set $QUADRIFLOW, "
                               "$INSTANT_MESHES, $BLENDER). isotropic_remesh() needs nothing extra.")
        name, cmd = got
    from .export import load_obj, to_obj
    T, _ = mesh.triangles
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.obj"), os.path.join(d, "out.obj")
        to_obj(Mesh(mesh.verts, [tuple(t) for t in T.tolist()]), src, y_up=False)
        n = str(int(target_faces))
        if name == "quadriflow":
            args = cmd + ["-i", src, "-o", dst, "-f", n]
        elif name == "instant-meshes":
            args = cmd + [src, "-o", dst, "-f", n, "-d"]
        else:
            script = os.path.join(d, "remesh.py")
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(_BLENDER_SCRIPT)
            args = cmd + ["-b", "--factory-startup", "--python", script, "--", src, dst, n]
        try:
            run = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"quad_remesh: {name} failed to run: {e}") from e
        if run.returncode != 0 or not os.path.exists(dst):
            tail = (run.stderr or run.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError(f"quad_remesh: {name} failed (exit {run.returncode}): {' | '.join(tail)}")
        got = load_obj(dst, y_up=False)
    out = Mesh(got.verts, got.faces, groups=_carry_groups(mesh, got.verts, got.faces))
    return out, Created(faces=range(len(out.faces)), faces_before=len(mesh.faces), faces_after=len(out.faces),
                        backend=name)
