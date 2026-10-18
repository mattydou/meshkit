import numpy as np
import pytest

from meshkit import Mesh
from meshkit.mesh import edge_key


def hand_grid(nx=3, ny=3, size=1.0):
    """nx x ny quads on z=0 built directly (no primitives), normals +z."""
    V = [(i * size, j * size, 0.0) for j in range(ny + 1) for i in range(nx + 1)]
    F = [(j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i)
         for j in range(ny) for i in range(nx)]
    return Mesh(V, F)


def bad_winding_edges(m):
    """Manifold edges whose two faces traverse them in the same direction."""
    seen, bad = {}, []
    for f in m.faces:
        for a, b in zip(f, f[1:] + f[:1]):
            k = edge_key(a, b)
            if k in seen and seen[k] == (a, b):
                bad.append(k)
            seen[k] = (a, b)
    return bad


def boundary_edges(m):
    return [e for e, fs in m.edge_faces.items() if len(fs) == 1]


@pytest.fixture
def grid3():
    return hand_grid(3, 3)


def approx(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)
