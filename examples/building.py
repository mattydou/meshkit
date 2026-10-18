"""Architecture, authored vertex -> edge -> face: a footprint loop is grid-filled, extruded into walls, the roof is
inset and recessed, front windows are inset and pushed in; an L-shaped annex starts from a hand-placed n-gon."""
from meshkit import *

BAYS_X, BAYS_Y, BAY, STOREY = 4, 2, 1.0, 3.0

# footprint perimeter, counter-clockwise seen from above (so the fill faces +z)
pts = ([(i * BAY, 0, 0) for i in range(BAYS_X)] + [(BAYS_X * BAY, j * BAY, 0) for j in range(BAYS_Y)] +
       [((BAYS_X - i) * BAY, BAYS_Y * BAY, 0) for i in range(BAYS_X)] + [(0, (BAYS_Y - j) * BAY, 0) for j in range(BAYS_Y)])
m, v = add_verts(Mesh(), pts)
m, e = add_edges(m, v.verts, closed=True)  # wire loop
m, floor = grid_fill(m, e.loop)            # 4x2 quads; corners found from the loop's turns
m, walls = extrude(m, floor.faces, STOREY)  # floor faces become the roof, side walls appear
m, _ = add_group(m, "walls", walls.faces)
m, base = grid_fill(m, boundary_loops(m)[0])  # close the bottom; winding follows the walls (faces -z)

m, roof = inset(m, walls.top, 0.15)          # parapet ring
m, _ = extrude(m, roof.inner, -0.3)          # recessed roof
m, _ = add_group(m, "roof", roof.inner)

front = faces_of(m, "walls") & select_by_normal(m, "-y", 5)
m, win = inset(m, front, 0.25, mode="individual")
m, _ = extrude(m, win.inner, -0.08, mode="individual")
m, _ = add_group(m, "windows", win.inner)

# annex: an L-shaped n-gon from hand-placed verts, extruded and closed with fill()
x0 = BAYS_X * BAY + 1.0
m, a = add_verts(m, [(x0, 0, 0), (x0 + 2, 0, 0), (x0 + 2, 2.5, 0), (x0 + 1, 2.5, 0), (x0 + 1, 1, 0), (x0, 1, 0)])
m, slab = add_face(m, a.verts)
m, annex = extrude(m, slab.faces, 2.0)
m, _ = fill(m, next(l for l in boundary_loops(m) if set(l) <= set(a.verts)))
m, _ = add_group(m, "annex", verts_of(m, a.verts) | annex.verts)
mesh = m
check_opts = {"closed": True}


@register_check
def windows_recessed(mesh):
    ys = mesh.verts[list(verts_of(mesh, "windows"))][:, 1]
    return [] if abs(ys.max() - 0.08) < 1e-9 else [f"window faces at y={ys.max():.3f}, expected 0.08"]


register_scorer(lambda mesh: ("roof_height", float(bbox(mesh, "roof")[1][2])))
