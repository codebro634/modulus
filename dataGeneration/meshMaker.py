import numpy as np
import pygmsh
import meshio
from copy import deepcopy
import math
import os
import json

from pygmsh.geo.geometry import Circle

class gObject:

    def __init__(self, shape: str = 'tri', args:dict = {'x0': [0.33, 0.2], 'x1': [0.38, 0], 'x2': [0.33, 0.25]}):
        self.shape = shape
        self.args = args

    def boundary_string(self):
        tol, dp = 0.05, 5
        if self.shape == 'ellipse':
            return f"on_boundary && x[0]>{round(self.args['x0'][0] - tol - self.args['w'],dp)} && " \
                   f"x[0]< {round(self.args['x0'][0] + tol + self.args['w'],dp)} && " \
                   f"x[1]>{round(self.args['x0'][1] - tol - self.args['h'],dp)} && " \
                   f"x[1]< {round(self.args['x0'][1] + tol + self.args['h'],dp)}"
        elif self.shape == 'rect':
            return f"on_boundary && x[0]>{round(self.args['x0'][0] - tol - self.args['w']/2,dp)} && " \
                   f"x[0]< {round(self.args['x0'][0] + tol + self.args['w']/2,dp)} && " \
                   f"x[1]>{round(self.args['x0'][1] - tol - self.args['h']/2,dp)} &&" \
                   f" x[1]< {round(self.args['x0'][1] + tol + self.args['h']/2,dp)}"
        elif self.shape == 'tri':
            return f"on_boundary && x[0]>{round(min(self.args['x0'][0], self.args['x1'][0], self.args['x2'][0]) - tol,dp)} && " \
                   f"x[0]< {round(max(self.args['x0'][0], self.args['x1'][0], self.args['x2'][0]) + tol,dp)} && " \
                   f"x[1]>{round(min(self.args['x0'][1], self.args['x1'][1], self.args['x2'][1]) - tol,dp)} && " \
                   f"x[1]< {round(max(self.args['x0'][1], self.args['x1'][1], self.args['x2'][1]) + tol,dp)}"
        else:
            raise Exception(f'Shape {self.shape} not supported for boundary string')


def move_object(obj: gObject, dx:float = 0.0, dy:float = 0.0) -> gObject:
    obj = deepcopy(obj)
    if obj.shape == 'ellipse':
        obj.args['x0'][0] += dx
        obj.args['x0'][1] += dy
    elif obj.shape == 'rect':
        obj.args['x0'][0] += dx
        obj.args['x0'][1] += dy
    elif obj.shape == 'tri':
        obj.args['x0'][0] += dx
        obj.args['x0'][1] += dy
        obj.args['x1'][0] += dx
        obj.args['x1'][1] += dy
        obj.args['x2'][0] += dx
        obj.args['x2'][1] += dy
    else:
        raise Exception(f'Shape {obj.shape} not supported for squishing')
    return obj

def create_ellipse(mid, w,h) -> gObject:
    return gObject(shape='ellipse', args={'x0': mid, 'w': w, 'h': h })

def create_rect(mid, w, h) -> gObject:
    return gObject(shape='rect', args={'x0': mid, 'w': w, 'h': h})

def create_equi_tri(mid, r, angle) -> gObject:
    a = math.radians(60)
    corners = [mid[0] , mid[1] - r], [mid[0] + r*math.sin(a), mid[1] + r*math.cos(a)], [mid[0] - r*math.sin(a), mid[1] + r*math.cos(a)]
    def rotate(x, y, angle):
        angle = math.radians(angle)
        return [(x[0]-y[0])*math.cos(angle) - (x[1]-y[1])*math.sin(angle) + y[0], (x[0]-y[0])*math.sin(angle) + (x[1]-y[1])*math.cos(angle) + y[1]]
    corners = [rotate(x, mid, angle) for x in corners]
    return gObject(shape='tri', args={'x0': corners[0], 'x1': corners[1], 'x2': corners[2]})

def squish_object(obj: gObject, xstretch: float = 1.0, ystretch: float = 1.0) -> gObject:
    obj = deepcopy(obj)
    if obj.shape == 'ellipse':
        obj.args['w'] *= xstretch
        obj.args['h'] *= ystretch
    elif obj.shape == 'rect':
        obj.args['h'] *= ystretch
        obj.args['w'] *= xstretch
    elif obj.shape == 'tri':
        mid = [(obj.args['x0'][0] + obj.args['x1'][0] + obj.args['x2'][0])/3, (obj.args['x0'][1] + obj.args['x1'][1] + obj.args['x2'][1])/3]
        obj.args['x0'][0] = mid[0] + (obj.args['x0'][0] - mid[0])*xstretch
        obj.args['x0'][1] = mid[1] + (obj.args['x0'][1] - mid[1])*ystretch
        obj.args['x1'][0] = mid[0] + (obj.args['x1'][0] - mid[0])*xstretch
        obj.args['x1'][1] = mid[1] + (obj.args['x1'][1] - mid[1])*ystretch
        obj.args['x2'][0] = mid[0] + (obj.args['x2'][0] - mid[0])*xstretch
        obj.args['x2'][1] = mid[1] + (obj.args['x2'][1] - mid[1])*ystretch
    else:
        raise Exception(f'Shape {obj.shape} not supported for squishing.')
    return obj


def add_ellipse(
    geo,
    x0: list[float],
    w: float,
    h: float,
    mesh_size: float | None = None,
    R=None,
    compound=False,
    num_sections: int = 3,
    holes=None,
    make_surface: bool = True,
):
    """Add ellipse in the :math:`x`-:math:`y`-plane."""
    if holes is None:
        holes = []
    else:
        assert make_surface

    # Define points that make the ellipse (midpoint and the four cardinal
    # directions).
    X = np.zeros((num_sections + 1, len(x0)))
    if num_sections == 4:
        # For accuracy, the points are provided explicitly.
        X[1:, [0, 1]] = np.array(
            [[w, 0.0], [0.0, h], [-w, 0.0], [0.0, -h]]
        )
    else:
        X[1:, [0, 1]] = np.array(
            [
                [
                    w * np.cos(2 * np.pi * k / num_sections),
                    h * np.sin(2 * np.pi * k / num_sections),
                ]
                for k in range(num_sections)
            ]
        )

    if R is not None:
        assert np.allclose(
            abs(np.linalg.eigvals(R)), np.ones(X.shape[1])
        ), "The transformation matrix doesn't preserve ellipses; at least one eigenvalue lies off the unit ellipse."
        X = np.dot(X, R.T)

    X += x0

    # Add Gmsh Points.
    p = [geo.add_point(x, mesh_size=mesh_size) for x in X]
    point_on_major_axis = geo.add_point(np.array([x0[0] - w/2, x0[1]]), mesh_size=mesh_size)

    arcs = [
        geo.add_ellipse_arc(p[k], p[0], point_on_major_axis, p[k + 1]) for k in range(1, len(p) - 1)
    ] + [geo.add_ellipse_arc(p[-1], p[0], point_on_major_axis, p[1])]

    if compound:
        geo._COMPOUND_ENTITIES.append((1, [arc._id for arc in arcs]))

    curve_loop = geo.add_curve_loop(arcs)

    if make_surface:
        plane_surface = geo.add_plane_surface(curve_loop, holes)
        if compound:
            geo._COMPOUND_ENTITIES.append((2, [plane_surface._id]))
    else:
        plane_surface = None

    return Circle(
        x0,
        (w+h)/2,
        R,
        compound,
        num_sections,
        holes,
        curve_loop,
        plane_surface,
        mesh_size=mesh_size,
    )

def create_mesh(height:float = 0.41, width:float= 1.6, objects: list[object] = [gObject()], mesh_size = 0.025):
    with pygmsh.geo.Geometry() as geom:
        geo_objects = []
        for obj in objects:
            if obj.shape == 'ellipse': #TODO random middle point removen
                geo_objects.append(add_ellipse(
                    geo=geom,
                    x0=obj.args['x0'],
                    w=obj.args['w'],
                    h=obj.args['h'],
                    mesh_size=mesh_size/4,
                    num_sections=32,
                    make_surface=False,
                ))
            elif obj.shape == 'rect':
                geo_objects.append(geom.add_rectangle(
                    xmin=obj.args['x0'][0] - obj.args['w']/2,
                    xmax=obj.args['x0'][0] + obj.args['w']/2,
                    ymin=obj.args['x0'][1] - obj.args['h']/2,
                    ymax=obj.args['x0'][1] + obj.args['h']/2,
                    z=0,
                    mesh_size=mesh_size/4,
                    make_surface=False,
                ))
            elif obj.shape == 'tri':
                geo_objects.append(geom.add_polygon(
                    points=[obj.args['x0'] + [0], obj.args['x1'] + [0], obj.args['x2'] + [0]],
                    mesh_size=mesh_size/4,
                    make_surface=False,
                ))
            else:
                raise Exception(f'Shape {obj.shape} not supported.')

        geom.add_rectangle(
            0.0, width, 0, height, 0.0, mesh_size=mesh_size, holes=[obj.curve_loop for obj in geo_objects]
        )

        mesh = geom.generate_mesh()

    # remove z-coordinate
    mesh = meshio.Mesh(mesh.points[:, :2], {"triangle": mesh.get_cells_type("triangle")})


    return mesh, {'nodes': len(mesh.points), 'object_boundaries': [obj.boundary_string() for obj in objects], 'height': height, 'width': width}

def save_mesh(mesh: meshio.Mesh, metadata: dict, mesh_name: str, folder: str):
    #Create folder if it does not exist
    path = os.path.join(folder, mesh_name)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    path_mesh = os.path.join(path, 'mesh.xdmf')
    path_img = os.path.join(path, 'img.svg')
    path_metadata = os.path.join(path, 'metadata.json')
    mesh.write(path_mesh)
    meshio.svg.write(path_img, mesh, float_fmt=".3f", stroke_width="0.1")
    with open(path_metadata, 'w') as f:
        json.dump(metadata, f)

#tri = create_equi_tri([0.33, 0.2], 0.02, -90)
#tri = squish_object(tri, 1.0, 1.0)
#rect = create_rect([0.5, 0.2], 0.1, 0.1)
#circ = create_ellipse([0.33, 0.2], 0.05,0.05)
#mesh, metadata = create_mesh(objects=[circ])
#save_mesh(mesh, metadata, 'standard', 'meshes')