import numpy as np
import pygmsh
import meshio
from copy import deepcopy
import math

#TODO: Add ellipse support, i.e. be able to stretch/squeeze the circle

class gObject:

    def __init__(self, shape:str = 'tri', args:dict = {'x0': [0.2, 0.2], 'x1': [0.25, 0], 'x2': [0.2, 0.25]}):
        self.shape = shape
        self.args = args


def move_object(obj: gObject, dx:float = 0.0, dy:float = 0.0) -> gObject:
    obj = deepcopy(obj)
    if obj.shape == 'circle':
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
    if obj.shape == 'circle':
        pass
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


def create_mesh(height:float = 0.41, width:float= 2.2, objects: list[object] = [gObject()], mesh_size = 0.0225):
    with pygmsh.geo.Geometry() as geom:
        geo_objects = []
        for obj in objects:
            if obj.shape == 'circle':
                geo_objects.append(geom.add_circle(
                    x0=obj.args['x0'],
                    radius=obj.args['radius'],
                    mesh_size=mesh_size,
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
                    mesh_size=mesh_size,
                    make_surface=False,
                ))
            elif obj.shape == 'tri':
                geo_objects.append(geom.add_polygon(
                    points=[obj.args['x0'] + [0], obj.args['x1'] + [0], obj.args['x2'] + [0]],
                    mesh_size=mesh_size,
                    make_surface=False,
                ))
            else:
                raise Exception(f'Shape {obj.shape} not supported.')

        geom.add_rectangle(
            0.0, width, 0.0, height, 0.0, mesh_size=mesh_size, holes=[obj.curve_loop for obj in geo_objects]
        )
        mesh = geom.generate_mesh()

    # pygmsh.optimize(mesh, verbose = True)

    # remove z-coordinate
    mesh = meshio.Mesh(mesh.points[:, :2], {"triangle": mesh.get_cells_type("triangle")})

    return mesh

tri = create_equi_tri([0.2, 0.2], 0.05, -90)
tri = squish_object(tri, 1.0, 2.0)
mesh = create_mesh(objects=[tri])

#meshio.svg.write("mesh.svg", mesh, float_fmt=".3f", stroke_width="0.1")
mesh.write("mesh.xdmf")