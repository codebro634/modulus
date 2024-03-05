import numpy as np
import pygmsh
import meshio
from copy import deepcopy
import math
import os
import json

"""
    gObject defines a geometric which will be a hole in the rectangular mesh domain.
    
    shape: str - The shape of the object. Supported shapes are 'ellipse', 'rect' and 'tri'
    args: dict - The arguments for the shape. For 'ellipse' the arguments are 'x0': [x, y], 'w': width, 'h': height
                                              For 'rect' the arguments are 'x0': [x, y], 'x1': [x, y], 'x2': [x, y], 'x3': [x, y]
                                              For 'tri' the arguments are 'x0': [x, y], 'x1': [x, y], 'x2': [x, y]
"""

class gObject:

    def __init__(self, shape: str = 'rect', args:dict = {'x0': [0.33, 0.2], 'x1': [0.38, 0], 'x2': [0.33, 0.25]}):
        self.shape = shape
        self.args = args

    #Returns the string to be used in the boundary condition of the object for Fenics
    def boundary_string(self):
        tol, dp = 0.05, 5
        if self.shape == 'ellipse':
            return f"on_boundary && x[0]>{round(self.args['x0'][0] - tol - self.args['w'],dp)} && " \
                   f"x[0]< {round(self.args['x0'][0] + tol + self.args['w'],dp)} && " \
                   f"x[1]>{round(self.args['x0'][1] - tol - self.args['h'],dp)} && " \
                   f"x[1]< {round(self.args['x0'][1] + tol + self.args['h'],dp)}"
        elif self.shape == 'rect':
            xmin = min(self.args['x0'][0], self.args['x1'][0], self.args['x2'][0], self.args['x3'][0])
            xmax = max(self.args['x0'][0], self.args['x1'][0], self.args['x2'][0], self.args['x3'][0])
            ymin = min(self.args['x0'][1], self.args['x1'][1], self.args['x2'][1], self.args['x3'][1])
            ymax = max(self.args['x0'][1], self.args['x1'][1], self.args['x2'][1], self.args['x3'][1])
            return f"on_boundary && x[0]>{round(xmin - tol,dp)} && " \
                      f"x[0]< {round(xmax + tol,dp)} && " \
                        f"x[1]>{round(ymin - tol,dp)} && " \
                        f"x[1]< {round(ymax + tol,dp)}"
        elif self.shape == 'tri':
            return f"on_boundary && x[0]>{round(min(self.args['x0'][0], self.args['x1'][0], self.args['x2'][0]) - tol,dp)} && " \
                   f"x[0]< {round(max(self.args['x0'][0], self.args['x1'][0], self.args['x2'][0]) + tol,dp)} && " \
                   f"x[1]>{round(min(self.args['x0'][1], self.args['x1'][1], self.args['x2'][1]) - tol,dp)} && " \
                   f"x[1]< {round(max(self.args['x0'][1], self.args['x1'][1], self.args['x2'][1]) + tol,dp)}"
        else:
            raise Exception(f'Shape {self.shape} not supported for boundary string')

"""
    Methods for manipulating gObjects:
    move_object: Moves the object by dx and dy
    rotate_object: Rotates the object by an angle
    squish_object: Squishes the object by xstretch and ystretch
"""

def move_object(obj: gObject, dx:float = 0.0, dy:float = 0.0) -> gObject:
    obj = deepcopy(obj)
    if obj.shape == 'ellipse':
        obj.args['x0'][0] += dx
        obj.args['x0'][1] += dy
    elif obj.shape == 'rect':
        obj.args['x0'][0] += dx
        obj.args['x0'][1] += dy
        obj.args['x1'][0] += dx
        obj.args['x1'][1] += dy
        obj.args['x2'][0] += dx
        obj.args['x2'][1] += dy
        obj.args['x3'][0] += dx
        obj.args['x3'][1] += dy
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

def rotate(x, y, angle):
    angle = math.radians(angle)
    return [(x[0] - y[0]) * math.cos(angle) - (x[1] - y[1]) * math.sin(angle) + y[0],
            (x[0] - y[0]) * math.sin(angle) + (x[1] - y[1]) * math.cos(angle) + y[1]]

def rotate_object(obj: gObject, angle: float) -> gObject:
    obj = deepcopy(obj)
    if obj.shape == 'ellipse':
        return obj
    elif obj.shape == 'rect':
        mid = (obj.args['x0'][0] + obj.args['x1'][0] + obj.args['x2'][0] + obj.args['x3'][0]) /4, (obj.args['x0'][1] + obj.args['x1'][1] + obj.args['x2'][1] + obj.args['x3'][1]) /4
        obj.args['x0'] = rotate(obj.args['x0'], mid, angle)
        obj.args['x1'] = rotate(obj.args['x1'], mid, angle)
        obj.args['x2'] = rotate(obj.args['x2'], mid, angle)
        obj.args['x3'] = rotate(obj.args['x3'], mid, angle)
        return obj
    elif obj.shape == 'tri':
        mid = [(obj.args['x0'][0] + obj.args['x1'][0] + obj.args['x2'][0])/3, (obj.args['x0'][1] + obj.args['x1'][1] + obj.args['x2'][1])/3]
        obj.args['x0'] = rotate(obj.args['x0'], mid, angle)
        obj.args['x1'] = rotate(obj.args['x1'], mid, angle)
        obj.args['x2'] = rotate(obj.args['x2'], mid, angle)
        return obj
    else:
        raise Exception(f'Shape {obj.shape} not supported for rotation')


def squish_object(obj: gObject, xstretch: float = 1.0, ystretch: float = 1.0) -> gObject:
    obj = deepcopy(obj)
    if obj.shape == 'ellipse':
        obj.args['w'] *= xstretch
        obj.args['h'] *= ystretch
    elif obj.shape == 'rect':
        mid = (obj.args['x0'][0] + obj.args['x1'][0] + obj.args['x2'][0] + obj.args['x3'][0]) /4, (obj.args['x0'][1] + obj.args['x1'][1] + obj.args['x2'][1] + obj.args['x3'][1]) /4
        obj.args['x0'][0] = mid[0] + (obj.args['x0'][0] - mid[0])*xstretch
        obj.args['x0'][1] = mid[1] + (obj.args['x0'][1] - mid[1])*ystretch
        obj.args['x1'][0] = mid[0] + (obj.args['x1'][0] - mid[0])*xstretch
        obj.args['x1'][1] = mid[1] + (obj.args['x1'][1] - mid[1])*ystretch
        obj.args['x2'][0] = mid[0] + (obj.args['x2'][0] - mid[0])*xstretch
        obj.args['x2'][1] = mid[1] + (obj.args['x2'][1] - mid[1])*ystretch
        obj.args['x3'][0] = mid[0] + (obj.args['x3'][0] - mid[0])*xstretch
        obj.args['x3'][1] = mid[1] + (obj.args['x3'][1] - mid[1])*ystretch
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




def create_ellipse(mid, w,h) -> gObject:
    return gObject(shape='ellipse', args={'x0': mid, 'w': w, 'h': h })

def create_rect(mid, w, h) -> gObject:
    return gObject(shape='rect', args={'x0': [mid[0]+w/2, mid[1]+h/2], 'x1': [mid[0]-w/2, mid[1]+h/2], 'x2': [mid[0]-w/2, mid[1]-h/2], 'x3': [mid[0]+w/2, mid[1]-h/2]})

def create_equi_tri(mid, r) -> gObject:
    a = math.radians(60)
    corners = [mid[0] , mid[1] - r], [mid[0] + r*math.sin(a), mid[1] + r*math.cos(a)], [mid[0] - r*math.sin(a), mid[1] + r*math.cos(a)]
    tri = gObject(shape='tri', args={'x0': corners[0], 'x1': corners[1], 'x2': corners[2]})
    return rotate_object(tri, 30)

"""
    create_mesh creates a triangular mesh with a rectangular domain and with holes defined by the objects in the objects list.
    
    height: float - The height of the rectangular domain
    width: float - The width of the rectangular domain
    objects: list[gObject] - The list of objects that will be holes in the mesh
    mesh_size: float - The size of the mesh triangles
"""
def create_mesh(height:float = 0.41, width:float= 1.6, objects: list[object] = [gObject()], mesh_size = 0.0225):
    with pygmsh.geo.Geometry() as geom:
        geo_objects = []
        for obj in objects:
            if obj.shape == 'ellipse':
                num_sections = 32
                points =  [(obj.args['x0'][0] + obj.args['w'] * np.cos(2 * np.pi * k / num_sections),
                            obj.args['x0'][1] + obj.args['h'] * np.sin(2 * np.pi * k / num_sections)) for k in range(num_sections) ]
                geo_objects.append(
                    geom.add_polygon(
                        points=points,
                        mesh_size=mesh_size,
                        make_surface=False,
                    )
                )
            elif obj.shape == 'rect':
                geo_objects.append(geom.add_polygon(
                    points=[obj.args['x0'], obj.args['x1'], obj.args['x2'], obj.args['x3']],
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

    #Collect paths for saving
    path_mesh = os.path.join(path, 'mesh.xdmf')
    path_img = os.path.join(path, 'img.svg')
    path_metadata = os.path.join(path, 'metadata.json')

    #Save mesh, image and metadata
    mesh.write(path_mesh)
    meshio.svg.write(path_img, mesh, float_fmt=".3f", stroke_width="0.1")
    with open(path_metadata, 'w') as f:
        json.dump(metadata, f)


#tri = create_equi_tri([0.33, 0.2], 0.05)
#tri = squish_object(tri, 1.0, 1.0)
#rect = rotate_object(create_rect([0.33, 0.2], 0.05, 0.1),45)
#circ = create_ellipse([0.33, 0.2], 0.05,0.05)
#mesh, metadata = create_mesh(objects=[circ])
#save_mesh(mesh, metadata, 'test', 'meshes')