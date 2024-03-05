from meshMaker import *
import argparse

"""
    Script to generate a dataset of meshes for the vortex shedding problem.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=float, default=1.6, help="Width of the channel.")
parser.add_argument("--height", type=float, default=0.41, help="Height of the channel.")
parser.add_argument("--ox", type=float, default=0.33, help="Mean object x mid point.")
parser.add_argument("--oy", type=float, default=0.2, help="Mean object y mid point.")
parser.add_argument("--osize", type=float, default=0.05, help="Mean object size. (e.g. radius for circles)")
parser.add_argument("--inflow_peak_mean", type=float, default=1.25, help="Mean of the inflow peak.")
parser.add_argument("--inflow_peak_std", type=float, default=0.55, help="Standard deviation of the inflow peak.")
parser.add_argument("--num_meshes", type=int, default=441, help="Number of meshes to generate.")
parser.add_argument('--name', type=str,help='Name of mesh dataset. Is saved under meshes/NAME. Default is standard_cylinder.')
parser.add_argument("--two_obj", action="store_true", help="If set, allow two objects to be generated.")
parser.add_argument("--rotate", action="store_true", help="If set, allow objects to be randomly rotated.")
parser.add_argument("--stretch", action="store_true", help="If set, allow objects to be randomly stretched/squeezed.")
parser.add_argument("--circs", action="store_true", help="If set, add circles to the set of possible objects.")
parser.add_argument("--tris", action="store_true", help="If set, add triangles to the set of possible objects.")
parser.add_argument("--quads", action="store_true", help="If set, add rectangles to the set of possible objects.")
args = parser.parse_args()

if not args.circs and not args.tris and not args.quads:
    raise Exception("At least one shape must be enabled.")

if not args.name:
    raise Exception("Name must be set.")

width = args.width
height = args.height
object_x_mid = args.ox
object_y_mid = args.oy
object_size = args.osize
inflow_peak, inflow_std = args.inflow_peak_mean, args.inflow_peak_std

num_meshes = args.num_meshes #Training +  test + intermediate evaluation

def sample_gauss(mean, std, cap=2):
    return min(mean+cap*std,max(mean-cap*std,np.random.normal(mean, std)))

def standard_cylinder_mesh_set():
    meshes = []

    #Generate meshes
    for i in range(num_meshes):
        print(f"Creating mesh {i+1}/{num_meshes}")

        x0 = [sample_gauss(object_x_mid, 0.1), sample_gauss(object_y_mid, 0.05)] if i > 0 else [object_x_mid, object_y_mid]
        r = sample_gauss(object_size, 0.02) if i > 0 else object_size
        circ = create_ellipse(x0, r, r)

        meshes.append(create_mesh(height=height,width=width,objects=[circ]))
        meshes[-1][1]["inflow_peak"] = round(sample_gauss(inflow_peak,inflow_std), 3) if i > 0 else inflow_peak

    #Save meshes:
    for i, mesh in enumerate(meshes):
        save_mesh(mesh[0],mesh[1], f"mesh{i+1}", "meshes/standard_cylinder")

def mixed_mesh_set(two_objs = False, circles = True, tris = False, quads = False, stretching = False, rotate = False, name: str = "mixed"):
    meshes = []

    def make_object(x0):
        ellipse_width = sample_gauss(object_size, 0.02)
        ellipse_height = sample_gauss(object_size, 0.02) if stretching else ellipse_width

        triangle_size = sample_gauss(object_size*1.5, 0.02)
        triangle_squish = sample_gauss(1, 0.2)
        triangle_angle = np.random.rand() * 180 if rotate else 0

        rect_width = sample_gauss(math.sqrt(2)*object_size, 0.02)
        rect_height = sample_gauss(math.sqrt(2)*object_size, 0.02) if stretching else rect_width
        rect_angle = (-45 + np.random.rand() * 90) if rotate else 0

        ellipse = create_ellipse(x0, ellipse_width, ellipse_height)
        tri = rotate_object(squish_object(create_equi_tri(x0, triangle_size), triangle_squish if stretching else 1, 1/triangle_squish if stretching else 1),triangle_angle)
        rect = rotate_object(create_rect(x0, rect_width, rect_height),rect_angle)

        objects = []
        objects.append(ellipse) if circles else None
        objects.append(tri) if tris else None
        objects.append(rect) if quads else None
        return objects[np.random.randint(len(objects))]

    #Generate meshes
    for i in range(num_meshes):
        if i %20== 0:
            print(f"Generating mesh {i+1} of {num_meshes}")
        x0 = [sample_gauss(object_x_mid, 0.1, 2), sample_gauss(object_y_mid, 0.05, 2)]

        objects = [make_object(x0)] if i > 0 else [create_ellipse([object_x_mid,object_y_mid], object_size, object_size)]
        if np.random.rand() <= 0.25 and two_objs and i > 0: #Make it 'rare' to have two objects
            x0 = [sample_gauss(object_x_mid + 0.5, 0.1), sample_gauss(object_y_mid, 0.05)]
            objects.append(make_object(x0))
        meshes.append(create_mesh(height=height,width=width,objects=objects))
        meshes[-1][1]["inflow_peak"] = round(sample_gauss(inflow_peak,inflow_std), 2) if i > 0 else inflow_peak

    #Save meshes:
    for i, mesh in enumerate(meshes):
        save_mesh(mesh[0],mesh[1], f"mesh{i+1}", f"meshes/{name}")


mixed_mesh_set(args.two_obj, args.circs, args.tris, args.quads, args.stretch, args.rotate, args.name)


# standard_cylinder_mesh_set()
# mixed_mesh_set(True,True,False,False,False,False,"2cylinders")
# mixed_mesh_set(False,True,True,True,False,True,"cylinder_tri_quad")
# mixed_mesh_set(False,True,False,False,True,False,"cylinder_stretch")
# mixed_mesh_set(True,True,True,True,True,True,"mixed_all")

