from meshMaker import *

width = 1.6
height = 0.41
object_x_mid = 0.33
object_y_mid = 0.2
object_size = 0.05
inflow_peak, inflow_std = 1.25, 0.55

def sample_gauss(mean, std, cap=2):
    return min(mean+cap*std,max(mean-cap*std,np.random.normal(mean, std)))

def standard_cylinder_mesh_set():
    meshes = []
    num_meshes = 400 + 10 + 1 #Training +  test + intermediate evaluation

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
    num_meshes = 0 + 10 + 1 #Training +  test + intermediate evaluation

    def make_object(x0):
        ellipse_width = sample_gauss(object_size, 0.02)
        ellipse_height = sample_gauss(object_size, 0.02) if stretching else ellipse_width

        angle = np.random.rand() * 180 if rotate else 0

        triangle_size = sample_gauss(object_size*1.5, 0.02)
        triangle_squish = sample_gauss(1, 0.2)

        rect_width = sample_gauss(2*object_size, 0.04)
        rect_height = sample_gauss(2*object_size, 0.04) if stretching else rect_width

        ellipse = create_ellipse(x0, ellipse_width, ellipse_height)
        tri = rotate_object(squish_object(create_equi_tri(x0, triangle_size), triangle_squish if stretching else 1, 1/triangle_squish if stretching else 1),angle)
        rect = rotate_object(create_rect(x0, rect_width, rect_height),angle)

        objects = []
        objects.append(ellipse) if circles else None
        objects.append(tri) if tris else None
        objects.append(rect) if quads else None
        return objects[np.random.randint(len(objects))]


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

#standard_cylinder_mesh_set()
#mixed_mesh_set(True,True,False,False,False,False,"2cylinders")
#mixed_mesh_set(False,True,True,False,False,True,"cylinder_tri")
#mixed_mesh_set(False,True,False,True,False,True,"cylinder_quad")
#mixed_mesh_set(False,True,False,False,True,False,"cylinder_stretch")
#mixed_mesh_set(True,True,True,True,True,True,"mixed_all")




