from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import imageio
import argparse
import json
import math
import psutil
import gc

"""
    Script for flow simulation using FEniCS and the Incremental Pressure Correction Scheme (IPCS). 
    The Code is a modification of the tutorial found at: https://fenicsproject.org/pub/tutorial/html/._ftut1009.html
"""

#TODO check if this code works for multiple obstacles

parser = argparse.ArgumentParser()
parser.add_argument("--num_frames",type=int, default=0, help="If > 0, save animation of simulation as gif with num_frames frames.")
parser.add_argument('--dont_save', action='store_true', help='If set, the simulation results are NOT saved and simply discarded.')
parser.add_argument('--qoi', action='store_true', help='If set, calculate and save quantities of interest for the first simulation. Assumes that the mesh/inflow is that of DFG cylinder flow 2D-2 benchmark.')
parser.add_argument("--vlevel", type=int, default=1, help="Verbosity level. 0 = no verbosity.")
parser.add_argument("--dt", type=float, default=0.0005, help="Delta t.")
parser.add_argument("--saveN", type=int, default=20, help="Every how many steps to save.")
parser.add_argument("--steps", type=int, default=6020, help="Num of simulation steps.")
parser.add_argument('--dir', default="datasets/test", help='Path to where results are stored')
parser.add_argument('--mesh', default=None, help='Path to mesh. May also be a folder containing meshes.')
parser.add_argument('--mesh_range', default=None, help='Range of meshes to use. If None, all meshes are used.')
parser.add_argument("--cleanup_dir", default=None, help="Recursively searches through and reruns all erroneous simulations.")
args = parser.parse_args()

results_dir = args.dir
os.makedirs(results_dir, exist_ok=True)

mesh_paths = []
# If necessary, add meshes to resimulate to list
if args.cleanup_dir is not None:
    for root, dirs, files in os.walk(args.cleanup_dir):
        if 'failed_meshes.txt' in files:
            with open(os.path.join(root, 'failed_meshes.txt'), 'r') as f:
                mesh_paths += f.readlines()
    mesh_paths = [x.strip() for x in mesh_paths]
    if len(mesh_paths) == 0:
        raise Exception(f"No erreoneous simulations found in {args.cleanup_dir}")
elif args.mesh is None:
    mesh_paths = [None]
# Collect all meshes found in the args.mesh folder and its subfolders
else:
    mesh_paths = []
    for root, dirs, files in os.walk(args.mesh):
        if 'metadata.json' in files:
            mesh_paths.append(root)
    if len(mesh_paths) == 0:
        raise Exception(f"No meshes found in {args.mesh}")

if args.mesh_range is not None:
    start,end = args.mesh_range.split(',')
    assert int(end) > int(start)
    mesh_paths = mesh_paths[int(start):int(end)]

sims_data, failed_meshes = [], [] #One entry for each simulation
num_frames = args.num_frames
t_thrs = 25.0 #Only needed when quantities of interest are calculated
for sim, mesh_path in enumerate(mesh_paths):

    if num_frames > 0 and sim==0:
        plot_path = os.path.join(results_dir,"animation")
        if not os.path.exists(plot_path):
            os.makedirs(plot_path,exist_ok=True)
    
    
    start = time.time()

    # Setup parameters
    num_steps = args.steps   # number of time steps
    dt = args.dt # time step size
    N_save = args.saveN #Every N-th time step is saved

    #Default PDE parameters
    mu = 0.001         # dynamic viscosity
    rho = 1            # density
    inflow_peak = 1.5 #Can be overwritten by if it is specified in Mesh's metadata
    
    # Create/Load mesh
    if mesh_path is not None:
        mesh = Mesh()
        with XDMFFile(os.path.join(mesh_path, "mesh.xdmf")) as infile:
            infile.read(mesh)
        with open(os.path.join(mesh_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        channel_width = metadata['width']
        channel_height = metadata['height']
        obstacle_condition = " || ".join(['(' + x + ')' for x in metadata['object_boundaries']])
        if inflow_peak in metadata:
            inflow_peak = metadata['inflow_peak']
    else:
        channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
        obstacle = Circle(Point(0.2, 0.2), 0.05)
        domain = channel - obstacle
        mesh = generate_mesh(domain, 64)
        channel_width = 2.2
        channel_height = 0.41
        obstacle_condition = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

    if args.vlevel > 0:
        print(f"{mesh.num_vertices()} vertices in mesh.",flush=True)
        print(f"Width: {channel_width}, Height: {channel_height}",flush=True)
        print(f"Obstacle condition: {obstacle_condition}",flush=True)
    
    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    
    # Define boundaries
    inflow   = 'near(x[0], 0)'
    outflow  = f'near(x[0], {channel_width})'
    walls    = f'near(x[1], 0) || near(x[1], {channel_height})'
    obstacle = obstacle_condition
    
    # Define inflow profile
    inflow_profile = (f'4.0*{inflow_peak}*x[1]*({channel_height} - x[1]) / pow({channel_height}, 2)', '0')
    
    # Define boundary conditions
    bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu_cylinder = DirichletBC(V, Constant((0, 0)), obstacle)
    bcp_outflow = DirichletBC(Q, Constant(0), outflow)
    bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
    bcp = [bcp_outflow]
    
    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    
    # Define functions for solutions at previous and current time steps
    u_n = Function(V)
    u_  = Function(V)
    p_n = Function(Q)
    p_  = Function(Q)
    
    # Define expressions used in variational forms
    U  = 0.5*(u_n + u)
    n  = FacetNormal(mesh)
    f  = Constant((0, 0))
    k  = Constant(dt)
    mu = Constant(mu)
    rho = Constant(rho)
    
    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))
    
    # Define stress tensor
    def sigma(u, p):
        return 2*mu*epsilon(u) - p*Identity(len(u))
    
    # Define variational problem for step 1
    F1 = rho*dot((u - u_n) / k, v)*dx + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx + inner(sigma(U, p_n), epsilon(v))*dx + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)
    
    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx
    
    # Define variational problem for step 3
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx
    
    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)
    
    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]
    
    
    # Create time series
    tut,tpt = os.path.join(results_dir, 'tmp1'), os.path.join(results_dir,'tmp2')
    timeseries_u = TimeSeries(tut)
    timeseries_p = TimeSeries(tpt)
    
    # Time-stepping
    t = 0
    image_v_locs, image_p_locs, error_raised = [],[], False
    
    for n in range(num_steps): #num_steps
    
        # Update current time
        t += dt

        try:
            # Step 1: Tentative velocity step
            b1 = assemble(L1)
            [bc.apply(b1) for bc in bcu]
            solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

            # Step 2: Pressure correction step
            b2 = assemble(L2)
            [bc.apply(b2) for bc in bcp]
            solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

            # Step 3: Velocity correction step
            b3 = assemble(L3)
            solve(A3, u_.vector(), b3, 'cg', 'sor')
        except RuntimeError:
            print(f"Error raised at time step {n} for mesh {mesh_path}.",flush=True)
            failed_meshes.append(mesh_path)
            error_raised = True
            break
    
        # Plot solution
        if num_frames > 0 and n % max(num_steps // num_frames, 1) == 0 and sim == 0:
            title=f"velocity{n}"
            plot(u_, title=title)
            plt.savefig(os.path.join(plot_path,title+".png"))
            image_v_locs.append(os.path.join(plot_path,title+".png"))
    
            title = f"pressure{n}"
            plot(p_,title=title)
            plt.savefig(os.path.join(plot_path,title+".png"))
            image_p_locs.append(os.path.join(plot_path,title+".png"))


        # Save nodal values to file
        if (not args.dont_save) or (t >= t_thrs and args.qoi):  # Second condition applies only for benchmark
            timeseries_u.store(u_.vector(), t)
            timeseries_p.store(p_.vector(), t)
    
        # Update previous solution
        u_n.assign(u_)
        p_n.assign(p_)
    
        # Print progress
        progress_str = None
        if args.vlevel == 2:
            progress_str = f"Progress {n/num_steps} in simulation {sim}/{len(mesh_paths)}"
        elif args.vlevel == 1 and n%100 == 0:
            progress_str = f"Progress {n/num_steps} in simulation {sim}/{len(mesh_paths)}"

        if progress_str is not None:
            print(progress_str,flush=True)
            with open(os.path.join(results_dir, 'progress.txt'), 'a') as f:
                f.write(progress_str+"\n")

    if error_raised:
        if os.path.exists(tut + ".h5"):
            os.remove(tut + ".h5")
        if os.path.exists(tpt + ".h5"):
            os.remove(tpt + ".h5")
        continue

    if args.vlevel > 0:
        print(f"Duration: {round(time.time() -  start,3)}s",flush=True)
    
    #Create animation
    if num_frames > 0 and sim == 0:
        duration = (num_steps // num_frames) * dt / 4 #Divided by 4 to have a bit of slow-motion
        with imageio.get_writer(os.path.join(plot_path,'velocity.gif'), mode='I', duration=duration) as writer:
            for image in image_v_locs:
                img = imageio.imread(image)
                writer.append_data(img)
    
        for image in image_v_locs:
            os.remove(image)
    
        with imageio.get_writer(os.path.join(plot_path,'pressure.gif'), mode='I', duration=duration) as writer:
            for image in image_p_locs:
                img = imageio.imread(image)
                writer.append_data(img)
    
        for image in image_p_locs:
            os.remove(image)
    
    
    """Extract data in numpy format from the simulation"""
    if not args.dont_save:
        n = mesh.num_vertices()
        sim_data = dict()

        #Extract velocities
        velocity = np.zeros(shape=(num_steps, n, 2),dtype=np.float32)
        times_v = timeseries_u.vector_times()
        for i,t in enumerate(times_v):
            timeseries_u.retrieve(u_.vector(),t)
            x = u_.compute_vertex_values(mesh) #The same as calling u_ at the coordinates of each vertex
            velo_t = np.concatenate( (x[0:n,np.newaxis],x[n:,np.newaxis]), axis=-1)
            velocity[i] = velo_t
        sim_data['velocity'] = velocity

        #Extract pressure values
        pressure = np.zeros(shape=(num_steps,n,1),dtype=np.float32)
        times_p = timeseries_p.vector_times()
        for i,t in enumerate(times_p):
            timeseries_p.retrieve(p_.vector(),t)
            x = p_.compute_vertex_values(mesh)
            pressure[i] = x[:,np.newaxis]
        sim_data['pressure'] = pressure

        #Extract mesh graph data
        sim_data['cells'] = np.repeat(np.array(list(mesh.cells()),dtype=np.int32)[np.newaxis,...],num_steps,axis=0)
        sim_data['mesh_pos'] = np.repeat(np.array(list(mesh.coordinates()),dtype=np.float32)[np.newaxis,...],num_steps,axis=0)


        def get_vertices_with_cond(mesh, condition): #From https://fenicsproject.org/qa/2989/vertex-on-mesh-boundary/
            V = FunctionSpace(mesh, 'CG', 1)
            bc = DirichletBC(V, 1, condition)
            u = Function(V)
            bc.apply(u.vector())
            d2v = dof_to_vertex_map(V)
            vertices_on_boundary = d2v[u.vector() == 1.0]
            return vertices_on_boundary

        #Extract node types
        vertex_types = []
        v_inflow, v_outflow, v_walls, v_cylinder  = get_vertices_with_cond(mesh,inflow), get_vertices_with_cond(mesh,outflow), get_vertices_with_cond(mesh,walls), get_vertices_with_cond(mesh, obstacle)
        for v in range(mesh.num_vertices()):
            if v in v_inflow and not v in v_walls: #As in deepminds dataset, the four corners are considered boundaries
                vertex_types.append(4)
            elif v in v_outflow and not v in v_walls:
                vertex_types.append(5)
            elif v in v_walls or v in v_cylinder:
                vertex_types.append(6)
            else:
                vertex_types.append(0)

        sim_data['node_type'] = np.repeat(np.array(vertex_types,dtype=np.int32)[np.newaxis,:,np.newaxis],num_steps,axis=0)

        #Only save every N-th time step and discard the rest
        for k, v in sim_data.items():
            sim_data[k] = sim_data[k][(N_save-1)::N_save] #Skip first data points to avoid initial chaos phase

        sims_data.append(sim_data)

    #Calculate quantities of interest (for benchmark only)
    if sim == 0 and args.qoi:

        times_v = timeseries_u.vector_times()
        times_p = timeseries_p.vector_times()

        if args.vlevel > 0:
            print("Calculating quantities of interest",flush=True)

        num_points = 64
        cpoints = [(0.2 + 0.05 * np.cos(2 * np.pi * k / num_points), 0.2 + 0.05 * np.sin(2 * np.pi * k / num_points))
                   for k in range(num_points)]

        normal_vecs = []
        for k in range(num_points):
            dx_dk = -0.05 * np.sin(2 * np.pi * k / num_points) * (2 * np.pi / num_points)
            dy_dk = 0.05 * np.cos(2 * np.pi * k / num_points) * (2 * np.pi / num_points)
            nx = dy_dk
            ny = -dx_dk
            length = np.sqrt(nx ** 2 + ny ** 2)
            nx /= length
            ny /= length
            normal_vecs.append((nx, ny))

        dp,cd,cl = [], [], []
        times = []
        for j in range(len(times_v)):
            if j % N_save != 0 or times_v[j] < t_thrs:
                continue
            assert times_v[j] == times_p[j]

            if (args.vlevel > 0 and j % 10 == 0) or args.vlevel > 1:
                print(f"Progress: {j/N_save}/{len(times_v)//N_save}", flush=True)

            times.append(times_v[j])
            timeseries_u.retrieve(u_.vector(), times_v[j])
            timeseries_p.retrieve(p_.vector(), times_p[j])

            fd, fl = 0, 0
            for i in range(num_points):
                normal_vec = normal_vecs[i]
                sigma = mu * (nabla_grad(u_) + grad(u_)) - p_ * Identity(2)
                vec = dolfin.project(dot(sigma, as_vector(normal_vec)))(cpoints[i])
                fd += vec[0]
                fl += vec[1]

            fd *= (0.1*math.pi)/num_points
            fl *= (0.1*math.pi)/num_points
            cd_ = 2 * fd / 0.1
            cl_ = 2 * fl / 0.1

            deltaP = p_((0.15,0.2)) - p_(0.25,0.2)

            dp.append(deltaP)
            cd.append(cd_)
            cl.append(cl_)

        dp = np.array(dp)
        cd = np.array(cd)
        cl = np.array(cl)

        #Get frequency of lift coefficient peaks
        max_idx = np.argmax(cl)
        past_peak_idx = None
        for i in range(max_idx, len(cl)):
            if times[i] - times[max_idx] >= 0.2:
                past_peak_idx = i
                break
        next_peaks = np.argwhere(cl[past_peak_idx:] == np.max(cl[past_peak_idx:]))
        frequency = 1 / (times[next_peaks[0][0] + past_peak_idx] - times[max_idx])

        strouhal = frequency * 0.1
        drag_coef = np.max(cd)
        lift_coef = np.max(cl)
        max_delta_p = np.max(dp)

        print(f"Frequency: {frequency} Strouhal number: {strouhal}, Drag coefficient: {drag_coef}, Lift coefficient: {lift_coef}, Max delta P: {max_delta_p}",flush=True)

        #Save arrays as human readable format
        np.savetxt(results_dir + "/drag_coefficient.txt",cd,delimiter=',',fmt="%s")
        np.savetxt(results_dir + "/lift_coefficient.txt",cl,delimiter=',',fmt="%s")
        np.savetxt(results_dir + "/delta_p.txt",dp,delimiter=',',fmt="%s")
        np.savetxt(results_dir + "/times.txt",times,delimiter=',',fmt="%s")

    #Memory cleanup
    os.remove(tut+".h5")
    os.remove(tpt+".h5")
    timeseries_u.close()
    timeseries_p.close()
    sims_data = []
    gc.collect()

#Save all simulations into a single file
if not args.dont_save:
    np.save(results_dir+"/simdata.npy",sims_data)
    np.savetxt(results_dir+"/failed_meshes.txt",failed_meshes, delimiter=',', fmt="%s")


