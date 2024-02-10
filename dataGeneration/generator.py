from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import imageio
import argparse
import json

#TODO check if this code works for multiple obstacles

parser = argparse.ArgumentParser()
parser.add_argument("--p", action="store_true", help="If set, save gif of the simulation.")
parser.add_argument("--v", action="store_true", help="Activate verbosity.")
parser.add_argument("--dt", type=float, default=0.001, help="Delta t.")
parser.add_argument("--steps", type=int, default=3000, help="Num of simulation steps.")
parser.add_argument('--dir', default="navier_stokes_cylinder", help='Path to where results are stored')
parser.add_argument('--mesh', default="meshes/standard", help='Path to mesh. May also be a folder containing meshes.')
args = parser.parse_args()

results_dir = args.dir
os.makedirs(results_dir, exist_ok=True)

if args.mesh is None:
    mesh_paths = [None]
else:
    mesh_paths = []
    for root, dirs, files in os.walk(args.mesh):
        if 'metadata.json' in files:
            mesh_paths.append(root)


sims_data = [] #One entry for each simulation
for sim, mesh_path in enumerate(mesh_paths):

    #Plotting params
    num_imgs = 100
    
    if args.p and sim==0:
        plot_path = os.path.join(results_dir,"animation")
        if not os.path.exists(plot_path):
            os.makedirs(plot_path,exist_ok=True)
    
    
    start = time.time()
    
    T = 6.0            # final time
    num_steps = args.steps   # number of time steps
    dt = args.dt # time step size
    N_save = 10 #Every N-th time step is saved
    mu = 0.001         # dynamic viscosity
    rho = 1            # density
    inflow_peak = 1.25
    
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
    else:
        channel = Rectangle(Point(0, 0), Point(1.6, 0.41))
        obstacle = Circle(Point(0.33, 0.2), 0.05)
        domain = channel - obstacle
        mesh = generate_mesh(domain, 64)
        channel_width = 1.6
        channel_height = 0.41
        obstacle_condition = 'on_boundary && x[0]>0.23 && x[0]<0.43 && x[1]>0.1 && x[1]<0.3'

    if args.v:
        print(f"{mesh.num_vertices()} vertices in mesh.")
        print(f"Width: {channel_width}, Height: {channel_height}")
        print(f"Obstacle condition: {obstacle_condition}")
    
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
    
    
    # Create time series (for use in reaction_system.py)
    tut,tpt = os.path.join(results_dir, 'tmp1'), os.path.join(results_dir,'tmp2')
    timeseries_u = TimeSeries(tut)
    timeseries_p = TimeSeries(tpt)
    
    # Time-stepping
    t = 0
    image_v_locs,image_p_locs = [],[]
    
    for n in range(num_steps): #num_steps
    
        # Update current time
        t += dt
    
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
    
        # Plot solution
        if args.p and n % (num_steps // num_imgs) == 0:
            title=f"velocity{n}"
            plot(u_, title=title)
            plt.savefig(os.path.join(plot_path,title+".png"))
            image_v_locs.append(os.path.join(plot_path,title+".png"))
    
            title = f"pressure{n}"
            plot(p_,title=title)
            plt.savefig(os.path.join(plot_path,title+".png"))
            image_p_locs.append(os.path.join(plot_path,title+".png"))
    
        # Save nodal values to file
        timeseries_u.store(u_.vector(), t)
        timeseries_p.store(p_.vector(), t)
    
        # Update previous solution
        u_n.assign(u_)
        p_n.assign(p_)
    
        # Print progress
        if args.v:
            print(f"Progress {t/T}")
        
    if args.v:
        print(f"Duration: {round(time.time() -  start,3)}s")
    
    #Create animation
    if args.p:
        duration = (num_steps // num_imgs) * dt / 4 #Divided by 4 to have a bit of slow-motion
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
    
    
    #Save data in numpy format
    n = mesh.num_vertices()
    sim_data = dict()
    
    velocity = np.zeros(shape=(num_steps,n,2),dtype=np.float32)
    times_v = timeseries_u.vector_times()
    for i,t in enumerate(times_v):
        timeseries_u.retrieve(u_.vector(),t)
        x = u_.compute_vertex_values(mesh) #The same as calling u_ at the coordinates of each vertex
        velo_t = np.concatenate( (x[0:n,np.newaxis],x[n:,np.newaxis]), axis=-1)
        velocity[i] = velo_t
    sim_data['velocity'] = velocity
    
    pressure = np.zeros(shape=(num_steps,n,1),dtype=np.float32)
    times_p = timeseries_p.vector_times()
    for i,t in enumerate(times_p):
        timeseries_p.retrieve(p_.vector(),t)
        x = p_.compute_vertex_values(mesh)
        pressure[i] = x[:,np.newaxis]
    sim_data['pressure'] = pressure
    
    sim_data['cells'] = np.repeat(np.array(list(mesh.cells()),dtype=np.int32)[np.newaxis,...],num_steps,axis=0)
    sim_data['mesh_pos'] = np.repeat(np.array(list(mesh.coordinates()),dtype=np.float32)[np.newaxis,...],num_steps,axis=0)
    
    #From https://fenicsproject.org/qa/2989/vertex-on-mesh-boundary/
    def get_vertices_with_cond(mesh,condition):
        V = FunctionSpace(mesh, 'CG', 1)
        bc = DirichletBC(V, 1, condition)
        u = Function(V)
        bc.apply(u.vector())
        d2v = dof_to_vertex_map(V)
        vertices_on_boundary = d2v[u.vector() == 1.0]
        return vertices_on_boundary
    
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

    for k, v in sim_data.items():
        sim_data[k] = sim_data[k][(N_save-1)::N_save] #Skip first data points to avoid initial chaos phase

    sims_data.append(sim_data)
    
    #Remove temp files
    os.remove(tut+".h5")
    os.remove(tpt+".h5")
    
np.save(results_dir+"/simdata.npy",sims_data)


