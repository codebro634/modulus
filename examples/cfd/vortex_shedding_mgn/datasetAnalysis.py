import math
import numpy as np
import argparse
from modulus.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

"""
    Helper script to analyze a vortex shedding dataset at a specific time step.
    Furthermore, the script calculates the min/average/max inflow/circle center/radius of the entire dataset. 
    Assumes that there is exactly 1 circle-object in the dataset.
"""

TOL = 0.025

parser = argparse.ArgumentParser()
parser.add_argument("--graph_num", type=int, default=0, help="Number of simulation inside dataset to analyze.")
parser.add_argument("--time_step", type=int, default=0, help="Number of time step to analyze.")
parser.add_argument('--dataset', type=str, default="raw_dataset/cylinder_flow/cylinder_flow/", help='Path to dataset.')
parser.add_argument('--split', type=str, default="train", help='Which split of the dataset to analyze.')
args = parser.parse_args()

graph_num = args.graph_num
time_step = args.time_step
dataset = args.dataset
split = args.split

def average_object_stats():
    itr = VortexSheddingDataset.get_dataset_iterator(dataset, split)
    inflows, object_centers_x, object_centers_y, object_radii = [], [], [], []
    num = 0
    while True:
        try:
            simulation = itr.get_next()
        except Exception:
            break
        num+= 1
        simulation = {key: arr if isinstance(arr, np.ndarray) else arr.numpy() for key, arr in simulation.items()}
        
        tmp_inf, object_points_x, object_points_y = [], [], []
        for k in range(simulation['mesh_pos'].shape[1]):
            if simulation['mesh_pos'][0, k, 0].item() < TOL:
                tmp_inf.append(simulation['velocity'][0, k, 0].item())
            if simulation['node_type'][0, k].item() == 6 and simulation['mesh_pos'][0, k, 1].item() > TOL and simulation['mesh_pos'][0, k, 1].item() < height - TOL:
                object_points_x.append(simulation['mesh_pos'][0, k, 0])
                object_points_y.append(simulation['mesh_pos'][0, k, 1])

        inflows.append(np.max(tmp_inf))
        object_centers_x.append(np.mean(object_points_x))
        object_centers_y.append(np.mean(object_points_y))
        object_radii.append(object_centers_x[-1] - np.min(object_points_x))

    print(f"---------- Dataset average of {num} graphs ----------")
    print(f"Inflow mean: {np.mean(inflows)}, std: {np.std(inflows)}, max: {np.max(inflows)}, min: {np.min(inflows)}")
    print(f"Object center x mean: {np.mean(object_centers_x)}, std: {np.std(object_centers_x)}, max: {np.max(object_centers_x)}, min: {np.min(object_centers_x)}")
    print(f"Object center y mean: {np.mean(object_centers_y)}, std: {np.std(object_centers_y)}, max: {np.max(object_centers_y)}, min: {np.min(object_centers_y)}")
    print(f"Object radius mean: {np.mean(object_radii)}, std: {np.std(object_radii)}, max: {np.max(object_radii)}, min: {np.min(object_radii)}")

itr = VortexSheddingDataset.get_dataset_iterator(dataset, split)
for _ in range(graph_num+1):
    simulation = itr.get_next()
simulation = {key: arr if isinstance(arr, np.ndarray) else arr.numpy() for key, arr in simulation.items()}


n = simulation['mesh_pos'].shape[1]
width = round(np.max(simulation['mesh_pos'][0, :, 0]).item(),3)
height = round(np.max(simulation['mesh_pos'][0, :, 1]).item(),3)

print(f"---------- Graph {graph_num} at t={time_step}----------")
print(f"Mesh width: {width}, height: {height}")
#Object nodes
print(f"Object nodes stats")
type_4_nodes, type_5_nodes, wall_nodes, object_nodes = [], [], [], []
max_inflow = -math.inf
object_x, object_y = [], []
for k in range(n):
    x, y = simulation['mesh_pos'][time_step, k]
    ntype = simulation['node_type'][time_step, k]
    datapoint = ((x,y),simulation['velocity'][time_step, k,0], simulation['pressure'][time_step, k,0])
    if ntype == 6 and y > TOL and y < height - TOL:
        object_nodes.append(datapoint)
        object_x.append(x)
        object_y.append(y)
    if ntype == 6 and (y <= TOL or y >= height - TOL):
        wall_nodes.append(datapoint)
    if ntype == 4:
        type_4_nodes.append(datapoint)
        max_inflow = max(max_inflow, datapoint[1])
    if ntype == 5:
        type_5_nodes.append(datapoint)
print(f"Type 4 nodes: {len(type_4_nodes)}")
print(f"Type 5 nodes: {len(type_5_nodes)}")
print(f"Wall nodes: {len(wall_nodes)}")
print(f"Object nodes: {len(object_nodes)}")
print(f"Max inflow: {max_inflow}")
print(f"Max flow of entire sim: {np.sqrt(np.max(simulation['velocity'][:,:,0] * simulation['velocity'][:,:,0] + simulation['velocity'][:,:,1] * simulation['velocity'][:,:,1]))}")
print(f"Object middle ({np.mean(object_x)}, {np.mean(object_y)}), Diameter x: {np.max(object_x) - np.min(object_x)}, Diameter y: {np.max(object_y) - np.min(object_y)}")
for node in type_4_nodes:
    print(f"Inflow {node[0]} -> {node[1]} | {node[2]}")
for node in type_5_nodes:
    print(f"Outflow {node[0]} -> {node[1]} | {node[2]}")
for node in wall_nodes:
    print(f"Wall {node[0]} -> {node[1]} | {node[2]}")
for node in object_nodes:
    print(f"Object {node[0]} -> {node[1]} | {node[2]}")


#View of middle at the beginning
print(f"View of middle at the beginning")
dx = 0.025
for i in range(100):
    #Find node close to dx*i and height/2
    min_dist = math.inf
    min_node = None
    for k in range(n):
        x, y = simulation['mesh_pos'][time_step, k]
        dist = (x - dx*i)**2 + (y - height/2)**2
        if dist < min_dist:
            min_dist = dist
            min_node = k
    print(f"({simulation['mesh_pos'][time_step,min_node][0].item()},{simulation['mesh_pos'][time_step,min_node][1].item()}): {simulation['velocity'][time_step,min_node,0]} | {simulation['pressure'][time_step,min_node,0]}")

average_object_stats()