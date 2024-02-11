import argparse
import os
import numpy as np

def merge_simulation_data(folders, output_folder):
    #Collect paths
    sim_paths = []
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.npy'):
                    sim_paths.append(os.path.join(root, file))
    #Load npy data
    merged_sims = []
    for path in sim_paths:
        merged_sims += np.load(path, mmap_mode='r')

    #Save merged data
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, "merged.npy"), merged_sims)


parser = argparse.ArgumentParser()
parser.add_argument("--out",  help="Output folder.")
parser.add_argument("--folders", help="Folders to merge. Comma separated.")
args = parser.parse_args()

merge_simulation_data(args.folders.split(','), args.out)

