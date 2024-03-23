import os
import numpy as np

from modulus.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

"""
    Repeat a range of a dataset n times and save it to a new folder.
    
    Args:
        data_path: Path to the dataset to repeat.
        output_folder: Folder to save the repeated data.
        output_name: Name of the output file.
        n: Number of times to repeat the dataset.
        range_to_repeat: Range of the dataset to repeat. If None, the entire dataset is repeated.
"""
def repeat_simulation_data(data_path, output_folder, output_name, n, range_to_repeat=None):

    #Load dataset range
    if data_path.endswith('.tfrecord'):
        if data_path.endswith("test.tfrecord"):
            itr = VortexSheddingDataset.get_dataset_iterator(data_path.removesuffix("test.tfrecord"), "test")
        elif data_path.endswith("train.tfrecord"):
            itr = VortexSheddingDataset.get_dataset_iterator(data_path.removesuffix("train.tfrecord"), "train")
        else:
            raise ValueError("Invalid dataset name")
        assert range_to_repeat is not None
        dataset = []
        for i in range(range_to_repeat[1]):
            simulation = itr.get_next()
            if i >= range_to_repeat[0]:
                simulation = {key: arr.numpy() for key, arr in simulation.items()}
                dataset.append(simulation)
    else:
        dataset = np.load(data_path, allow_pickle=True)
        dataset = dataset if range_to_repeat is None else dataset[range_to_repeat[0]:range_to_repeat[1]]

    #Concatenate dataset n times
    repeated_dataset = []
    for i in range(n):
        repeated_dataset += [simulation.copy() for simulation in dataset]

    #Save
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, output_name), repeated_dataset)


"""
    Iterate all subdirectories of a list of folders and merge all .npy files into a single .npy file.
    
    Args:
        folders: List of folder paths to merge.
        output_folder: Folder to save the merged data.
"""
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
        merged_sims = np.concatenate((merged_sims, np.load(path, allow_pickle=True)))

    #Save merged data
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, "merged.npy"), merged_sims)

#out = "../examples/cfd/vortex_shedding_mgn/raw_dataset/cylinder_flow/merged/"
#folders = ["../examples/cfd/vortex_shedding_mgn/raw_dataset/cylinder_flow/standard_cylinder_test/"]
#merge_simulation_data(folders, out)

#parser = argparse.ArgumentParser()
#parser.add_argument("--out",  help="Output folder.")
#parser.add_argument("--folders", help="Folders to merge. Comma separated.")
#args = parser.parse_args()

#merge_simulation_data(args.folders.split(','), args.out)

#repeat_simulation_data("../examples/cfd/vortex_shedding_mgn/raw_dataset/cylinder_flow/cylinder_flow/test.tfrecord", "../examples/cfd/vortex_shedding_mgn/raw_dataset/cylinder_flow/repeated73/", "train.npy", 20, range_to_repeat=[73,74])
