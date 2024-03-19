import os
import argparse
import torch
from constants import Constants
from examples.cfd.vortex_shedding_mgn.inference import evaluate_model
from examples.cfd.vortex_shedding_mgn.train import train

"""
Recursively search all subdirectories from the dirctory the script has been called
from and change the working directory to the subdirectory containing 'raw dataset'
"""
#All produly written by ChatGPT!
def set_cwd(start_path='.'):
    # Check if the current directory contains 'raw_dataset'
    if 'raw_dataset' in os.listdir(start_path):
        os.chdir(start_path)  # Change to the directory containing 'raw_dataset'
        return True

    # Recursively search in subdirectories
    for subdir in os.listdir(start_path):
        subpath = os.path.join(start_path, subdir)
        if os.path.isdir(subpath):
            if set_cwd(subpath):
                return True

    return False  # 'raw_dataset' directory not found in any subdirectory


if __name__ == "__main__":
    if torch.cuda.is_available():
        x = torch.tensor([1,2,3],device='cuda:0') #Memory hack to grab some initial memory on the GPU to avoid CUDA out of memory

    #Change cwd
    set_cwd()

    C = Constants()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="./raw_dataset/cylinder_flow/cylinder_flow", help='Path to the dataset relative to the folder vortex_shedding_mgn.')
    parser.add_argument('--load_name', default="model", help='Name of the model to load/init.')
    parser.add_argument('--save_name', default="model", help='Name the model will be saved under.')
    parser.add_argument('--inter_eval', action='store_true', help='Does tiny intermediate evaluations after each epoch.')
    parser.add_argument('--ckp', type=int, help='Number of checkpoint to load. If none is set, the latest is taken. -1 any checkpoints are ignored.')
    parser.add_argument('--epochs', type=int, default=C.epochs, help='Number of epochs for training.')
    parser.add_argument('--hidden', type=int, default=C.hidden_dim, help='Hidden dim size for edge/node processor/encoder/decoder.')
    parser.add_argument('--num_samples', type=int, default=C.num_training_samples, help='Number of different simulations used in training.')
    parser.add_argument('--num_time_steps', type=int, default=C.num_training_time_steps, help='Number of time steps per simulation in training.')
    parser.add_argument('--first_step', type=int, default=C.first_step, help='Simulation time step to start from.')
    parser.add_argument('--num_inf_samples', type=int, default=C.num_test_samples, help='Number of different simulations used for inference.')
    parser.add_argument('--num_inf_time_steps', type=int, default=C.num_test_time_steps, help='Number of time steps per simulation used for inference.')
    parser.add_argument('--multihop', default="none", help='Which multihop method to use. Choose from {none,sum,concat_sum,concat}.')
    parser.add_argument('--weight', type=float, default=0.5, help='The weight to be used for multihop if mode=sum is chosen.')
    parser.add_argument('--lr_decay', type=float, default=0.82540418526, help='Learning rate decay.')
    parser.add_argument('--train', action='store_true', help='')
    parser.add_argument('--fresh_optim', action='store_true', help='')
    parser.add_argument('--eval', action='store_true', help='')
    parser.add_argument('--verbose', action='store_true', help='')
    args = parser.parse_args()

    # Instantiate constants
    C.verbose = args.verbose
    C.num_test_time_steps = args.num_inf_time_steps
    C.num_test_samples = args.num_inf_samples
    C.num_training_time_steps = args.num_time_steps
    C.ckp = args.ckp
    C.lr_decay_rate = args.lr_decay
    C.first_step = args.first_step
    C.num_training_samples = args.num_samples
    C.epochs = args.epochs
    C.data_dir = args.data_dir
    C.save_name = args.save_name
    C.load_name = args.load_name
    C.data_dir = args.data_dir
    C.inter_eval = args.inter_eval
    C.hidden_dim = args.hidden
    C.fresh_optim = args.fresh_optim

    if args.multihop != "none":
        C.multi_hop_edges = {"agg": args.multihop, "weight": args.weight}
    else:
        C.multi_hop_edges = None

    # run training
    if args.train:
        train(C)

    # run evaluation
    if args.eval:
        evaluate_model(C)
