import os
import argparse
from constants import Constants
from examples.cfd.vortex_shedding_mgn.inference import evaluate_model
from modulus.distributed.manager import DistributedManager
from examples.cfd.vortex_shedding_mgn.train import train

#By ChatGPT
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
    #Change cwd
    set_cwd()

    C = Constants()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="./raw_dataset/cylinder_flow/repeated73", help='Path to the dataset.')
    parser.add_argument('--exp_name', default="model", help='Name of the experiment.')
    parser.add_argument('--exp_group', default="multihop", help='Group of the experiment.')
    parser.add_argument('--inter_eval', action='store_true', help='Does tiny intermediate evaluations after each epoch.')
    parser.add_argument('--ckp', type=int, help='Number of checkpoint to load. If none is set, the latest is taken. -1 any checkpoints are ignored.')
    parser.add_argument('--epochs', type=int, default=C.epochs,help='Number of epochs for training.')
    parser.add_argument('--hidden', type=int, default=C.hidden_dim_edge_processor, help='Hidden dim width for edge processor.')
    parser.add_argument('--num_samples', type=int, default=C.num_training_samples, help='Number of different simulation used in training.')
    parser.add_argument('--num_time_steps', type=int, default=C.num_training_time_steps, help='Number of time steps per simulation.')
    parser.add_argument('--first_step', type=int, default=C.first_step, help='Simulation time step to start from.')
    parser.add_argument('--num_inf_samples', type=int, default=C.num_test_samples, help='Number of different simulation used for inference.')
    parser.add_argument('--num_inf_time_steps', type=int, default=C.num_test_time_steps, help='Number of time steps per simulation used for inference.')
    parser.add_argument('--multihop', default="none", help='Which multihop to use. Choose from {none,sum,sum_concat,concat}.')
    parser.add_argument('--weight', type=float, default=0.5, help='The weight to be used for multihop if mode=sum is chosen.')
    parser.add_argument('--wandb', action='store_true', help='Tracks experiment with wandb.')
    parser.add_argument('--train', action='store_true', help='Tracks experiment with wandb.')
    parser.add_argument('--eval', action='store_true', help='Tracks experiment with wandb.')
    args = parser.parse_args()
    # Instantiate constants

    C.num_test_time_steps = args.num_inf_time_steps
    C.num_test_samples = args.num_inf_samples
    C.num_training_time_steps = args.num_time_steps
    C.ckp = args.ckp
    C.first_step = args.first_step
    C.num_training_samples = args.num_samples
    C.epochs = args.epochs
    C.data_dir = args.data_dir
    C.ckpt_name = f"{args.exp_name}.pt"
    C.exp_name = args.exp_name
    C.exp_group = args.exp_group
    C.data_dir = args.data_dir
    C.inter_eval = args.inter_eval
    C.hidden_dim_edge_processor = args.hidden

    if args.multihop != "none":
        C.multi_hop_edges = {"agg": args.multihop, "weight": args.weight}
    else:
        C.multi_hop_edges = None

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # run training
    if args.train:
        train(C, dist)

    # run evaluation
    if dist.rank == 0 and args.eval:
        evaluate_model(C)
