# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
import time, os
import wandb as wb
import json
import argparse
from inference import evaluate_model

try:
    import apex
except:
    pass

from modulus.models.meshgraphnet import MeshGraphNet
from modulus.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
from modulus.distributed.manager import DistributedManager

from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from constants import Constants

# Instantiate constants
C = Constants()


class MGNTrainer:
    def __init__(self, wb, dist):
        self.dist = dist

        # instantiate dataset
        dataset = VortexSheddingDataset(
            name="vortex_shedding_train",
            data_dir=C.data_dir,
            split="train",
            num_samples=C.num_training_samples,
            num_steps=C.num_training_time_steps,
        )

        # instantiate dataloader
        print("Creating dataloader...")
        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=C.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )

        # instantiate the model
        print("Instantiating model...")
        self.model = MeshGraphNet(
            C.num_input_features, C.num_edge_features, C.num_output_features, multi_hop_edges=C.multi_hop_edges
        )
        if C.jit:
            self.model = torch.jit.script(self.model).to(dist.device)
        else:
            self.model = self.model.to(dist.device)
        if C.watch_model and not C.jit and dist.rank == 0:
            wb.watch(self.model)

        # distributed data parallel for multi-node training
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()
        try:
            self.optimizer = apex.optimizers.FusedAdam(self.model.parameters(), lr=C.lr)
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=C.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: C.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

        print("Finished MGN Trainer init")

    def train(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(enabled=C.amp):
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            loss = self.criterion(pred, graph.ndata["y"])
            return loss

    def backward(self, loss):
        # backward pass
        if C.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

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

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="./raw_dataset/cylinder_flow/cylinder_flow", help='Path to the dataset.')
    parser.add_argument('--exp_name', default="standard", help='Name of the experiment.')
    parser.add_argument('--exp_group', default="multihop", help='Group of the experiment.')
    parser.add_argument('--epochs', type=int, default=C.num_training_samples,help='Number of epochs for training.')
    parser.add_argument('--num_samples', type=int, default=C.num_training_samples, help='Number of different simulation used in training.')
    parser.add_argument('--num_time_steps', type=int, default=C.num_training_time_steps, help='Number of time steps per simulation.')
    parser.add_argument('--num_inf_samples', type=int, default=C.num_test_samples, help='Number of different simulation used for inference.')
    parser.add_argument('--num_inf_time_steps', type=int, default=C.num_test_time_steps, help='Number of time steps per simulation used for inference.')
    parser.add_argument('--multihop', default="none", help='Which multihop to use. Choose from {none,sum,sum_concat,concat}.')
    parser.add_argument('--weight', type=int, default=0.5, help='The weight to be used for multihop if mode=sum is chosen.')
    parser.add_argument('--wandb', action='store_true', help='Tracks experiment with wandb.')
    args = parser.parse_args()

    C.num_test_time_steps = args.num_inf_time_steps
    C.num_test_samples = args.num_inf_samples
    C.num_training_time_steps = args.num_time_steps
    C.num_training_samples = args.num_samples
    C.epochs = args.epochs
    C.data_dir = args.data_dir
    C.ckpt_name = f"{args.exp_name}.pt"
    C.data_dir = args.data_dir
    if args.wandb:
        C.wandb_tracking = True
        C.watch_model = True
        C.wandb_mode = "online"
    if args.multihop != "none":
        C.multi_hop_edges = {"agg": args.multihop, "weight": args.weight}

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # save constants to JSON file
    if dist.rank == 0:
        os.makedirs(C.ckpt_path, exist_ok=True)
        with open(
            os.path.join(C.ckpt_path, C.ckpt_name.replace(".pt", ".json")), "w"
        ) as json_file:
            json.dump({**C.__dict__,**{"worlds": dist.world_size}}, json_file, indent=4)

    # initialize loggers
    initialize_wandb(
        project="MGNs",
        entity="besteteam",
        name=args.exp_name,
        group=args.exp_group,
        mode=C.wandb_mode,
        config={**C.__dict__,**{"worlds": dist.world_size}},
    )  # Wandb logger

    trainer = MGNTrainer(wb, dist)
    start = time.time()
    print("Start training")
    for epoch in range(trainer.epoch_init, C.epochs):
        for i, graph in enumerate(trainer.dataloader):
            loss = trainer.train(graph)
            print(i)

        log_string = f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}"
        if C.wandb_tracking:
            wb.log({"loss": loss.detach().cpu()})
        with open(os.path.join(C.ckpt_path, C.ckpt_name.replace(".pt", ".txt")), 'a') as file:
            file.write(log_string+ "\n")
        print(log_string)

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                os.path.join(C.ckpt_path, C.ckpt_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            print(f"Saved model on rank {dist.rank}")
        start = time.time()
    print("Training finished")

    if dist.rank == 0:
        evaluate_model(C)
