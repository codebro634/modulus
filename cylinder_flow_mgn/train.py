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
#
# File modified by Robin Schmöcker, Leibniz University Hannover, Germany, Copyright (c) 2024

import torch
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast, GradScaler
import time, os
import json
from inference import evaluate_model
import psutil

import GPUtil

try:
    import apex
except:
    pass

from modulus.models.meshgraphnet import MeshGraphNet
from modulus.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

from modulus.launch.utils import load_checkpoint, save_checkpoint
from constants import Constants

#MGN Trainer manages the training loop for the MeshGraphNet model
class MGNTrainer:

    def __init__(self, C: Constants):
        self.C = C

        # instantiate dataset
        dataset = VortexSheddingDataset(
            name="vortex_shedding_train",
            data_dir=C.data_dir,
            split="train",
            start_step=C.first_step,
            num_samples=C.num_training_samples,
            num_steps=C.num_training_time_steps,
            verbose=C.verbose
        )

        # instantiate dataloader
        if C.verbose:
            print("Creating dataloader...", flush=True)
        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=C.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
        )

        # instantiate the model
        if C.verbose:
            print("Instantiating model...", flush=True)
        self.model = MeshGraphNet(
            C.num_input_features, C.num_edge_features, C.num_output_features, hidden_dim_edge_processor=C.hidden_dim,
            hidden_dim_processor=C.hidden_dim,
            hidden_dim_node_encoder=C.hidden_dim,
            hidden_dim_node_decoder=C.hidden_dim,
            hidden_dim_edge_encoder=C.hidden_dim,
            multi_hop_edges=C.multi_hop_edges
        )
        if C.jit:
            self.model = torch.jit.script(self.model).to(C.device)
        else:
            self.model = self.model.to(C.device)

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
        self.epoch_init = load_checkpoint(
            os.path.join(C.ckpt_path, C.load_name),
            models=self.model,
            optimizer=self.optimizer if not C.fresh_optim else None,
            scheduler=self.scheduler if not C.fresh_optim else None,
            scaler=self.scaler if not C.fresh_optim else None,
            device=C.device,
            epoch=C.ckp,
            verbose=C.verbose
        )

        if C.verbose:
            print("Finished MGN Trainer init", flush=True)

    def train(self, graph):
        graph = graph.to(self.C.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(enabled=self.C.amp):
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            loss = self.criterion(pred, graph.ndata["y"])
            return loss

    def backward(self, loss):
        # backward pass
        if self.C.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()




#Proudly written by ChatGPT!
#Prints memory info for debugging purposes
def print_memory_info():
    # Get memory information
    memory_info = psutil.virtual_memory()

    # Convert bytes to a human-readable format
    def convert_bytes(bytes_size):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                break
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} {unit}"

    print(f"Used Memory: {convert_bytes(memory_info.used)}")
    print(f"Available Memory: {convert_bytes(memory_info.available)}")

    #Output GPU memory info
    GPUtil.showUtilization()

"""
 Starts the training process for the MeshGraphNet model
"""
def train(C: Constants):

    # save training constants to JSON file
    log_path = os.path.join(C.ckpt_path, C.save_name)
    os.makedirs(log_path, exist_ok=True)
    with open(
        os.path.join(log_path, "hyperparams.json"), "w"
    ) as json_file:
        json.dump(C.__dict__, json_file, indent=4)

    # Start training loop
    trainer = MGNTrainer(C)
    start = time.time()

    if C.verbose:
        print("Start training", flush=True)
    for epoch in range(trainer.epoch_init, (C.fresh_optim + trainer.epoch_init) if C.fresh_optim else C.epochs):
        for i, graph in enumerate(trainer.dataloader):
            loss = trainer.train(graph)
            if i < 10 or (i % 10 == 0 and i < 100) or (i % 100 == 0 and i < 1000) or (i % 1000 == 0 and i < 10000) or i % 10000 == 0:
                if C.verbose:
                    print(f"Epoch {epoch} | Graphs processed:{i}", flush=True)

        #Log epoch info
        log_string = f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}"
        with open(os.path.join(log_path, "log.txt"), 'a') as file:
            file.write(log_string+ "\n")
        if C.verbose:
            print(log_string, flush=True)

        trainer.scheduler.step()

        #Save current model
        save_checkpoint(
            os.path.join(log_path),
            models=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            epoch=epoch,
        )

        if C.verbose:
            print(f"Saved model", flush=True)

        #If necessary, evaluate the model at this stage
        if C.inter_eval:
            evaluate_model(C=C, intermediate_eval=True)

        start = time.time()

    if C.verbose:
        print("Training finished", flush=True)