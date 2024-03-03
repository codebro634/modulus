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


class MGNTrainer:

    def __init__(self, C: Constants):
        self.C = C

        # instantiate dataset
        dataset = VortexSheddingDataset(
            name="vortex_shedding_train",
            data_dir=C.data_dir,
            split="train",
            start_step = C.first_step,
            num_samples=C.num_training_samples,
            num_steps=C.num_training_time_steps,
        )

        # instantiate dataloader
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
        print("Instantiating model...", flush=True)
        self.model = MeshGraphNet(
            C.num_input_features, C.num_edge_features, C.num_output_features, hidden_dim_edge_processor=C.hidden_dim_edge_processor,
            hidden_dim_processor=C.hidden_dim_edge_processor,
            hidden_dim_node_encoder=C.hidden_dim_edge_processor,
            hidden_dim_node_decoder=C.hidden_dim_edge_processor,
            hidden_dim_edge_encoder=C.hidden_dim_edge_processor,
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
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=C.device,
            epoch=C.ckp,
        )

        print("Finished MGN Trainer init", flush=True)

    def train(self, graph):
        graph = graph.to(self.C.device)
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
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


#By ChatGPT
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

    # Print used and available memory
    print(f"Used Memory: {convert_bytes(memory_info.used)}")
    print(f"Available Memory: {convert_bytes(memory_info.available)}")
    GPUtil.showUtilization()

def train(C: Constants):

    # save constants to JSON file
    os.makedirs(C.ckpt_path, exist_ok=True)
    with open(
        os.path.join(C.ckpt_path, C.ckpt_name.replace(".pt", ".json")), "w"
    ) as json_file:
        json.dump(C.__dict__, json_file, indent=4)

    trainer = MGNTrainer(C)

    start = time.time()
    print("Start training", flush=True)
    for epoch in range(trainer.epoch_init, C.epochs):

        for i, graph in enumerate(trainer.dataloader):
            loss = trainer.train(graph)
            if i < 10 or (i % 10 == 0 and i < 100) or (i % 100 == 0 and i < 1000) or (i % 1000 == 0 and i < 10000) or i % 10000 == 0:
                print(f"Epoch {epoch} | Graphs processed:{i}", flush=True)

        log_string = f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}"

        with open(os.path.join(C.ckpt_path, C.ckpt_name.replace(".pt", ".txt")), 'a') as file:
            file.write(log_string+ "\n")
        print(log_string, flush=True)

        save_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            epoch=epoch,
        )
        print(f"Saved model", flush=True)

        if C.inter_eval:
            evaluate_model(C=C, intermediate_eval=True)

        start = time.time()
    print("Training finished", flush=True)