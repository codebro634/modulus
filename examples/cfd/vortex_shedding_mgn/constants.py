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

from pydantic import BaseModel
from typing import Tuple
import torch


class Constants(BaseModel):
    """vortex shedding constants"""

    # data configs
    data_dir: str = "./raw_dataset/cylinder_flow/cylinder_flow"
    norm_data_dir: str = None #If set, take the edge/node normalization from this dir for inference
    load_name: str = "model"
    save_name: str = "model"
    ckp: int = None #None loads the latest checkpoint, -1 initializes a new model
    ckpt_path: str = "checkpoints"

    # training configs
    batch_size: int = 1
    epochs: int = 25
    first_step: int = 0  # start time step of each simulation
    num_training_samples: int = 400
    num_training_time_steps: int = 300
    lr: float = 0.0001
    lr_decay_rate: float = 0.82540418526
    fresh_optim: bool = False
    hidden_dim: int = 128
    num_input_features: int = 6
    num_output_features: int = 3
    num_edge_features: int = 3
    multi_hop_edges: dict = None # Possible keys: agg in {sum, concat, concat_sum} and weight of agg=sum

    # performance configs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = False
    jit: bool = False

    # test & visualization configs
    animate: bool = True
    verbose: bool = True
    inter_eval: bool = False #Whether or not to do a mini-evaluation after each epoch
    num_test_samples: int = 40
    num_test_time_steps: int = 300
    viz_vars: Tuple[str, ...] = ("u", "v", "p")
    frame_skip: int = 10
    frame_interval: int = 1

