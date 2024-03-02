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


import functools
import json
import os

import numpy as np
import torch

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the Tensorflow library. Install the "
        + "package at: https://www.tensorflow.org/install"
    )

try:
    import dgl
    from dgl.data import DGLDataset
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the DGL library. Install the "
        + "desired CUDA version at: https://www.dgl.ai/pages/start.html"
    )
from torch.nn import functional as F

from .utils import load_json, save_json

# Hide GPU from visible devices for TF
tf.config.set_visible_devices([], "GPU")


class VortexSheddingDataset(DGLDataset):
    """In-memory MeshGraphNet Dataset for stationary mesh
    Notes:
        - This dataset prepares and processes the data available in MeshGraphNet's repo:
            https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
        - A single adj matrix is used for each transient simulation.
            Do not use with adaptive mesh or remeshing

    Parameters
    ----------
    name : str, optional
        Name of the dataset, by default "dataset"
    data_dir : _type_, optional
        Specifying the directory that stores the raw data in .TFRecord format., by default None
    split : str, optional
        Dataset split ["train", "eval", "test"], by default "train"
    num_samples : int, optional
        Number of samples, by default 1000
    num_steps : int, optional
        Number of time steps in each sample, by default 600
    noise_std : float, optional
        The standard deviation of the noise added to the "train" split, by default 0.02
    force_reload : bool, optional
        force reload, by default False
    verbose : bool, optional
        verbose, by default False
    """

    def __init__(
        self,
        name="dataset",
        data_dir=None,
        split="train",
        num_samples=None, #If none, the entire dataset is taken
        start_step=0,
        start_sim=0,
        num_steps=None, #If none, the entire simulation is taken
        noise_std=0.02,
        force_reload=False,
        verbose=True,
    ):
        super().__init__(
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )

        self.data_dir = data_dir
        self.split = split
        self.start_step = start_step
        if num_samples is None:
            dataset_iterator = VortexSheddingDataset.get_dataset_iterator(data_dir, split,start_sim=start_sim)
            num_samples = 0
            while True:
                try:
                    x = dataset_iterator.get_next()
                    max_steps = x["mesh_pos"].shape[0]
                    num_samples += 1
                except Exception:
                    break
        num_samples = num_samples - start_sim
        self.num_samples = num_samples
        num_steps = num_steps if num_steps is not None else max_steps - start_step
        self.num_steps = num_steps
        self.noise_std = noise_std
        self.length = self.num_samples * (self.num_steps - 1)

        if verbose:
            print(f"Preparing the {split} dataset of {data_dir} ...", flush=True)
        # create the graphs with edge features

        dataset_iterator = VortexSheddingDataset.get_dataset_iterator(data_dir, split)
        self.graphs, self.cells, self.node_type = [], [], []
        noise_mask, self.rollout_mask = [], []
        self.mesh_pos = []

        for i in range(self.num_samples):

            if i % 20 == 0 and verbose:
                print(f"Loaded {i} / {self.num_samples} samples...", flush=True)
            data_np = dataset_iterator.get_next()
            data_np = {key: arr[start_step:num_steps+start_step] if isinstance(arr, np.ndarray) else arr[start_step:num_steps+start_step].numpy() for key, arr in data_np.items()}

            src, dst = self.cell_to_adj(data_np["cells"][0])  # assuming stationary mesh
            graph = self.create_graph(src, dst, dtype=torch.int32)
            graph = self.add_edge_features(graph, data_np["mesh_pos"][0])
            self.graphs.append(graph)
            node_type = torch.tensor(data_np["node_type"][0], dtype=torch.uint8)
            self.node_type.append(self._one_hot_encode(node_type))
            noise_mask.append(torch.eq(node_type, torch.zeros_like(node_type)))

            if self.split != "train":
                self.mesh_pos.append(torch.tensor(data_np["mesh_pos"][0]))
                self.cells.append(data_np["cells"][0])
                self.rollout_mask.append(self._get_rollout_mask(node_type))
        if verbose:
            print("Computing the edge stats...", flush=True)

        # compute or load edge data stats
        if self.split == "train":
            self.edge_stats = self._get_edge_stats()
        else:
            if not os.path.exists(f"{self.data_dir}/edge_stats.json"):
                if verbose:
                    print("Warning: edge_state.json not found. Therefore computing those with the test data set.")
                self.edge_stats = self._get_edge_stats()
            else:
                self.edge_stats = load_json(f"{self.data_dir}/edge_stats.json")

        # normalize edge features
        for i in range(num_samples):
            self.graphs[i].edata["x"] = self.normalize_edge(
                self.graphs[i],
                self.edge_stats["edge_mean"],
                self.edge_stats["edge_std"],
            )

        # create the node features
        if verbose:
            print("Computing the node features...", flush=True)

        dataset_iterator = VortexSheddingDataset.get_dataset_iterator(data_dir,split, start_sim=start_sim)
        self.node_features, self.node_targets = [], []
        for i in range(self.num_samples):
            data_np = dataset_iterator.get_next()
            data_np = {key: arr[start_step:num_steps+start_step] if isinstance(arr, np.ndarray) else arr[start_step:num_steps+start_step].numpy() for key, arr in data_np.items()}
            features, targets = {}, {}
            features["velocity"] = self._drop_last(data_np["velocity"])
            targets["velocity"] = self._push_forward_diff(data_np["velocity"])
            targets["pressure"] = self._push_forward(data_np["pressure"])

            # add noise
            if split == "train":
                features["velocity"], targets["velocity"] = self._add_noise(
                    features["velocity"],
                    targets["velocity"],
                    self.noise_std,
                    noise_mask[i],
                )
            self.node_features.append(features)
            self.node_targets.append(targets)

        # compute or load node data stats
        if self.split == "train":
            self.node_stats = self._get_node_stats()
        else:
            if not os.path.exists(f"{self.data_dir}/node_stats.json"):
                if verbose:
                    print("Warning: node_stats.json not found. Therefore computing those with the test data set.")
                self.node_stats = self._get_node_stats()
            else:
                self.node_stats = load_json(f"{self.data_dir}/node_stats.json")

        # normalize node
        if verbose:
            print("Normalizing the node features...", flush=True)
        for i in range(num_samples):
            self.node_features[i]["velocity"] = self.normalize_node(
                self.node_features[i]["velocity"],
                self.node_stats["velocity_mean"],
                self.node_stats["velocity_std"],
            )
            self.node_targets[i]["velocity"] = self.normalize_node(
                self.node_targets[i]["velocity"],
                self.node_stats["velocity_diff_mean"],
                self.node_stats["velocity_diff_std"],
            )
            self.node_targets[i]["pressure"] = self.normalize_node(
                self.node_targets[i]["pressure"],
                self.node_stats["pressure_mean"],
                self.node_stats["pressure_std"],
            )

        # self.divisions = divisions
        # self.current_division = 0
        #
        # os.makedirs(self.data_dir + "/train_divs", exist_ok=True)
        # for div in range(divisions):
        #     current_division = {"graphs": [], "node_features": [], "node_targets_vel": [], "node_targets_pr": [], "node_type": [], "mesh_pos": [],"cells": [], "rollout_mask": []}
        #     l, u = self.get_ith_division_range(div)
        #     for idx in range(l,u):
        #         print(idx)
        #         gidx = idx // (self.num_steps - 1)  # graph index
        #         tidx = idx % (self.num_steps - 1)  # time step index
        #         #current_division["graphs"].append(self.graphs[gidx])
        #         #current_division["node_features"].append(self.node_features[0]["velocity"][0])
        #         #current_division["node_targets_vel"].append(self.node_targets[gidx]["velocity"][tidx])
        #         #current_division["node_targets_pr"].append(self.node_targets[gidx]["pressure"][tidx])
        #         #current_division["node_type"].append(self.node_type[gidx])
        #         #current_division["mesh_pos"].append(self.mesh_pos[gidx]) if self.split != "train" else None
        #         #current_division["cells"].append(self.cells[gidx]) if self.split != "train" else None
        #         #current_division["rollout_mask"].append(self.rollout_mask[gidx]) if self.split != "train" else None
        #
        #     #Save division
        #     #current_division["node_features"] = np.array(current_division["node_features"])
        #     current_division["node_features"] = np.repeat(self.node_features[0]["velocity"][0][:,np.newaxis],u-l, axis=1)
        #     np.save(self.data_dir + "/train_divs/" + f"division_{div}.npy", current_division)
        #
        #
        # self.node_features = None
        # self.node_targets = None
        # self.node_type = None
        # self.mesh_pos = None
        # self.cells = None
        # self.rollout_mask = None
        # self.graphs = None




    # def get_ith_division_range(self, i):
    #     return (i * self.length // self.divisions, (i + 1) * self.length // self.divisions)
    #
    # def save_as_divisions(self, divisions):
    #     pass
    #
    # def load_division(self, division):
    #     pass
    #
    # def load_full(self):
    #     pass
    #
    # def shuffle(self):
    #     #Shuffle all arrays with the same permutation with np function
    #     pass


    def __getitem__(self, idx):
        gidx = idx // (self.num_steps - 1)  # graph index
        tidx = idx % (self.num_steps - 1)  # time step index
        graph = self.graphs[gidx]
        node_features = torch.cat(
            (self.node_features[gidx]["velocity"][tidx], self.node_type[gidx]), dim=-1
        )
        node_targets = torch.cat(
            (
                self.node_targets[gidx]["velocity"][tidx],
                self.node_targets[gidx]["pressure"][tidx],
            ),
            dim=-1,
        )
        graph.ndata["x"] = node_features
        graph.ndata["y"] = node_targets
        if self.split == "train":
            return graph
        else:
            graph.ndata["mesh_pos"] = self.mesh_pos[gidx]
            cells = self.cells[gidx]
            rollout_mask = self.rollout_mask[gidx]
            return graph, cells, rollout_mask

    def __len__(self):
        return self.length

    def _get_edge_stats(self):
        stats = {
            "edge_mean": 0,
            "edge_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["edge_mean"] += (
                torch.mean(self.graphs[i].edata["x"], dim=0) / self.num_samples
            )
            stats["edge_meansqr"] += (
                torch.mean(torch.square(self.graphs[i].edata["x"]), dim=0)
                / self.num_samples
            )
        stats["edge_std"] = torch.sqrt(
            stats["edge_meansqr"] - torch.square(stats["edge_mean"])
        )
        stats.pop("edge_meansqr")

        # save to file
        save_json(stats, f"{self.data_dir}/edge_stats.json")
        return stats

    def _get_node_stats(self):
        stats = {
            "velocity_mean": 0,
            "velocity_meansqr": 0,
            "velocity_diff_mean": 0,
            "velocity_diff_meansqr": 0,
            "pressure_mean": 0,
            "pressure_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["velocity_mean"] += (
                torch.mean(self.node_features[i]["velocity"], dim=(0, 1))
                / self.num_samples
            )
            stats["velocity_meansqr"] += (
                torch.mean(torch.square(self.node_features[i]["velocity"]), dim=(0, 1))
                / self.num_samples
            )
            stats["pressure_mean"] += (
                torch.mean(self.node_targets[i]["pressure"], dim=(0, 1))
                / self.num_samples
            )
            stats["pressure_meansqr"] += (
                torch.mean(torch.square(self.node_targets[i]["pressure"]), dim=(0, 1))
                / self.num_samples
            )
            stats["velocity_diff_mean"] += (
                torch.mean(
                    self.node_targets[i]["velocity"],
                    dim=(0, 1),
                )
                / self.num_samples
            )
            stats["velocity_diff_meansqr"] += (
                torch.mean(
                    torch.square(self.node_targets[i]["velocity"]),
                    dim=(0, 1),
                )
                / self.num_samples
            )
        stats["velocity_std"] = torch.sqrt(
            stats["velocity_meansqr"] - torch.square(stats["velocity_mean"])
        )
        stats["pressure_std"] = torch.sqrt(
            stats["pressure_meansqr"] - torch.square(stats["pressure_mean"])
        )
        stats["velocity_diff_std"] = torch.sqrt(
            stats["velocity_diff_meansqr"] - torch.square(stats["velocity_diff_mean"])
        )
        stats.pop("velocity_meansqr")
        stats.pop("pressure_meansqr")
        stats.pop("velocity_diff_meansqr")

        # save to file
        save_json(stats, f"{self.data_dir}/node_stats.json")
        return stats

    @staticmethod
    def _load_tf_data(path, split):
        with open(os.path.join(path, "meta.json"), "r") as fp:
            meta = json.loads(fp.read())
        dataset = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
        dataset = dataset.map(
            functools.partial(VortexSheddingDataset._parse_data, meta=meta), num_parallel_calls=8
        ).prefetch(tf.data.AUTOTUNE)
        dataset_iterator = tf.data.make_one_shot_iterator(dataset)
        return dataset_iterator


    @staticmethod
    def cell_to_adj(cells):
        """creates adjancy matrix in COO format from mesh cells"""
        num_cells = np.shape(cells)[0]
        src = [cells[i][indx] for i in range(num_cells) for indx in [0, 1, 2]]
        dst = [cells[i][indx] for i in range(num_cells) for indx in [1, 2, 0]]
        return src, dst

    @staticmethod
    def create_graph(src, dst, dtype=torch.int32):
        """
        creates a DGL graph from an adj matrix in COO format.
        torch.int32 can handle graphs with up to 2**31-1 nodes or edges.
        """
        graph = dgl.to_bidirected(dgl.graph((src, dst), idtype=dtype))
        return graph

    @staticmethod
    def add_edge_features(graph, pos):
        """
        adds relative displacement & displacement norm as edge features
        """
        row, col = graph.edges()
        disp = torch.tensor(pos[row.long()] - pos[col.long()])
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=1)
        return graph

    @staticmethod
    def normalize_node(invar, mu, std):
        """normalizes a tensor"""
        if (invar.size()[-1] != mu.size()[-1]) or (invar.size()[-1] != std.size()[-1]):
            raise AssertionError("input and stats must have the same size")
        return (invar - mu.expand(invar.size())) / std.expand(invar.size())

    @staticmethod
    def normalize_edge(graph, mu, std):
        """normalizes a tensor"""
        if (
            graph.edata["x"].size()[-1] != mu.size()[-1]
            or graph.edata["x"].size()[-1] != std.size()[-1]
        ):
            raise AssertionError("Graph edge data must be same size as stats.")
        return (graph.edata["x"] - mu) / std

    @staticmethod
    def denormalize(invar, mu, std):
        """denormalizes a tensor"""
        denormalized_invar = invar * std + mu
        return denormalized_invar

    @staticmethod
    def _one_hot_encode(node_type):  # TODO generalize
        node_type = torch.squeeze(node_type, dim=-1)
        node_type = torch.where(
            node_type == 0,
            torch.zeros_like(node_type),
            node_type - 3,
        )
        node_type = F.one_hot(node_type.long(), num_classes=4)
        return node_type

    @staticmethod
    def _drop_last(invar):
        return torch.tensor(invar[0:-1], dtype=torch.float)

    @staticmethod
    def _push_forward(invar):
        return torch.tensor(invar[1:], dtype=torch.float)

    @staticmethod
    def _push_forward_diff(invar):
        return torch.tensor(invar[1:] - invar[0:-1], dtype=torch.float)

    @staticmethod
    def _get_rollout_mask(node_type):
        mask = torch.logical_or(
            torch.eq(node_type, torch.zeros_like(node_type)),
            torch.eq(
                node_type,
                torch.zeros_like(node_type) + 5,
            ),
        )
        return mask

    @staticmethod
    def _add_noise(features, targets, noise_std, noise_mask):
        noise = torch.normal(mean=0, std=noise_std, size=features.size())
        noise_mask = noise_mask.expand(features.size()[0], -1, 2)
        noise = torch.where(noise_mask, noise, torch.zeros_like(noise))
        features += noise
        targets -= noise
        return features, targets

    @staticmethod
    def _parse_data(p, meta):
        outvar = {}
        feature_dict = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
        features = tf.io.parse_single_example(p, feature_dict)
        for k, v in meta["features"].items():
            data = tf.reshape(
                tf.io.decode_raw(features[k].values, getattr(tf, v["dtype"])),
                v["shape"],
            )
            if v["type"] == "static":
                data = tf.tile(data, [meta["trajectory_length"], 1, 1])
            elif v["type"] == "dynamic_varlen":
                row_len = tf.reshape(
                    tf.io.decode_raw(features["length_" + k].values, tf.int32), [-1]
                )
                data = tf.RaggedTensor.from_row_lengths(data, row_lengths=row_len)
            outvar[k] = data
        return outvar

    @staticmethod
    def get_dataset_iterator(data_dir, split, start_sim = 0):
        if os.path.exists(data_dir + "/" + split + ".tfrecord"):
            itr = VortexSheddingDataset._load_tf_data(data_dir, split)
            [itr.get_next() for _ in range(start_sim)]
            return itr
        elif os.path.exists(data_dir + "/" + split + ".npy"):
            dataset = np.load(data_dir + "/" + split + ".npy", allow_pickle=True)

            class _dataset_iterator:
                def __init__(self, dataset):
                    self.dataset = dataset
                    self.i = 0

                def get_next(self):
                    self.i += 1
                    if self.i > len(self.dataset):
                        raise StopIteration
                    else:
                        return self.dataset[self.i - 1 + start_sim]

            return _dataset_iterator(dataset)
        else:
            raise FileNotFoundError("No dataset found in " + data_dir + " for " + split)