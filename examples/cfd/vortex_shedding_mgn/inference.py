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

import torch, dgl
from dgl.dataloading import GraphDataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import tri as mtri
import os
from matplotlib.patches import Rectangle

from modulus.models.meshgraphnet import MeshGraphNet
from modulus.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
from modulus.launch.utils import load_checkpoint
from constants import Constants

import wandb as wb


class MGNRollout:
    def __init__(self, C):
        #set contants
        self.C = C

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # instantiate dataset
        self.dataset = VortexSheddingDataset(
            name="vortex_shedding_test",
            data_dir=C.data_dir,
            split="test",
            num_samples=C.num_test_samples,
            num_steps=C.num_test_time_steps,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=1,  # TODO add support for batch_size > 1
            shuffle=False,
            drop_last=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            C.num_input_features, C.num_edge_features, C.num_output_features, hidden_dim_edge_processor=C.hidden_dim_edge_processor,multi_hop_edges=C.multi_hop_edges
        )
        if C.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # enable eval mode
        self.model.eval()

        # load checkpoint
        _ = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            device=self.device,
        )

        self.var_identifier = {"u": 0, "v": 1, "p": 2}

    def predict(self):
        self.pred, self.exact, self.faces, self.graphs, self.pred_one_step = [], [], [], [], []
        stats = {
            key: value.to(self.device) for key, value in self.dataset.node_stats.items()
        }

        """
            Index 0 veloctiy, 1 pressure + veloctiy
        """
        mse = torch.nn.MSELoss()
        mse_1_step, mse_50_step, mse_all_step = np.zeros(2), np.zeros(2), np.zeros(2)
        num_steps, num_50_steps = 0, 0

        for i, (graph, cells, mask) in enumerate(self.dataloader):
            graph = graph.to(self.device)
            # denormalize data
            graph.ndata["x"][:, 0:2] = self.dataset.denormalize(
                graph.ndata["x"][:, 0:2], stats["velocity_mean"], stats["velocity_std"]
            )
            graph.ndata["y"][:, 0:2] = self.dataset.denormalize(
                graph.ndata["y"][:, 0:2],
                stats["velocity_diff_mean"],
                stats["velocity_diff_std"],
            )
            graph.ndata["y"][:, [2]] = self.dataset.denormalize(
                graph.ndata["y"][:, [2]],
                stats["pressure_mean"],
                stats["pressure_std"],
            )

            # inference step
            invar = graph.ndata["x"].clone()

            if i % (self.C.num_test_time_steps - 1) != 0: #If = 0, then new graph starts
                invar[:, 0:2] = self.pred[i - 1][:, 0:2].clone()
                i += 1
            invar[:, 0:2] = self.dataset.normalize_node(
                invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
            )
            pred_i = self.model(invar, graph.edata["x"], graph).detach()  # predict
            pred_i_one_step = self.model(graph.ndata["x"].clone(), graph.edata["x"], graph).detach()

            # denormalize prediction
            pred_i[:, 0:2] = self.dataset.denormalize(
                pred_i[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"]
            )
            pred_i[:, 2] = self.dataset.denormalize(
                pred_i[:, 2], stats["pressure_mean"], stats["pressure_std"]
            )
            pred_i_one_step[:, 0:2] = self.dataset.denormalize(
                pred_i_one_step[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"]
            )
            pred_i_one_step[:, 2] = self.dataset.denormalize(
                pred_i_one_step[:, 2], stats["pressure_mean"], stats["pressure_std"]
            )

            invar[:, 0:2] = self.dataset.denormalize(
                invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
            )


            # do not update the "wall_boundary" & "outflow" nodes
            mask = torch.cat((mask, mask), dim=-1).to(self.device)
            pred_i[:, 0:2] = torch.where(
                mask, pred_i[:, 0:2], torch.zeros_like(pred_i[:, 0:2])
            )
            pred_i_one_step[:, 0:2] = torch.where(
                mask, pred_i_one_step[:, 0:2], torch.zeros_like(pred_i_one_step[:, 0:2])
            )

            # integration
            self.pred.append(
                torch.cat(
                    ((pred_i[:, 0:2] + invar[:, 0:2]), pred_i[:, [2]]), dim=-1
                ).cpu()
            )

            self.pred_one_step.append(
                torch.cat(
                    ((pred_i_one_step[:, 0:2] + invar[:, 0:2]), pred_i_one_step[:, [2]]), dim=-1
                ).cpu()
            )

            self.exact.append(
                torch.cat(
                    (
                        (graph.ndata["y"][:, 0:2] + graph.ndata["x"][:, 0:2]),
                        graph.ndata["y"][:, [2]],
                    ),
                    dim=-1,
                ).cpu()
            )

            #Loss calculation
            mse_all_step[0] += mse(self.pred[-1][:, 0:2], self.exact[-1][:, 0:2]).item() #Velocity prediction
            mse_all_step[1] += mse(self.pred[-1], self.exact[-1]).item() #Pressure + velocity prediction

            if i % self.C.num_test_time_steps < 50:
                mse_50_step[0] += mse(self.pred[-1][:, 0:2], self.exact[-1][:, 0:2]).item()
                mse_50_step[1] += mse(self.pred[-1], self.exact[-1]).item()
                num_50_steps += 1

            mse_1_step[0] += mse(self.pred_one_step[-1][:, 0:2], self.exact[-1][:, 0:2]).item()
            mse_1_step[1] += mse(self.pred_one_step[-1], self.exact[-1]).item()

            num_steps += 1

            self.faces.append(torch.squeeze(cells).numpy())
            self.graphs.append(graph.cpu())

        #Save results
        rmse_1_step = np.sqrt(mse_1_step / num_steps)
        rmse_50_step = np.sqrt(mse_50_step / num_50_steps)
        rmse_all_step = np.sqrt(mse_all_step / num_steps)

        result_dict = {
                    "RMSE (velo) 1 step": rmse_1_step[0],
                    "RMSE (velo) 50 step": rmse_50_step[0],
                    "RMSE (velo) all step": rmse_all_step[0],
                    "RMSE (velo+pressure) 1 step": rmse_1_step[1],
                    "RMSE (velo+pressure) 50 step": rmse_50_step[1],
                    "RMSE (velo+pressure) all step": rmse_all_step[1],
        }

        with open(os.path.join(self.C.ckpt_path, self.C.ckpt_name.replace(".pt", ".txt")), 'a') as file:
            for key, value in result_dict.items():
                out_str = f"{key}: {value}"
                file.write(out_str+"\n")
                print(out_str, flush=True)

        if self.C.wandb_tracking:
            wb.log(result_dict)

    def init_animation(self, idx):
        self.pred_i = [var[:, idx] for var in self.pred]
        self.exact_i = [var[:, idx] for var in self.exact]

        # fig configs
        plt.rcParams["image.cmap"] = "inferno"
        self.fig, self.ax = plt.subplots(2, 1, figsize=(16, 9))

        # Set background color to black
        self.fig.set_facecolor("black")
        self.ax[0].set_facecolor("black")
        self.ax[1].set_facecolor("black")

        # make animations dir
        if not os.path.exists("./animations"):
            os.makedirs("./animations")

    def animate(self, num):
        num *= self.C.frame_skip
        graph = self.graphs[num]
        y_star = self.pred_i[num].numpy()
        y_exact = self.exact_i[num].numpy()
        triang = mtri.Triangulation(
            graph.ndata["mesh_pos"][:, 0].numpy(),
            graph.ndata["mesh_pos"][:, 1].numpy(),
            self.faces[num],
        )
        self.ax[0].cla()
        self.ax[0].set_aspect("equal")
        self.ax[0].set_axis_off()
        navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
        self.ax[0].add_patch(navy_box)  # Add a navy box to the first subplot
        self.ax[0].tripcolor(triang, y_star, vmin=np.min(y_star), vmax=np.max(y_star))
        self.ax[0].triplot(triang, "ko-", ms=0.5, lw=0.3)
        self.ax[0].set_title("Modulus MeshGraphNet Prediction", color="white")
        self.ax[1].cla()
        self.ax[1].set_aspect("equal")
        self.ax[1].set_axis_off()
        navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
        self.ax[1].add_patch(navy_box)  # Add a navy box to the second subplot
        self.ax[1].tripcolor(
            triang, y_exact, vmin=np.min(y_exact), vmax=np.max(y_exact)
        )
        self.ax[1].triplot(triang, "ko-", ms=0.5, lw=0.3)
        self.ax[1].set_title("Ground Truth", color="white")

        # Adjust subplots to minimize empty space
        self.ax[0].set_aspect("auto", adjustable="box")
        self.ax[1].set_aspect("auto", adjustable="box")
        self.ax[0].autoscale(enable=True, tight=True)
        self.ax[1].autoscale(enable=True, tight=True)
        self.fig.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.2
        )
        return self.fig


def evaluate_model(C: Constants):
    print("Rollout started...", flush=True)
    rollout = MGNRollout(C)
    idx = [rollout.var_identifier[k] for k in C.viz_vars]
    rollout.predict()
    for i in idx:
        rollout.init_animation(i)
        ani = animation.FuncAnimation(
            rollout.fig,
            rollout.animate,
            frames=len(rollout.graphs) // C.frame_skip,
            interval=C.frame_interval,
        )
        ani.save(f"animations/{C.ckpt_name.split('.')[0]}_animation_" + C.viz_vars[i] + ".gif")
