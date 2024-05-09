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
from os.path import exists

from dgl.dataloading import GraphDataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import tri as mtri
import os
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from modulus.models.meshgraphnet import MeshGraphNet
from modulus.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
from modulus.launch.utils import load_checkpoint
from constants import Constants
from time import time
import math
from copy import deepcopy
import matplotlib
import json

"""
MGNRollout manages the inference loop for the MeshGraphNet model.

inter_sim: [None,int]:
 If int, then the model will be evaluated on the inter_sim-th simulation in the test_tiny dataset split.
 If None, then the model will be evaluated on the test dataset split.

animate_only: bool:
 If True, no model will be loaded and only the exact values will be animated.
"""


class MGNRollout:
    def __init__(self, C, inter_sim=None, animate_only=False):
        self.C = C
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # instantiate dataset
        self.dataset = VortexSheddingDataset(
            name="vortex_shedding_test",
            data_dir=C.data_dir,
            norm_data_dir=C.norm_data_dir,
            split="test_tiny" if inter_sim is not None else "test",
            start_step=C.first_step,
            start_sim=inter_sim if inter_sim is not None else C.test_start_sample,
            num_samples=C.num_test_samples if inter_sim is None else 1,
            num_steps=C.num_test_time_steps,
            verbose=C.verbose and inter_sim is None,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=1,  # TODO add support for batch_size > 1
            shuffle=False,
            drop_last=False,
        )

        if animate_only:
            self.model = None
        else:
            # instantiate the model
            self.model = MeshGraphNet(
                C.num_input_features, C.num_edge_features, C.num_output_features,
                hidden_dim_edge_processor=C.hidden_dim,
                hidden_dim_processor=C.hidden_dim,
                hidden_dim_node_encoder=C.hidden_dim,
                hidden_dim_node_decoder=C.hidden_dim,
                hidden_dim_edge_encoder=C.hidden_dim,
                multi_hop_edges=C.multi_hop_edges
            )
            if C.jit:
                self.model = torch.jit.script(self.model).to(self.device)
            else:
                self.model = self.model.to(self.device)

            # enable eval mode
            self.model.eval()

            # load checkpoint
            _ = load_checkpoint(
                os.path.join(C.ckpt_path, C.load_name),
                models=self.model,
                device=self.device,
                epoch=C.ckp,
                verbose=C.verbose and inter_sim is None,
            )

    """
        Rollout the model and calculate the RMSE for the velocity and pressure fields for 1, 50 and all steps in the loaded dataset. 

        inter_sim: If not None, then the results will also be logged to 'logs.txt'. Furthermore, in 'details.json' the results for each individual simulation will be stored.

        The results are returned as dictionary.
    """

    def predict(self, inter_sim=None):
        self.pred, self.exact, self.faces, self.graphs, self.pred_one_step = [], [], [], [], []
        stats = {
            key: value.to(self.device) for key, value in self.dataset.node_stats.items()
        }

        # Prepare list where the results for each individual simulation will be stored
        sims_results = []
        if exists(os.path.join(self.C.data_dir, "test_meshes_metadata.json")):
            with open(os.path.join(self.C.data_dir, "test_meshes_metadata.json"), 'r') as file:
                sims_results = json.load(file)
            sims_results = sims_results[self.C.test_start_sample:self.C.test_start_sample + self.C.num_test_samples]
            if self.C.verbose:
                print(f"Loaded mesh-metdata.")

        # Indices. 0: velocity, 1: pressure
        mse = torch.nn.MSELoss()
        mse_1_step_sum, mse_50_step_sum, mse_all_step_sum = np.zeros(2), np.zeros(2), np.zeros(
            2)  # Cumulative MSE of pressure + velocity for all simulations
        vse_last_sim_1s, pse_last_sim_1s, vse_last_sim_50s, pse_last_sim_50s, vse_last_sim_allstep, pse_last_sim_allstep = 0, 0, 0, 0, 0, 0  # squared-error for pressure and velocity for the last simulation only
        num_steps, num_50_steps = 0, 0
        t1step, t50step, tallstep = 0, 0, 0

        def add_sim_result(i, vse_1s, pse_1s, vse_50s, pse_50s, vse_all, pse_all):
            num = i // (self.C.num_test_time_steps - 1) - 1
            if num == len(sims_results):  # In this case, no mesh metadata available
                sims_results.append({})
            else:
                assert self.pred[i - 1].shape[0] == sims_results[num]["nodes"]

            sims_results[num]["v"] = {"1s": vse_1s / (self.C.num_test_time_steps - 1),
                                      "50s": vse_50s / min(50, (self.C.num_test_time_steps - 1)),
                                      "all": vse_all / (self.C.num_test_time_steps - 1)}
            sims_results[num]["p"] = {"1s": pse_1s / (self.C.num_test_time_steps - 1),
                                      "50s": pse_50s / min(50, (self.C.num_test_time_steps - 1)),
                                      "all": pse_all / (self.C.num_test_time_steps - 1)}

            if self.C.verbose:
                for key in {"v", "p"}:
                    value = sims_results[num][key]["all"]
                    print(f"Sim {num} (R)MSE of {key}: {value}, {math.sqrt(value)}")

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

            # Prepare data for inference step
            invar = graph.ndata["x"].clone()

            if i % (self.C.num_test_time_steps - 1) != 0:  # If = 0, then new graph starts
                invar[:, 0:2] = self.pred[i - 1][:, 0:2].clone()
                i += 1
            elif i > 0:
                if inter_sim is None:
                    add_sim_result(i, vse_last_sim_1s, pse_last_sim_1s, vse_last_sim_50s, pse_last_sim_50s,
                                   vse_last_sim_allstep, pse_last_sim_allstep)
                mse_1_step_sum += np.array([vse_last_sim_1s, pse_last_sim_1s])
                mse_50_step_sum += np.array([vse_last_sim_50s, pse_last_sim_50s])
                mse_all_step_sum += np.array([vse_last_sim_allstep, pse_last_sim_allstep])
                vse_last_sim_1s, pse_last_sim_1s, vse_last_sim_50s, pse_last_sim_50s, vse_last_sim_allstep, pse_last_sim_allstep = 0, 0, 0, 0, 0, 0

            invar[:, 0:2] = self.dataset.normalize_node(invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"])
            one_step_invar = graph.ndata["x"].clone()
            one_step_invar[:, 0:2] = self.dataset.normalize_node(one_step_invar[:, 0:2], stats["velocity_mean"],
                                                                 stats["velocity_std"])

            if self.model is not None:
                # Make prediction and track time
                start = time()
                pred_i = self.model(invar, graph.edata["x"], graph).detach()  # predict
                dt_allstep = time() - start
                start = time()
                pred_i_one_step = self.model(one_step_invar, graph.edata["x"], graph).detach()
                dt_1step = time() - start

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
                one_step_invar[:, 0:2] = self.dataset.denormalize(
                    one_step_invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
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
                        ((pred_i_one_step[:, 0:2] + one_step_invar[:, 0:2]), pred_i_one_step[:, [2]]), dim=-1
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

            if self.model is None:
                self.pred.append(self.exact[-1])
                self.pred_one_step.append(self.exact[-1])

            # Calculate MSE of current sim
            vse_last_sim_allstep += mse(self.pred[-1][:, 0:2], self.exact[-1][:, 0:2]).item()
            pse_last_sim_allstep += mse(self.pred[-1][:, 2], self.exact[-1][:, 2]).item()
            vse_last_sim_1s += mse(self.pred_one_step[-1][:, 0:2], self.exact[-1][:, 0:2]).item()
            pse_last_sim_1s += mse(self.pred_one_step[-1][:, 2], self.exact[-1][:, 2]).item()
            if i % self.C.num_test_time_steps < 50:
                vse_last_sim_50s += mse(self.pred[-1][:, 0:2], self.exact[-1][:, 0:2]).item()
                pse_last_sim_50s += mse(self.pred[-1][:, 2], self.exact[-1][:, 2]).item()
                t50step += dt_allstep if self.model is not None else 0
                num_50_steps += 1
            num_steps += 1
            tallstep += dt_allstep if self.model is not None else 0
            t1step += dt_1step if self.model is not None else 0

            self.faces.append(torch.squeeze(cells).numpy())
            self.graphs.append(graph.cpu())

        if inter_sim is None:
            add_sim_result(num_steps, vse_last_sim_1s, pse_last_sim_1s, vse_last_sim_50s, pse_last_sim_50s,
                           vse_last_sim_allstep, pse_last_sim_allstep)

        mse_1_step_sum += np.array([vse_last_sim_1s, pse_last_sim_1s])
        mse_50_step_sum += np.array([vse_last_sim_50s, pse_last_sim_50s])
        mse_all_step_sum += np.array([vse_last_sim_allstep, pse_last_sim_allstep])

        # Take average and sqrt
        rmse_1_step = np.sqrt(mse_1_step_sum / num_steps)
        rmse_50_step = np.sqrt(mse_50_step_sum / num_50_steps)
        rmse_all_step = np.sqrt(mse_all_step_sum / num_steps)
        t1step /= num_steps
        t50step /= num_50_steps
        tallstep /= num_steps

        result_dict = {
            f"Model {self.C.load_name} checkpoint": self.C.ckp,
            "RMSE (velo) 1 step": rmse_1_step[0],
            "RMSE (velo) 50 step": rmse_50_step[0],
            "RMSE (velo) all step": rmse_all_step[0],
            "RMSE (pressure) 1 step": rmse_1_step[1],
            "RMSE (pressure) 50 step": rmse_50_step[1],
            "RMSE (pressure) all step": rmse_all_step[1],
            "Avg time 1 step:": t1step,
            "Avg time 50 step": t50step,
            "Avg time all steps": tallstep
        }

        # Print and log results
        if inter_sim is None:
            path = os.path.join(self.C.ckpt_path, self.C.save_name)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            with open(os.path.join(self.C.ckpt_path, self.C.save_name, "log.txt"), 'a') as file:
                for key, value in result_dict.items():
                    out_str = f"{key}: {value}"
                    file.write(out_str + "\n")
                    if self.C.verbose:
                        print(out_str, flush=True)

            details_name = "details.json"
            if self.C.verbose:
                print(f"Saving individual simulation results to {details_name}")
            with open(os.path.join(self.C.ckpt_path, self.C.save_name, details_name), 'w') as file:
                json.dump(sims_results, file)

        else:
            if self.C.verbose:
                print(
                    f"Inter eval sim {inter_sim}: 1step {rmse_1_step[0]}, 50step {rmse_50_step[0]}, allstep {rmse_all_step[0]}")

        return result_dict

    def init_animation(self, viz_var, start, end):
        pred = self.pred[start:end]
        exact = self.exact[start:end]
        if viz_var == "vx":
            self.pred_i = [var[:, 0] for var in pred]
            self.exact_i = [var[:, 0] for var in exact]
        elif viz_var == "vy":
            self.pred_i = [var[:, 1] for var in pred]
            self.exact_i = [var[:, 1] for var in exact]
        elif viz_var == "p":
            self.pred_i = [var[:, 2] for var in pred]
            self.exact_i = [var[:, 2] for var in exact]
        elif viz_var == "v":
            self.pred_i = [torch.sqrt(var[:, 0] ** 2 + var[:, 1] ** 2) for var in pred]
            self.exact_i = [torch.sqrt(var[:, 0] ** 2 + var[:, 1] ** 2) for var in exact]
        else:
            raise ValueError(f"Invalid viz_var {viz_var}")

        self.graphs_i = self.graphs[start:end]
        self.faces_i = self.faces[start:end]

        # fig configs
        plt.rcParams["image.cmap"] = "inferno"
        self.fig, self.ax = plt.subplots(3, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1, 0.15]})

        # Set background color to white
        self.fig.set_facecolor("white")
        self.ax[0].set_facecolor("white")
        self.ax[1].set_facecolor("white")
        self.ax[2].set_facecolor("white")

    def animate(self, num):

        num *= self.C.frame_skip
        graph = self.graphs_i[num]
        y_star = self.pred_i[num].numpy()
        y_exact = self.exact_i[num].numpy()
        triang = mtri.Triangulation(
            graph.ndata["mesh_pos"][:, 0].numpy(),
            graph.ndata["mesh_pos"][:, 1].numpy(),
            self.faces_i[num],
        )

        vmin = min([np.min(x.numpy()) for x in self.pred_i] + [np.min(x.numpy()) for x in self.exact_i])
        vmax = max([np.max(x.numpy()) for x in self.pred_i] + [np.max(x.numpy()) for x in self.exact_i])
        # vmin = 0.0
        # vmax = 2.0

        # Add prediction plot
        self.ax[0].cla()
        self.ax[0].set_aspect("equal")
        self.ax[0].set_axis_off()
        navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
        self.ax[0].add_patch(navy_box)  # Add a navy box to the first subplot
        self.ax[0].tripcolor(triang, y_star, vmin=vmin, vmax=vmax)
        self.ax[0].triplot(triang, "ko-", ms=0.5, lw=0.3)
        self.ax[0].set_title("Modulus MeshGraphNet Prediction", color="white")

        # Add ground truth plot
        self.ax[1].cla()
        self.ax[1].set_aspect("equal")
        self.ax[1].set_axis_off()
        navy_box = Rectangle((0, 0), 1.4, 0.4, facecolor="navy")
        self.ax[1].add_patch(navy_box)  # Add a navy box to the second subplot
        self.ax[1].tripcolor(triang, y_exact, vmin=vmin, vmax=vmax)
        self.ax[1].triplot(triang, "ko-", ms=0.5, lw=0.3)
        self.ax[1].set_title("Ground Truth", color="white")

        # Add colorbar
        self.ax[2].cla()
        self.ax[2].set_aspect("equal")
        self.ax[2].set_axis_off()
        sm = plt.cm.ScalarMappable(cmap=plt.rcParams["image.cmap"])
        sm.set_array([])
        divider = make_axes_locatable(self.ax[-1])
        cbar_ax = divider.append_axes("top", size="50%", pad=-0.5)  # Position the colorbar below the last subplot
        self.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        sm.set_clim(vmin, vmax)
        cbar_ax.tick_params(labelsize=40)

        # Adjust subplots to minimize empty space
        self.ax[0].set_aspect("auto", adjustable="box")
        self.ax[1].set_aspect("auto", adjustable="box")
        self.ax[2].set_aspect("auto", adjustable="box")
        self.ax[0].autoscale(enable=True, tight=True)
        self.ax[1].autoscale(enable=True, tight=True)
        self.ax[2].autoscale(enable=True, tight=True)
        self.fig.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.2
        )
        return self.fig


def animate_rollout(rollout, C: Constants):
    path = f"animations/{C.save_name.split('.')[0]}"
    os.makedirs(path, exist_ok=True)
    for viz_var in C.viz_vars:
        var_path = f"{path}/{viz_var}"
        os.makedirs(var_path, exist_ok=True)
        for j in range(C.num_test_samples):
            #Png
            rollout.init_animation(viz_var=viz_var, start=(j + 1) * (C.num_test_time_steps - 1)-C.frame_skip,
                                   end=(j + 1) * (C.num_test_time_steps - 1) - C.frame_skip + 1)
            ani = animation.FuncAnimation(
                rollout.fig,
                rollout.animate,
                frames=1,
                interval=1,
            )
            ani_path = f"{var_path}/sim{j + rollout.C.test_start_sample}.png"
            ani.save(ani_path)

            #Gif
            rollout.init_animation(viz_var=viz_var, start=j * (C.num_test_time_steps - 1),
                                   end=(j + 1) * (C.num_test_time_steps - 1))
            ani = animation.FuncAnimation(
                rollout.fig,
                rollout.animate,
                frames=(C.num_test_time_steps - 1) // C.frame_skip,
                interval=C.frame_interval,
            )
            ani_path = f"{var_path}/sim{j + rollout.C.test_start_sample}.gif"
            ani.save(ani_path)

            matplotlib.pyplot.close()


"""
    intermediate_eval: If True, then the model will only evaluated on each simulation in the test_tiny dataset split. 
                       If False, then the model will be evaluated on the entire test dataset split, the result will be 
                       logged into 'logs.txt'. Furthermore, the model's prediction is animated (if set in Constants C ) and saved into 'animations'.
"""


def evaluate_model(C: Constants, intermediate_eval: bool = False):
    if intermediate_eval:
        num_samples = VortexSheddingDataset(name="vortex_shedding_test", data_dir=C.data_dir, split="test_tiny",
                                            verbose=False).num_samples
        for i in range(num_samples):  # Get results for each individual simulation
            rollout = MGNRollout(C, inter_sim=i)
            rollout.predict(inter_sim=i)
    else:
        if C.verbose:
            print("Rollout started...", flush=True)

        # Evaluate model
        rollout = MGNRollout(C)
        res_dict = rollout.predict()

        # Animate model's predictions on all test graphs
        if C.animate:
            animate_rollout(rollout, C)

        return res_dict


def animate_dataset(dataset: str, C: Constants = Constants(), vars=("v",), ranges=[0]):
    for range in ranges:
        print("Range: ", range)
        start_sim = range[0] if isinstance(range, list) else range

        # Setup Config
        C = deepcopy(C)
        C.viz_vars = vars
        C.data_dir = "./raw_dataset/cylinder_flow/" + dataset
        C.save_name = dataset
        C.num_test_samples = (1 + range[1] - range[0]) if isinstance(range, list) else 1
        C.animate = True
        C.test_start_sample = start_sim
        # Rollout and animate simulations
        rollout = MGNRollout(C, animate_only=True)
        rollout.predict()
        animate_rollout(rollout, C)


#animate_dataset("2cylinders", ranges = [[10,10]], vars = ("v",))

"""
    Evaluate each given model group on each given dataset.

    model_groups: List of lists of model names. Can also be a tuple with dir from which the normalization data should be taken in second position.
                  The results for all models within one model group will be averaged.
    datasets: List of datasets to be evaluted.
"""


def pairwise_evaluation(model_groups: list[list[str] | tuple[list[str], str]], datasets: list[str],
                        C: Constants = Constants(), animate: bool = False):
    C = deepcopy(C)

    for dataset in datasets:
        C.data_dir = dataset
        # Evaluate each model within model group and average results
        for model_group in model_groups:
            if isinstance(model_group, tuple):
                model_group, C.norm_data_dir = model_group[0], model_group[1]
            else:
                C.norm_data_dir = None
            result_sum = {}
            for i, model in enumerate(model_group):
                C.animate = (i == 0) and animate
                C.load_name = model
                C.save_name = model + "_on_" + (dataset.split("/")[-1])
                res_dict = evaluate_model(C, intermediate_eval=False)
                for key, value in res_dict.items():
                    if key not in result_sum:
                        result_sum[key] = [value]
                    else:
                        result_sum[key] += [value]

            # Print results
            print(f"-------{C.load_name} --> {C.data_dir}----------")
            for key, value in result_sum.items():
                if not isinstance(value[0], (int, float)):
                    continue
                minv, maxx, avg = min(value), max(value), sum(value) / len(value)
                max_dist_avg = max(abs(minv - avg), abs(maxx - avg))
                print(f"{key} | Min:{min(value)} Max:{max(value)} Avg:{sum(value) / len(value)} Range:{max_dist_avg}")

# Non existent path => newly initialized model
data_paths = ["./raw_dataset/cylinder_flow/standard_cylinder", "./raw_dataset/cylinder_flow/2cylinders", "./raw_dataset/cylinder_flow/cylinder_stretch",
               "./raw_dataset/cylinder_flow/cylinder_tri_quad", "./raw_dataset/cylinder_flow/mixed_all"]
# fresh_models = ["fresh1","fresh2","fresh3"]
# standard_cylinder_model = (["standard_cylinder1","standard_cylinder2","standard_cylinder3"], "./raw_dataset/cylinder_flow/standard_cylinder")
# cylinder_stretch_model = (["stretch1","stretch2","stretch3"], "./raw_dataset/cylinder_flow/cylinder_stretch")
# cylinder_tri_quad_model = (["ctq1","ctq2","ctq3"], "./raw_dataset/cylinder_flow/cylinder_tri_quad")
# mixed_all_model = (["mixed1","mixed2","mixed3"], "./raw_dataset/cylinder_flow/mixed_all")
# cyl2_model = (["2cyl_1","2cyl_2","2cyl_3"], "./raw_dataset/cylinder_flow/2cylinders")
#
# pairwise_evaluation([["teest"],standard_cylinder_model,cylinder_stretch_model,cylinder_tri_quad_model,mixed_all_model,cyl2_model, fresh_models],["./raw_dataset/cylinder_flow/mixed_all"], animate=False)

# models = [["concsum3/checkpoints",{"agg": "concat_sum"}],["sum_01_3/checkpoints",{"agg": "sum", "weight":0.1}],["concat3/checkpoints",{"agg": "concat"}],["standard3/checkpoints",None]]
#
# for model in models:
#     C = Constants()
#     C.num_test_time_steps = 2
#     C.multi_hop_edges = model[1]
#     pairwise_evaluation([[model[0]]], ["./raw_dataset/cylinder_flow/deepmind"], animate=False, C= C)

models = ["standard_cylinder3", "stretch1","ctq1","2cyl_1","mixed1"]
anims = [(models[1],data_paths[2],6) , (models[0],data_paths[0],28), (models[0],data_paths[0],10), (models[0],data_paths[0],1),
         (models[0],data_paths[0],16), (models[0],data_paths[0],3), (models[1],data_paths[2],15), (models[2],data_paths[3],3), (models[2],data_paths[3],17),
         (models[3],data_paths[1],4), (models[4],data_paths[4],25), (models[0],data_paths[3],13), (models[0],data_paths[1],10), (models[0],data_paths[2],25),
         (models[2],data_paths[3],16),(models[0],data_paths[3],16),(models[0],data_paths[3],17),(models[0],data_paths[3],3),(models[0],data_paths[1],9),(models[0],data_paths[1],10)]

for model, data_path, sim in anims:
    C = Constants()
    C.test_start_sample = sim
    C.num_test_samples = 1
    pairwise_evaluation([[model]],[data_path], C=C,animate=True)
