[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11504394.svg)](https://doi.org/10.5281/zenodo.11504394)

# Purpose
Code for the Master's thesis: "Evaluation and Improvements to Mesh Graph Nets for Computational Fluid Dynamics Simulations."

# Acknowledgements
This repository is a fork of https://github.com/NVIDIA/modulus.git and has been pruned
and modified to fit the needs of the thesis. This is not an official NVIDIA repository and not the README of the original repository.
Every file that has been modified or added is marked with a comment at the top of the file.

# Run Spillover / Unmodified MGN

```
python cylinder_flow_mgn/main.py --load_name <str> --save_name <str> --data_dir <str> [--ckp <ckp> --epochs <int> --hidden <int> --num_samples <int> --num_time_steps <int> --first_step <int> --num_inf_samples <int> --num_inf_time_steps <int> --inf_start_sample <int> --multihop <str> --weight <float> --lr_decay <float> --fresh_scheduler --train --eval --verbose --inter_eval --animate]
```

- `load_name`: Name of the model to load. If this model is not saved under `checkpoints/` a fresh model is initialized.
- `save_name`: Under which name the model will be saved. The results/logs/animations will be saved under checkpoints/`save_name`.
- `data_dir`: Path (relative to cylinder_flow_mgn) to directory containing the dataset.
- `num_inf_samples`: Number of different simulations used for inference.
- `num_inf_time_steps`: Number of time steps per simulation used for inference.
- `num_samples`: Number of different simulations used in training.
- `num_time_steps`: Number of time steps per simulation in training.

### Optional parameters
- `ckp`: Number of checkpoint to load. If None is set, the latest is taken. If -1 or there are no saved checkpoints, a model will be newly initialized. Default: None.
- `epochs`: Number of training epochs. Default: 25.
- `hidden`: Hidden dim size for edge/node processor/encoder/decoder. Default: 128.
- `first_step`: Simulation time step to start from. Default: 0.
- `multihop`: Which Spillover method to use. Choose from {none, sum,concat_sum,concat}. Default: none.
- `weight`: The weight to be used for Spillover if mode=sum is chosen. Default: 0.5.
- `lr_decay`: Learning rate decay factor. Default: 0.82540418526. lr at epoch i = lr * lr_decay^(i-1).
- `train`: If set, the MGN is trained.
- `eval`: If set, the MGN is evaluated. If `train` is also set, then evaluation takes places after training. If neither `train` nor `eval` is set, nothing happens.
- `inf_start_sample`: Which simulation from the test dataset to start the evaluation from if `eval` has been set. Default: 0.
- `animate`: Whether to animate rollout predictions if `eval` has been set.
- `verbose`: If set, verbosity is activated.
- `inter_eval`: If set, a very small-scale evaluation is performed after every epoch on the test_tiny split.
- `fresh_optim`: If set, the learning rate scheduler and the optimizer is newly initialized and training will take place for `epochs` epochs even if a checkpoint is loaded.

### Animations
Note that by setting only the `eval` flag, it is possible to directly evaluate pretrained models. It is possible to simulate a single simulation from the test split, say the `n`-th simulation by setting `inf_start_sample` = `n` and `num_inf_samples` = `1`.

# Generate meshes

```
python data_generation/meshDatasetMaker.py --name <str> --num_meshes <int> [--width <float> --height <float> --ox <float> --oy <float> --osize <float> --inflow_peak_mean <float> --inflow_peak_max_deviation <float> --two_obj_prob <float> --rotate --stretch --circs --tris --quads]
```
- `name`: Name of the mesh dataset. The generated meshes are saved into meshes/`name`.
- `num_meshes`: Number of meshes to generate.

### Optional parameters
- `width`: Width of the channel. Default: 1.6.
- `height`: Height of the channel. Default: 0.41.
- `ox`: Mean of object's mid point's x-coordinate. Default: 0.325.
- `oy`: Mean of object's mid point's y-coordinate. Default: 0.2.
- `osize`: Mean object size (e.g. radius for circles). Default: 0.05.
- `inflow_peak_mean`:  Mean of the inflow peak. Default: 1.25.
- `inflow_peak_max_derivation`: Inflow peak is sampled from uniform[inflow_peak_mean-inflow_peak_max_derivation,inflow_peak_mean+inflow_peak_max_derivation]. Default: 1.0.
- `two_obj_prob`: A second object is randomly added with this probability. Default: 0.
- `rotate`: If set, the object(s) get randomly rotated.
- `stretch`: If set, the object(s) get randomly stretched/squeezed in the x/y-direction.
- `circs`: If set, circles are part of the possible objects that can be generated.
- `tris`: If set, triangles are part of the possible objects that can be generated.
- `quads`: If set, squares are part of the possible objects that can be generated.

# Simulate flow on a mesh

```
python data_generation/generator.py --dir <str> --mesh <str> [--t <float> --dt_sim <float> --dt_real <float> --mesh_range <tuple(int,int)> --vlevel <int> --cleanup_dir <str> --num_frames <int> --dont_save --qoi]
```
- `dir`: Name of the dir to save the simulation results into.
- `mesh`: Path to the mesh, the simulation is supposed to be run on. Can also be a directory. In that case, all meshes within all subfolders of that directory are simulated.

This method saves every 10 simulations into a .npy file. If one wants to merge all of these into a single .npy file, then use the provided `merge_simulation_data` method in `dataMerger.py`.

### Optional parameters
- `t`: Second till which the flow is simulated. Default: 3.0.
- `dt_real`: Delta t (step size) in the final dataset. Default: 0.01.
- `dt_sim`: Base Delta t (step size) that is used for the under-the-hood calculation. The step size value adapts dynamically with the inflow peak where `dt_sim` is used for an inflow peak of 1.25.  Default: 0.00025.
- `mesh_range`: If `mesh` is a directory, this sets the range of meshes to be used (ordering is the lexicographical order of the mesh paths). None, means all meshes are used. If not None, then `mesh_range` has the form 'a,b' where a,b are two integers that specify the lower (inclusive) and upper bound (exclusive) of the mesh range. Default: None.
- `vlevel`: Verbosity level. Min:0, Max:2. Default:1.
- `cleanup_dir`: If not None, then instead of using `mesh`, the directory  `cleanup_dir` and its subdirectories are searched for files named 'failed_meshes.txt'. These files are assumed to contain paths to meshes. All meshes found this way are used for simulation. Default: None.
- `num_frames`: If > 0, save animation of simulation as gif with num_frames frames. Default: 0.
- `dont_save`: If set, the simulation data is discarded and not saved (could be used if only the animation is needed).
- `qoi`: If set, calculate and save quantities of interest for the first simulation. Assumes that the mesh/inflow is that of DFG cylinder flow 2D-2 benchmark.

# Analyze dataset (Used for reverse-engineering parameters)

```
python cylinder_flow_mgn/datasetAnalysis.py --dataset <str> --split <str> [--graph_num <int> --time_step <int>]
```
- `dataset`: Path to the dataset to be analyzed.
- `split`: Which split inside the dataset is to be analyzed.

### Optional parameters
- `graph_num`: Number of the simulation inside the dataset to analyze. Default: None.
- `time_step`: Number of the time step within the `graph_num` to analyze. Default: None.

