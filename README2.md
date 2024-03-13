
# Purpose
Code for the Master's thesis: Evaluation and improvements to mesh graph nets
for computational fluid dynamics simulations

# Acknowledgements
MODULUS NVIDIA

# Run Spillover

```
python examples/cfd/vortex_sheddin_mgn/main.py --exp_name <str> --data_dir <str> [--ckp <ckp> --epochs <int> --hidden <int> --num_samples <int> --num_time_steps <int> --first_step <int> --num_inf_samples <int> --num_inf_time_steps <int> --multihop <str> --weight <float> --train --eval --verbose --inter_eval]
```

- `exp_name`: Name the experiment to be run. The results will be saved under checkpoints/`exp_name`
- `data_dir`: Path (relative to examples/cfd/vortex_shedding_mgn) to directory containing the dataset
- `num_inf_samples`: Number of different simulations used for inference.
- `num_inf_time_steps`: Number of time steps per simulation used for inference
- `num_samples`: Number of different simulations used in training
- `num_time_steps`: Number of time steps per simulation in training

### Optional parameters
- `ckp`: Number of checkpoint to load. If None is set, the latest is taken. If -1 or there are no saved checkpoints, a model will be newly initialized. Default: None
- `epochs`: Number of training epochs. Default: 25
- `hidden`: Hidden dim size for edge/node processor/encoder/decoder. Default: 128
- `first_step`: Simulation time step to start from. Default: 0
- `multihop`: Which Spillover method to use. Choose from {none, sum,concat_sum,concat}. Default: none
- `weight`: The weight to be used for Spillover if mode=sum is chosen. Default: 0.5
- `train`: If set, the MGN is trained
- `eval`: If set, the MGN is trained. If neither train nor eval is set, nothing happens
- `verbose`: If set, verbosity is activated
- `inter_eval`: If set, a very small-scale evaluation is performed after every epoch on the test_tiny split

# Generate meshes

```
python dataGeneration/meshDatasetMaker.py --name <str> --num_meshes <int> [--width <float> --height <float> --ox <float> --oy <float> --osize <float> --inflow_peak_mean <float> --inflow_peak_max_deviation <float> --two_obj --rotate --stretch --circs --tris --quads]
```
- `name`: Name of the mesh dataset. The generated meshes are saved into meshes/`name`
- `num_meshes`: Number of meshes to generate

### Optional parameters
- `width`: Width of the channel. Default: 1.6
- `height`: Height of the channel. Default: 0.41
- `ox`: Mean of object's mid point's x-coordinate. Default: 0.325
- `oy`: Mean of object's mid point's y-coordinate. Default: 0.2
- `osize`: Mean object size (e.g. radius for circles). Default: 0.05
- `inflow_peak_mean`:  Mean of the inflow peak. Default: 1.25
- `inflow_peak_max_derivation`: Inflow peak is sampled from uniform[inflow_peak_mean-inflow_peak_max_derivation,inflow_peak_mean+inflow_peak_max_derivation]. Default: 1.0
- `two_obj`: If set, a second object is randomly added
- `rotate`: If set, the object(s) get randomly rotated
- `stretch`: If set, the object(s) get randomly stretched/squeezed in the x/y-direction
- `circs`: If set, circles are part of the possible objects that can be generated
- `tris`: If set, triangles are part of the possible objects that can be generated
- `quads`: If set, squares are part of the possible objects that can be generated

# Simulate flow on a mesh

```
python dataGeneration/generator.py --dir <str> --mesh <str> [--steps <int> --dt <float> --saveN <int> --mesh_range <tuple(int,int)> --vlevel <int> --cleanup_dir <str> --p]
```
- `dir`: Name of the dir to save the simulation results into.
- `mesh`: Path to the mesh, the simulation is supposed to be run on. Can also be a directory. In that case, all meshes within all subfolders of that directory are simulated.

### Optional parameters
- `steps`: Number of simulation steps. Default: 6020
- `dt`: Each step progresses `dt` in time. Default: 0.0005
- `saveN`: Every `saveN`-th step is saved into the final simulation data. The rest is discarded. Default: 20
- `mesh_range`: If `mesh` is a directory, this sets the range of meshes to be used (order determined by python's os.walk). None, means all meshes are used. Default: None
- `vlevel`: Verbosity level. Min:0, Max:2. Default:1
- `cleanup_dir`: If not None, then instead of using `mesh`, the directory  `cleanup_dir` and its subdirectories are searched for files named 'failed_meshes.txt'. These files are assumed to contain paths to meshes. All meshes found this way are used for simulation. Default: None
- `num_frames`: If > 0, save animation of simulation as gif with num_frames frames. Default: 0

  
# Analyze dataset (Used for reverse-engineering parameters)

```
python examples/cfd/vortex_shedding_mgn/datasetAnalysis.py --dataset <str> --split <str> [--graph_num <int> --time_step <int>]
```
- `dataset`: Path to the dataset to be analyzed
- `split`: Which split inside the dataset is to be analyzed

### Optional parameters
- `graph_num`: Number of the simulation inside the dataset to analyze. Default: None
- `time_step`: Number of the time step within the `graph_num` to analyze. Default: None

