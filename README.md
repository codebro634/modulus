
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

# Generate Meshes

```
python dataGeneration/meshDatasetMaker.py --name <str> --num_meshes <int> [--width <float> --height <float> --ox <float> --oy <float> --osize <float> --inflow_peak_mean <float> --inflow_peak_max_deviation <float> --two_obj --rotate --stretch --circs --tris --quads]
```
- `name`: Name of the mesh dataset. The generated meshes are saved into meshes/`name`
- `num_meshes`: Number of meshes to generate

### Optional parameters
- `width`: Width of the channel
- `height`: Height of the channel
- `ox`: Mean of object's mid point's x-coordinate
- `oy`: Mean of object's mid point's y-coordinate
- `osize`: Mean object size (e.g. radius for circles)
- `inflow_peak_mean`:  Mean of the inflow peak
- `inflow_peak_max_derivation`: Inflow peak is sampled from uniform[inflow_peak_mean-inflow_peak_max_derivation,inflow_peak_mean+inflow_peak_max_derivation]
- `two_obj`: If set, there is a 50% chance a second object is added
- `rotate`: If set, the object(s) get randomly rotated
- `stretch`: If set, the object(s) get randomly stretched/squeezed in the x/y-direction
- `circs`: If set, circles are part of the possible objects that can be generated
- `tris`: If set, triangles are part of the possible objects that can be generated
- `quads`: If set, squares are part of the possible objects that can be generated

