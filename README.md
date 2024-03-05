
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
- `num_samples: Number of different simulations used in training
- `num_time_steps`: Number of time steps per simulation in training

Optional parameters
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
