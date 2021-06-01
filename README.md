# Policy Information Capacity: Information-Theoretic Measure for Task Complexity in Deep Reinforcement Learning
[[arxiv]](https://arxiv.org/abs/2103.12726)

If you use this codebase for your research, please cite the paper:
```
@inproceedings{furuta2021pic,
  title={Policy Information Capacity: Information-Theoretic Measure for Task Complexity in Deep Reinforcement Learning},
  author={Hiroki Furuta and Tatsuya Matsushima and Tadashi Kozuno and Yutaka Matsuo and Sergey Levine and Ofir Nachum and Shixiang Shane Gu},
  booktitle={International Conference on Machine Learning},
  year={2021}
}
```


## Dependencies
We recommend you to use Docker. See [README](./docker/README.md) for setting up.

## Examples
See [examples](./examples) for the details.

For synthetic experiments:
```
python multi_step_mdp_optimality.py --iterations 100 --population_size 1000 --episodes_per_param 1000 --prior_mean 0.0 --prior_sigma 1.0 --horizon 3 --multiprocess 64

python multi_step_mdp.py --iterations 100 --population_size 1000 --episodes_per_param 1000 --prior_mean 0.0 --prior_sigma 1.0 --horizon 2 --multiprocess 64
```

For random sampling:
```
python random_sampling.py --env CartPole-v0 --random_dist normal --multiprocess 64 --n_units 64 --n_layers 2 --n_samples 1000 --n_episodes 1000

python random_sampling.py --env dm2gym:CheetahRun-v0 --random_dist uniform --multiprocess 64 --n_units 64 --n_layers 2 --n_samples 1000 --n_episodes 1000
```

For mutual information estimation (PIC and POIC):
```
python mi_estimate.py --sourse_path ./results/CartPole-v0.npy --env CartPole-v0
```

## Environment List
```
CartPole-v0
Pendulum-v0
MountainCar-v0
MountainCarContinuous-v0
Acrobot-v1
Ant-v2
HalfCheetah-v2
Walker2d-v2
Humanoid-v2
Hopper-v2
dm2gym:CheetahRun-v0
dm2gym:ReacherEasy-v0
dm2gym:Ball_in_cupCatch-v0
```

For reward shaping experiments, see [here](./pic/gym/__init__.py) for the details.

## Reference
This codebase is based on [RWG](https://github.com/declanoller/RWG_benchmarking). We use the implementation of pointmaze environment in [D4RL](https://github.com/rail-berkeley/d4rl).
