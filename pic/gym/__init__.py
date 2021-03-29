from gym.envs.registration import register
from pic.gym.reward_shaping.maze_model import U_MAZE


register(
    id='OneStep-v0',
    entry_point='pic.gym.multi_step.multi_step:MultiStepEnv',
    kwargs={'horizon': 1},
)

register(
    id='TwoStep-v0',
    entry_point='pic.gym.multi_step.multi_step:MultiStepEnv',
    kwargs={'horizon': 2},
)

register(
    id='ThreeStep-v0',
    entry_point='pic.gym.multi_step.multi_step:MultiStepEnv',
    kwargs={'horizon': 3},
)

register(
    id='ReacherL1-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherL1Env',
    max_episode_steps=50,
)

register(
    id='ReacherL2-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherL2Env',
    max_episode_steps=50,
)

register(
    id='ReacherSparse-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherSparseEnv',
    max_episode_steps=50,
)

register(
    id='ReacherFrac-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherFracEnv',
    max_episode_steps=50,
)

register(
    id='ReacherL1_c05-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherL1Env',
    max_episode_steps=50,
    kwargs={'coefficent': 0.5},
)

register(
    id='ReacherL1_c20-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherL1Env',
    max_episode_steps=50,
    kwargs={'coefficent': 2.0},
)

register(
    id='ReacherL1_c50-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherL1Env',
    max_episode_steps=50,
    kwargs={'coefficent': 5.0},
)

register(
    id='ReacherL2_c05-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherL2Env',
    max_episode_steps=50,
    kwargs={'coefficent': 0.5},
)

register(
    id='ReacherL2_c20-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherL2Env',
    max_episode_steps=50,
    kwargs={'coefficent': 2.0},
)

register(
    id='ReacherL2_c50-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherL2Env',
    max_episode_steps=50,
    kwargs={'coefficent': 5.0},
)

register(
    id='ReacherSparse_d001-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherSparseEnv',
    max_episode_steps=50,
    kwargs={'distance_threshold': 0.01},
)

register(
    id='ReacherSparse_d01-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherSparseEnv',
    max_episode_steps=50,
    kwargs={'distance_threshold': 0.1},
)

register(
    id='ReacherSparse_d015-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherSparseEnv',
    max_episode_steps=50,
    kwargs={'distance_threshold': 0.15},
)

register(
    id='ReacherFrac_m01o01-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherFracEnv',
    max_episode_steps=50,
    kwargs={'multiplier': 0.1, 'offset': 0.1},
)

register(
    id='ReacherFrac_m001o001-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherFracEnv',
    max_episode_steps=50,
    kwargs={'multiplier': 0.01, 'offset': 0.01},
)

register(
    id='ReacherFrac_m005o01-v0',
    entry_point='pic.gym.reward_shaping.reacher_norm:ReacherFracEnv',
    max_episode_steps=50,
    kwargs={'multiplier': 0.05, 'offset': 0.1},
)

register(
    id='maze2d-umaze-negative_sparse-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'negative_sparse',
        'distance_threshold': 0.5,
    }
)

register(
    id='maze2d-umaze-densel2-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'densel2',
        'coefficent': 1.0,
    }
)

register(
    id='maze2d-umaze-densel1-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'densel1',
        'coefficent': 1.0,
    }
)

register(
    id='maze2d-umaze-frac-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'frac',
        'multiplier': 0.01,
        'offset': 0.1,
    }
)

register(
    id='maze2d-umaze-negative_sparse_d10-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'negative_sparse',
        'distance_threshold': 1.0,
    }
)

register(
    id='maze2d-umaze-negative_sparse_d01-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'negative_sparse',
        'distance_threshold': 0.1,
    }
)

register(
    id='maze2d-umaze-negative_sparse_d02-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'negative_sparse',
        'distance_threshold': 0.2,
    }
)

register(
    id='maze2d-umaze-densel1_c05-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'densel1',
        'coefficent': 0.5,
    }
)

register(
    id='maze2d-umaze-densel1_c50-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'densel1',
        'coefficent': 5.0,
    }
)

register(
    id='maze2d-umaze-densel1_c20-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'densel1',
        'coefficent': 2.0,
    }
)

register(
    id='maze2d-umaze-densel2_c05-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'densel2',
        'coefficent': 0.5,
    }
)

register(
    id='maze2d-umaze-densel2_c50-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'densel2',
        'coefficent': 5.0,
    }
)

register(
    id='maze2d-umaze-densel2_c20-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'densel2',
        'coefficent': 2.0,
    }
)

register(
    id='maze2d-umaze-frac_m01o01-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'frac',
        'multiplier': 0.1,
        'offset': 0.1,
    }
)

register(
    id='maze2d-umaze-frac_m001o001-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'frac',
        'multiplier': 0.01,
        'offset': 0.01,
    }
)

register(
    id='maze2d-umaze-frac_m005o01-v0',
    entry_point='pic.gym.reward_shaping.maze_model:MazeEnv',
    max_episode_steps=150,
    kwargs={
        'reward_type': 'frac',
        'multiplier': 0.05,
        'offset': 0.1,
    }
)

register(
    id='CartPoleNoise-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.0,
        'init_scale': 0.0,
    }
)

register(
    id='HalfCheetahNoise-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.0,
        'init_scale': 0.0,
    }
)

register(
    id='HumanoidNoise-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.0,
        'init_scale': 0.0,
    }
)

# noisy_dynamics
# CartPole
register(
    id='CartPoleNoiseInit005Dynamics003-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.03,
        'init_scale': 0.05,
    }
)

register(
    id='CartPoleNoiseInit005Dynamics005-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.05,
        'init_scale': 0.05,
    }
)

register(
    id='CartPoleNoiseInit005Dynamics010-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.1,
        'init_scale': 0.05,
    }
)

register(
    id='CartPoleNoiseInit010Dynamics000-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.0,
        'init_scale': 0.1,
    }
)

register(
    id='CartPoleNoiseInit010Dynamics003-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.03,
        'init_scale': 0.1,
    }
)

register(
    id='CartPoleNoiseInit010Dynamics005-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.05,
        'init_scale': 0.1,
    }
)

register(
    id='CartPoleNoiseInit010Dynamics010-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.1,
        'init_scale': 0.1,
    }
)

register(
    id='CartPoleNoiseInit015Dynamics000-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.0,
        'init_scale': 0.15,
    }
)

register(
    id='CartPoleNoiseInit015Dynamics003-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.03,
        'init_scale': 0.15,
    }
)

register(
    id='CartPoleNoiseInit015Dynamics005-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.05,
        'init_scale': 0.15,
    }
)

register(
    id='CartPoleNoiseInit015Dynamics010-v0',
    entry_point='pic.gym.noisy_dynamics.cartpole_noise:CartPoleNoiseEnv',
    max_episode_steps=200,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.1,
        'init_scale': 0.15,
    }
)

# HalfCheetah
register(
    id='HalfCheetahNoiseInit010Dynamics003-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.03,
        'init_scale': 0.1,
    }
)

register(
    id='HalfCheetahNoiseInit010Dynamics005-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.05,
        'init_scale': 0.1,
    }
)

register(
    id='HalfCheetahNoiseInit010Dynamics010-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.1,
        'init_scale': 0.1,
    }
)

register(
    id='HalfCheetahNoiseInit030Dynamics000-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.00,
        'init_scale': 0.3,
    }
)

register(
    id='HalfCheetahNoiseInit030Dynamics003-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.03,
        'init_scale': 0.3,
    }
)

register(
    id='HalfCheetahNoiseInit030Dynamics005-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.05,
        'init_scale': 0.3,
    }
)

register(
    id='HalfCheetahNoiseInit030Dynamics010-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.1,
        'init_scale': 0.3,
    }
)

register(
    id='HalfCheetahNoiseInit050Dynamics000-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.00,
        'init_scale': 0.5,
    }
)

register(
    id='HalfCheetahNoiseInit050Dynamics003-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.03,
        'init_scale': 0.5,
    }
)

register(
    id='HalfCheetahNoiseInit050Dynamics005-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.05,
        'init_scale': 0.5,
    }
)

register(
    id='HalfCheetahNoiseInit050Dynamics010-v2',
    entry_point='pic.gym.noisy_dynamics.halfcheetah_noise:HalfCheetahNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.1,
        'init_scale': 0.5,
    }
)

# Humanoid
register(
    id='HumanoidNoiseInit001Dynamics003-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.03,
        'init_scale': 0.01,
    }
)

register(
    id='HumanoidNoiseInit001Dynamics005-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.05,
        'init_scale': 0.01,
    }
)

register(
    id='HumanoidNoiseInit001Dynamics010-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.1,
        'init_scale': 0.01,
    }
)

register(
    id='HumanoidNoiseInit003Dynamics000-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.0,
        'init_scale': 0.03,
    }
)

register(
    id='HumanoidNoiseInit003Dynamics003-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.03,
        'init_scale': 0.03,
    }
)

register(
    id='HumanoidNoiseInit003Dynamics005-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.05,
        'init_scale': 0.03,
    }
)

register(
    id='HumanoidNoiseInit003Dynamics010-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.1,
        'init_scale': 0.03,
    }
)

register(
    id='HumanoidNoiseInit005Dynamics000-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.0,
        'init_scale': 0.05,
    }
)

register(
    id='HumanoidNoiseInit005Dynamics003-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.03,
        'init_scale': 0.05,
    }
)

register(
    id='HumanoidNoiseInit005Dynamics005-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.05,
        'init_scale': 0.05,
    }
)

register(
    id='HumanoidNoiseInit005Dynamics010-v2',
    entry_point='pic.gym.noisy_dynamics.humanoid_noise:HumanoidNoiseEnv',
    max_episode_steps=1000,
    kwargs={
        'noise_type': 'uniform',
        'noise_scale': 0.1,
        'init_scale': 0.05,
    }
)
