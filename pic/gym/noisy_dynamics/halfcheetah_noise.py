import numpy as np
from gym.envs.mujoco import HalfCheetahEnv


class HalfCheetahNoiseEnv(HalfCheetahEnv):
    def __init__(self, noise_type='uniform', noise_scale=0.0, init_scale=0.0):
        self.noise_type = noise_type
        assert self.noise_type in ['normal', 'uniform']
        self.noise_scale = noise_scale
        self.init_scale = init_scale

        HalfCheetahEnv.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]

        # noise
        if self.noise_scale == 0.0:
            noise = np.zeros((1,))
        elif self.noise_type == 'normal':
            noise = np.random.normal(loc=0., scale=self.noise_scale, size=1)
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(-self.noise_scale, self.noise_scale, 1)
        self.do_simulation(action, self.frame_skip)
        # add noise
        self.sim.data.qvel[0] += noise[0]

        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        # original: self.init_scale=0.1
        qpos = self.init_qpos + self.np_random.uniform(low=-self.init_scale, high=self.init_scale, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * self.init_scale
        self.set_state(qpos, qvel)
        return self._get_obs()
