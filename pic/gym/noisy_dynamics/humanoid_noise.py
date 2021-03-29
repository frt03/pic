import numpy as np
from gym.envs.mujoco import HumanoidEnv


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidNoiseEnv(HumanoidEnv):
    def __init__(self, noise_type='uniform', noise_scale=0.0, init_scale=0.0):
        self.noise_type = noise_type
        assert self.noise_type in ['normal', 'uniform']
        self.noise_scale = noise_scale
        self.init_scale = init_scale

        HumanoidEnv.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat
                ]
                )

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)

        # noise
        if self.noise_scale == 0.0:
            noise = np.zeros((1,))
        elif self.noise_type == 'normal':
            noise = np.random.normal(loc=0., scale=self.noise_scale, size=1)
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(-self.noise_scale, self.noise_scale, 1)
        self.do_simulation(a, self.frame_skip)
        # add noise
        self.sim.data.qvel[0] += noise[0]

        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        # original: self.init_scale = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-self.init_scale, high=self.init_scale, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-self.init_scale, high=self.init_scale, size=self.model.nv,)
        )
        return self._get_obs()
