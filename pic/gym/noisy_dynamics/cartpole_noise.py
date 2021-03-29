import numpy as np

from gym.envs.classic_control import CartPoleEnv


class CartPoleNoiseEnv(CartPoleEnv):
    def __init__(self, noise_type='uniform', noise_scale=0.0, init_scale=0.0):
        self.noise_type = noise_type
        assert self.noise_type in ['normal', 'uniform']
        self.noise_scale = noise_scale
        self.init_scale = init_scale

        CartPoleEnv.__init__(self)

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.noise_scale == 0.0:
            noise = np.zeros((1,))
        elif self.noise_type == 'normal':
            noise = np.random.normal(loc=0., scale=self.noise_scale, size=1)
        elif self.noise_type == 'uniform':
            noise = np.random.uniform(-self.noise_scale, self.noise_scale, 1)

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc + noise[0]
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc + noise[0]
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        # original: low=-0.05, high=0.05
        self.state = self.np_random.uniform(low=-self.init_scale, high=self.init_scale, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)
