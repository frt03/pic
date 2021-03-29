import numpy as np
from gym.envs.mujoco import ReacherEnv


class ReacherL1Env(ReacherEnv):
    def __init__(self, coefficent=1.0):
        self.coefficent = coefficent

        ReacherEnv.__init__(self)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward = - self.coefficent * np.linalg.norm(vec, ord=1)  # L1 norm
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}


class ReacherL2Env(ReacherEnv):
    def __init__(self, coefficent=1.0):
        self.coefficent = coefficent

        ReacherEnv.__init__(self)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward = - self.coefficent * np.linalg.norm(vec)  # L2 norm
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}


class ReacherSparseEnv(ReacherEnv):
    def __init__(self, distance_threshold=0.05):
        self.distance_threshold = distance_threshold

        ReacherEnv.__init__(self)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        d = np.linalg.norm(vec)
        reward = -(d > self.distance_threshold).astype(np.float32)  # Sparse
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}


class ReacherFracEnv(ReacherEnv):
    def __init__(self, multiplier=0.01, offset=0.1):
        self.multiplier = multiplier
        self.offset = offset

        ReacherEnv.__init__(self)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        d = np.linalg.norm(vec)
        reward = self.multiplier / (self.offset + d)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}
