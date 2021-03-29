from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding

import numpy as np


class MultiStepEnv(Env):
    def __init__(self, horizon=1):
        self.horizon = horizon
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=1, shape=(3,))
        self.state_dict = {
            's_1': np.array([1, 0, 0]),
            's_2': np.array([0, 1, 0]),
            's_3': np.array([0, 0, 1]),
            's_4': np.array([1, 1, 1]),
            's_5': np.array([0, 0, 0]),
        }
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = 's_1'
        self.time_step = 0
        return self.state_dict[self.state]

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.0
        if self.state == 's_1':
            if action == 0:
                self.state = 's_2'
            elif action == 1:
                self.state = 's_3'
        elif self.state == 's_2':
            if action == 1:
                self.state = 's_4'
        elif self.state =='s_4':
            if action == 0:
                self.state = 's_5'

        self.time_step += 1
        if self.time_step == self.horizon:
            done = True
            if self.state == 's_2' and self.time_step == 1:
                reward = 1.0
            elif self.state == 's_4' and self.time_step == 2:
                reward = 1.0
            elif self.state == 's_5' and self.time_step == 3:
                reward = 1.0
        else:
            done = False

        return self.state_dict[self.state], reward, done, {'state': self.state}


if __name__ == "__main__":
    # test
    env = MultiStepEnv(horizon=1)
    print('=======horizon1=======')
    for i in range(2):
        print(i)
        obs = env.reset()
        obs, reward, done, info = env.step(i)
        print('======================')
        print('obs: {}'.format(obs))
        print('reward: {}'.format(reward))
        print('done: {}'.format(done))
        print('state: {}'.format(info['state']))
        print('======================')

    env = MultiStepEnv(horizon=2)
    for actions in [[0,0], [0,1], [1,0], [1,1]]:
        print('=======horizon2=======')
        print(actions)
        obs = env.reset()
        for i in range(2):
            obs, reward, done, info = env.step(actions[i])
            print('======================')
            print('obs: {}'.format(obs))
            print('reward: {}'.format(reward))
            print('done: {}'.format(done))
            print('state: {}'.format(info['state']))
            print('======================')

    env = MultiStepEnv(horizon=3)
    for actions in [[0,0,0], [0,0,1], [0,1,0], [1,0,0], [0,1,1], [1,0,1], [1,1,0], [1,1,1]]:
        print('=======horizon3=======')
        print(actions)
        obs = env.reset()
        for i in range(3):
            obs, reward, done, info = env.step(actions[i])
            print('======================')
            print('obs: {}'.format(obs))
            print('reward: {}'.format(reward))
            print('done: {}'.format(done))
            print('state: {}'.format(info['state']))
            print('======================')
