import dm2gym
import gym
import random
import multiprocessing as mp
import numpy as np

import pic


class Sampler(object):
    def __init__(self, env_name, agent, max_episode_steps, n_samples=10**4, n_episodes=10**3, multiprocess=0):
        self.env_name = env_name
        self.agent = agent
        self.n_samples = n_samples
        self.n_episodes = n_episodes
        self.multiprocess = multiprocess
        self.max_episode_steps = max_episode_steps

    def sample(self):
        all_scores_per_param = []
        if self.multiprocess > 0:
            num_worker = mp.cpu_count()
            if self.multiprocess > num_worker:
                self.multiprocess = num_worker
            p = mp.Pool(self.multiprocess)
            print("num_worker: {}/{}".format(self.multiprocess, num_worker))

        for samp_num in range(self.n_samples):
            if samp_num % max(1, self.n_samples // 10) == 0:
                print(f"Sample {samp_num}/{self.n_samples}")
            score_episodes = []
            if self.multiprocess > 0:
                episodes_per_worker = max(1, int(np.ceil(self.n_episodes / self.multiprocess)))
                scores = p.starmap(run_episode_wrapper, [[i, self.env_name, self.agent, self.max_episode_steps, episodes_per_worker] for i in range(self.multiprocess)])
                scores = list(itertools.chain(*scores))[:self.n_episodes]
                assert len(scores) == self.n_episodes, f'{len(scores)} != {self.n_episodes}'
                score_episodes += scores
            else:
                env = make_env(env_name, seed=None)
                for _ in range(self.n_episodes):
                    score = run_episode(env, self.agent, self.max_episode_steps)
                    score_episodes.append(score)
            all_scores_per_param.append(score_episodes)
            self.agent.init_weights()

        if self.multiprocess > 0:
            p.close()

        return np.array(all_scores_per_param)


def make_env(env_name, seed=None):
    if "dm2gym" in env_name:
        env = gym.make(env_name, environment_kwargs={'flat_observation': True})
    else:
        env = gym.make(env_name)
    if seed is not None:
        env.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    return env


def run_episode(env, agent, max_episode_steps):
    obs = env.reset()
    score = 0
    steps = 0
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, r, done, _ = env.step(action)
        score += r
        steps += 1
        if steps >= max_episode_steps:
            done = True
    return score


def run_episode_wrapper(index, env_name, agent, max_episode_steps, num_episodes):
    env = make_env(env_name, index)
    return [run_episode(env, agent, max_episode_steps) for _ in range(num_episodes)]
