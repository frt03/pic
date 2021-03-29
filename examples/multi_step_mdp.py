import argparse
import datetime
import gym
import os

import pic
import multiprocessing as mp
import numpy as np
from multiprocessing import Pool
from scipy.special import expit as sigmoid, logit


_basic_columns = (
    "episodes",
    "mutual_infomation",
    "marginal",
    "conditional",
    "mean",
    "min",
    "max",
    "var",
    "eval_reward_mean",
    "prior_mean_0",
    "prior_mean_1",
    "prior_mean_2",
)


class Agent(object):
    def __init__(self, state_dim=3):
        self.param = np.zeros(3)

    def set_weight(self, weight):
        self.param = weight  # (3, )

    def sample(self, state):
        b = np.random.uniform(0, 1, 1)
        theta_s = np.dot(self.param.T, state)
        action = int(1 - (theta_s > logit(b)).astype('int'))
        return action


class MultiSampler(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run_episode(self, index):
        obs = self.env.reset()
        score = 0
        steps = 0
        done = False
        while not done:
            action = self.agent.sample(obs)
            obs, reward, done, _ = self.env.step(action)
            score += reward
        return score

    def set_weight(self, weight):
        self.agent.set_weight(weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_sigma", type=float, default=1.0, help="std of p(\theta)")
    parser.add_argument("--prior_mean", type=float, default=0.0, help="mean of p(\theta)")
    parser.add_argument("--horizon", type=int, default=1, help="horizon of MDP")
    parser.add_argument("--multiprocess", type=int, default=1, help="number of prosess for distrbuted experiments")
    parser.add_argument("--population_size", type=int, default=1000, help="number of population for optimization")
    parser.add_argument("--episodes_per_param", type=int, default=1000, help="the number of episodes per parameter")
    parser.add_argument("--iterations", type=int, default=100, help="the number of episodes for optimization")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="learning rate of the parameter")
    parser.add_argument("--decay", type=float, default=0.999, help="decay of learning rate")
    parser.add_argument("--output", type=str, default='./outputs/')
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    np.random.seed(seed=args.seed)

    # save dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
    output_dir = os.path.join(
        args.output,
        'multi_step_es_horizon{}'.format(args.horizon),
        'ps{}_pm{}_pop{}_ep{}_seed-{}'.format(
            args.prior_sigma,
            args.prior_mean,
            args.population_size,
            args.episodes_per_param,
            args.seed
            ),
        timestamp
        )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "scores.txt"), "w") as f:
        print("\t".join(_basic_columns), file=f)

    if args.horizon == 1:
        env = gym.make('OneStep-v0')
    elif args.horizon == 2:
        env = gym.make('TwoStep-v0')
    elif args.horizon == 3:
        env = gym.make('ThreeStep-v0')

    agent = Agent()
    multisampler = MultiSampler(env, agent)

    if args.multiprocess > 0:
        num_worker = mp.cpu_count()
        if args.multiprocess > num_worker:
            args.multiprocess = num_worker
        pool = Pool(args.multiprocess)
        print("num_worker: {}/{}".format(args.multiprocess, num_worker))

    # sampling parameters
    prior_mu = np.array([args.prior_mean] * 3)
    for itr in range(args.iterations):
        # sampling parameters
        # [population_size, episodes_per_param]
        mu = np.ones((args.population_size, 3)) * prior_mu
        noise = np.random.randn(args.population_size, 3)
        theta = mu + args.prior_sigma * noise

        all_scores = []
        all_scores_per_param = []
        # simulate
        for population in theta:
            multisampler.set_weight(population)
            if args.multiprocess > 0:
                scores = pool.map(multisampler.run_episode, range(args.episodes_per_param))
                assert len(scores) == args.episodes_per_param
            all_scores += scores
            all_scores_per_param.append(scores)

        all_scores = np.array(all_scores)
        all_scores_per_param = np.array(all_scores_per_param)

        p = all_scores.sum()/(args.population_size * args.episodes_per_param)
        marginal = -p * np.log(p + 1e-10) -(1 - p) * np.log((1 - p) + 1e-10)
        ps = all_scores_per_param.sum(axis=1) / args.episodes_per_param
        conditional = np.mean(-ps * np.log(ps + 1e-10) -(1 - ps) * np.log((1 - ps) + 1e-10))
        mutual_info = marginal - conditional
        reward_mean = all_scores.mean()
        reward_variance = all_scores.var()
        reward_mean_min = all_scores_per_param.mean(axis=1).min()
        reward_mean_max = all_scores_per_param.mean(axis=1).max()

        # update
        reward_mean_over_ep = all_scores_per_param.mean(axis=1)
        std = reward_mean_over_ep.std()
        if std == 0:
            std = 1e-10
        # normalize
        reward_mean_over_ep = (reward_mean_over_ep - reward_mean) / std
        update_factor = 1. / (args.population_size * args.prior_sigma)
        g = update_factor * np.dot(noise.T, reward_mean_over_ep).T
        prior_mu += args.learning_rate * g

        # evaluation
        n_eval = 100
        if args.multiprocess > 0:
            eval_reward = pool.map(multisampler.run_episode, range(n_eval))
            assert len(eval_reward) == n_eval
        eval_reward_mean = np.array(eval_reward).mean()

        values = (
            (itr + 1) * args.episodes_per_param * args.population_size,
            mutual_info,
            marginal,
            conditional,
            reward_mean,
            reward_mean_min,
            reward_mean_max,
            reward_variance,
            eval_reward_mean,
            prior_mu[0],
            prior_mu[1],
            prior_mu[2],
            )
        with open(os.path.join(output_dir, "scores.txt"), "a+") as f:
            print("\t".join(str(x) for x in values), file=f)


if __name__ == "__main__":
    main()
