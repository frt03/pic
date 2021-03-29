import numpy as np

from pic.nn import NumpyMLP


class NumpyAgent:
    def __init__(
        self,
        env,
        n_hidden_layers=2,
        n_hidden_units=4,
        random_dist="normal",
        normal_mean=0.0,
        normal_sigma=1.0,
        uniform_bound=1.0,
        act_fn="tanh",
        use_bias=False,
        env_name='CartPole-v0',
        policy_type="deterministic",
    ):
        self.env_name = env_name

        if not ("dm2gym" in self.env_name):
            self.n_inputs = env.reset().size
        else:  # for dm2gym
            self.n_inputs = env.reset()['observations'].size

        self.policy_type = policy_type
        assert self.policy_type in ["deterministic", "stochastic"]

        if type(env.action_space).__name__ == "Discrete":
            self.action_space_type = "discrete"
            self.n_actions = env.action_space.n
            self.n_outputs = self.n_actions
            if self.policy_type == "stochastic":
                self.output_fn = self.discrete_dist_sample
            else:
                self.output_fn = np.argmax
        elif type(env.action_space).__name__ == "Box":
            self.action_space_type = "continuous"
            self.action_scale = env.action_space.high.max()
            self.n_actions = env.action_space.shape[0]
            if self.policy_type == "stochastic":
                self.output_fn = self.continuous_dist_sample
                self.n_outputs = 2 * self.n_actions
            else:
                self.output_fn = self.scale_continuous_action
                self.n_outputs = self.n_actions

        self.nn = NumpyMLP(
            n_inputs=self.n_inputs,
            n_outputs=self.n_outputs,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
            random_dist=random_dist,
            normal_mean=normal_mean,
            normal_sigma=normal_sigma,
            uniform_bound=uniform_bound,
            act_fn=act_fn,
            use_bias=use_bias
            )

    def get_action(self, state):
        if ("dm2gym" in self.env_name):
            state = state['observations']
        x = self.nn.forward(state)
        return self.output_fn(x)

    def scale_continuous_action(self, x):
        return self.action_scale * np.tanh(x)

    def discrete_dist_sample(self, x):
        softmax_x = np.exp(x) / np.exp(x).sum()
        return np.random.choice(list(range(self.N_actions)), p=softmax_x)

    def continuous_dist_sample(self, x):
        mus_NN = x[: self.n_actions]
        sigmas_NN = x[self.n_actions:]
        mus = np.tanh(mus_NN) * self.action_scale
        sigmas = np.log(1 + np.exp(sigmas_NN))
        return np.random.normal(loc=mus, scale=sigmas)

    def init_weights(self):
        self.nn.init_weights()
