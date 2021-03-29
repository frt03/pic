import numpy as np


class NumpyMLP:
    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden_layers=2,
        n_hidden_units=4,
        random_dist="normal",
        normal_mean=0.0,
        normal_sigma=1.0,
        uniform_bound=1.0,
        act_fn="tanh",
        use_bias=False
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        random_dists = ['normal', 'uniform', 'xavier_uniform', 'xavier_normal']
        assert (random_dist in random_dists)
        self.random_dist = random_dist
        self.random_dist_scaling = 1.0
        self.normal_mean = normal_mean
        self.normal_sigma = normal_sigma
        self.uniform_bound = uniform_bound
        self.use_bias = use_bias

        self.init_weights()

        activation_fn_d = {
            "tanh": np.tanh,
            "linear": lambda x: x,
            "relu": lambda x: np.maximum(0, x),
        }
        assert (act_fn in activation_fn_d.keys())
        self.act_fn = activation_fn_d[act_fn]

    def init_weights(self):
        self.weights_matrix = []
        mat_input_size = self.n_inputs
        if self.use_bias:
            mat_input_size += 1

        for i in range(self.n_hidden_layers):
            mat_output_size = self.n_hidden_units
            if self.random_dist == "normal":
                mat = np.random.normal(loc=self.normal_mean, scale=self.normal_sigma, size=(mat_output_size, mat_input_size))
            elif self.random_dist == "uniform":
                mat = np.random.uniform(-self.uniform_bound, self.uniform_bound, (mat_output_size, mat_input_size))
            elif self.random_dist == "xavier_uniform":
                bound = 5 / 3 * np.sqrt(6 / (mat_output_size + mat_input_size))  # for tanh
                mat = np.random.uniform(-bound, bound, (mat_output_size, mat_input_size))
            elif self.random_dist == "xavier_normal":
                bound = 5 / 3 * np.sqrt(2 / (mat_output_size + mat_input_size))  # for tanh
                mat = np.random.normal(loc=0.0, scale=bound, size=(mat_output_size, mat_input_size))
            else:
                raise
            self.weights_matrix.append(self.random_dist_scaling * mat)
            mat_input_size = mat_output_size
            if self.use_bias:
                mat_input_size += 1
        # for the last layer:
        if self.random_dist == "normal":
            mat = np.random.normal(loc=self.normal_mean, scale=self.normal_sigma, size=(self.n_outputs, mat_input_size))
        elif self.random_dist == "uniform":
            mat = np.random.uniform(-self.uniform_bound, self.uniform_bound, (self.n_outputs, mat_input_size))
        elif self.random_dist == "xavier_uniform":
            bound = 5 / 3 * np.sqrt(6 / (self.N_outputs + mat_input_size))  # for tanh
            mat = np.random.uniform(-bound, bound, (self.N_outputs, mat_input_size))
        elif self.random_dist == "xavier_normal":
            bound = 5 / 3 * np.sqrt(2 / (self.N_outputs + mat_input_size))  # for tanh
            mat = np.random.normal(loc=0., scale=bound, size=(self.N_outputs, mat_input_size))
        self.weights_matrix.append(self.random_dist_scaling * mat)

        self.w_mat_shapes = [w.shape for w in self.weights_matrix]
        self.w_mat_lens = [len(w.flatten()) for w in self.weights_matrix]
        self.n_weights = sum(self.w_mat_lens)

    def forward(self, x):
        for i, w in enumerate(self.weights_matrix):
            if self.use_bias:
                x = np.concatenate((x, [1.0]))
            x = np.dot(w, x)
            if i < self.n_hidden_layers:
                x = self.act_fn(x)
        return x
