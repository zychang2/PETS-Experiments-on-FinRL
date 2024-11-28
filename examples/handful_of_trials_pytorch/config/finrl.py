from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DotmapUtils import get_required_argument

import gym
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PtModel(nn.Module):
    def __init__(self, ensemble_size, in_features, out_features, hidden_features=500):
        super().__init__()
        self.num_nets = ensemble_size
        self.in_features = in_features
        self.out_features = out_features

        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

        # Input normalization parameters
        self.inputs_mu = nn.Parameter(torch.zeros(in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.ones(in_features), requires_grad=False)

        # Log variance clamping
        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, out_features // 2) * 10.0)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Custom weight initialization."""
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.truncated_normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.constant_(layer.bias, 0.0)

    def compute_decays(self):
        """L2 regularization on weights."""
        decay = 0.0
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                decay += 0.0001 * (layer.weight ** 2).sum()
        return decay / 2.0

    def fit_input_stats(self, data):
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(TORCH_DEVICE).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(TORCH_DEVICE).float()

    def forward(self, inputs, ret_logvar=False):
        # Normalize inputs
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        # Pass through the model
        outputs = self.model(inputs)

        # Split outputs into mean and log variance
        mean = outputs[:, :, :self.out_features // 2]
        logvar = outputs[:, :, self.out_features // 2:]

        # Clamp log variance
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)


class FinRLConfigModule:
    ENV_NAME = "FinRL"
    TASK_HORIZON = 200
    NTRAIN_ITERS = 15
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 320, 291
    GP_NINDUCING_POINTS = 200

    # Create and move this tensor to GPU so that
    # we do not waste time moving it repeatedly to GPU later
    #ee_sub = torch.tensor([0.0, 0.6], device=TORCH_DEVICE, dtype=torch.float)

    def __init__(self, env):
        self.ENV = env #gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        return obs

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        # this cost function is wrong, cost depends on action and state
        stock_dim = self.ENV.stock_dim
        return -1*(obs[0] + sum(
                np.array(obs[1 : (stock_dim + 1)])
                * np.array(obs[(stock_dim + 1) : (stock_dim * 2 + 1)])
            ))

    @staticmethod
    def ac_cost_fn(acs):
        return 0.01 * (acs ** 2).sum(dim=1)

    def nn_constructor(self, model_init_cfg):

        ensemble_size = get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size")

        load_model = model_init_cfg.get("load_model", False)

        assert load_model is False, 'Has yet to support loading model'

        model = PtModel(ensemble_size,
                        self.MODEL_IN, self.MODEL_OUT * 2).to(TORCH_DEVICE)
        # * 2 because we output both the mean and the variance
        print(model.parameters())

        model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

        return model


CONFIG_MODULE = FinRLConfigModule
