from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

from dotmap import DotMap

from MBExperiment import MBExperiment
from MPC import MPC
from config import create_config
import env # We run this so that the env is registered

import torch
import numpy as np
import random
import tensorflow as tf

from time import localtime, strftime

from dotmap import DotMap
from scipy.io import savemat
from tqdm import trange

from Agent import Agent
from DotmapUtils import get_required_argument

class PETS_adaptor:
    # def __init__(self, env, policy, horizon, params):
    def __init__(self, env):
        """
        usage: PETS_adaptor(env)
        env: gym environment, not stable baseline env!
        """
        env_str = "finrl"
        crtl_type = 'MPC' # fixed 
        crtl_args = []
        overrides = []
        logdir = './log'

        ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
        cfg = create_config(env_str, ctrl_type, ctrl_args, overrides, logdir)
        cfg.pprint()

        assert ctrl_type == 'MPC'

        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
        # have change how environment is passed here
        exp = MBExperiment(cfg.exp_cfg, env)

        os.makedirs(exp.logdir)
        with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
            f.write(pprint.pformat(cfg.toDict()))

        self._exp = exp

    def learn(self, total_timesteps, tb_log_name=None, callback=None):
        # do nothing with tensorboard for now
        self._exp.run_experiment()


    def predict(self, obs, deterministic=True):
        # def sample(self, horizon, policy, record_fname=None):
        self._exp.agent.sample()
        

    def save(self, filepath):
        """
        Saves the model parameters to a file.
        
        Arguments:
            filepath: Path to save the model.
        """
        np.save(filepath, self.reward_history)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Loads the model parameters from a file.
        
        Arguments:
            filepath: Path to load the model.
            
        Returns:
            A new instance of CustomModelWrapper with loaded parameters.
        """
        reward_history = np.load(filepath, allow_pickle=True)
        instance = CustomModelWrapper(None, None, None, None)  # Initialize with placeholders
        instance.reward_history = reward_history
        print(f"Model loaded from {filepath}")
        return instance
