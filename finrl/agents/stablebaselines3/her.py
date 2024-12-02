from __future__ import annotations

import numpy as np
import pandas as pd
from stable_baselines3 import DDPG, HerReplayBuffer
# from stable_baselines3.her import HER
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl import config
from finrl.meta.preprocessor.preprocessors import data_split


DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
HER_PARAMS = {"n_sampled_goal": 4, "goal_selection_strategy": "future"}

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except:
            pass
        return True
    
class DRLHERAgent:
    """Provides implementation for DDPG algorithm with HER for stock trading."""

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        tensorboard_log=None,
        buffer_kwargs=None,
    ):
        if model_kwargs is None:
            model_kwargs = DDPG_PARAMS
        if buffer_kwargs is None:
            buffer_kwargs = HER_PARAMS
        model = DDPG(
            policy=policy,
            env=self.env,
            # replay_buffer_class=HerReplayBuffer,
            # replay_buffer_kwargs=buffer_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )
        return model

    @staticmethod
    def train_model(
        model,
        tb_log_name="her_ddpg",
        total_timesteps=10000,
        callback=None,
    ):
        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=callback,
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        """Make a prediction and get results."""
        test_env, test_obs = environment.get_sb_env()
        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption
        # state_memory=[] #add memory pool to store states

        test_env.reset()
        max_steps = len(environment.df.index.unique()) - 1

        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)

            if (
                i == max_steps - 1
            ):  # more descriptive condition for early termination to clarify the logic
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            # add current state to state memory
            # state_memory=test_env.env_method(method_name="save_state_memory")

            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]