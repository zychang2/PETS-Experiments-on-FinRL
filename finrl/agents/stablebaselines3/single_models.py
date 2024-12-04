# DRL models from Stable Baselines 3
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

MODELS = {"A2C": A2C, "DDPG": DDPG, "TD3": TD3, "SAC": SAC, "PPO": PPO}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])

        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])

            except BaseException as inner_error:
                # Handle the case where neither "rewards" nor "reward" is found
                self.logger.record(key="train/reward", value=None)
                # Print the original error and the inner error for debugging
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True



class DRLSingleAgent:
    @staticmethod
    def get_model(
        model_name,
        env,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        seed=None,
        verbose=1,
    ):
        if model_kwargs is None:
            temp_model_kwargs = MODEL_KWARGS[model_name]
        else:
            temp_model_kwargs = model_kwargs.copy()

        if "action_noise" in temp_model_kwargs:
            n_actions = env.action_space.shape[-1]
            temp_model_kwargs["action_noise"] = NOISE[
                temp_model_kwargs["action_noise"]
            ](mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        print(temp_model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **temp_model_kwargs,
        )

    @staticmethod
    def train_model(model, model_name, tb_log_name, iter_num, total_timesteps=5000):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        model.save(
            f"{config.TRAINED_MODEL_DIR}/{model_name}_{total_timesteps // 1000}k_{iter_num}"
        )
        return model

    @staticmethod
    def get_validation_sharpe(iteration, model_name):
        """Calculate Sharpe ratio based on validation results"""
        df_total_value = pd.read_csv(
            f"results/account_value_validation_{model_name}_{iteration}.csv"
        )
        # If the agent did not make any transaction
        if df_total_value["daily_return"].var() == 0:
            if df_total_value["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            return (
                (4**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )

    def __init__(
        self,
        df,
        train_period,
        val_test_period,
        rebalance_window,
        validation_window,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        print_verbosity,
        model_name,
    ):
        self.model_name = model_name.upper()
        self.df = df
        self.train_period = train_period
        self.val_test_period = val_test_period

        self.unique_trade_date = df[
            (df.date > val_test_period[0]) & (df.date <= val_test_period[1])
        ].date.unique()
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.print_verbosity = print_verbosity
        self.train_env = None  # defined in train_validation() function

    def DRL_validation(self, model, test_data, test_env, test_obs):
        """validation process"""
        for _ in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)

    def DRL_prediction(
        self, model, last_state, iter_num, turbulence_threshold, initial
    ):
        """make a prediction based on trained model"""

        # trading env
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[iter_num - self.rebalance_window],
            end=self.unique_trade_date[iter_num],
        )
        trade_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=trade_data,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    initial=initial,
                    previous_state=last_state,
                    model_name=self.model_name,
                    mode="trade",
                    iteration=iter_num,
                    print_verbosity=self.print_verbosity,
                )
            ]
        )

        trade_obs = trade_env.reset()

        for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(trade_obs)
            trade_obs, rewards, dones, info = trade_env.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                # print(env_test.render())
                last_state = trade_env.envs[0].render()

        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(f"results/last_state_{self.model_name}_{i}.csv", index=False)
        return last_state

    def _train_window(
        self,
        model_kwargs,
        sharpe_list,
        validation_start_date,
        validation_end_date,
        timesteps,
        i,
        validation,
        turbulence_threshold,
    ):
        """
        Train the model for a single window.
        """
        if model_kwargs is None:
            return None, sharpe_list, -1

        print(f"======{self.model_name} Training========")
        model = self.get_model(
            self.model_name, self.train_env, policy="MlpPolicy", model_kwargs=model_kwargs
        )
        model = self.train_model(
            model,
            self.model_name,
            tb_log_name=f"{self.model_name}_{i}",
            iter_num=i,
            total_timesteps=timesteps,
        )  # 100_000
        print(
            f"======{self.model_name} Validation from: ",
            validation_start_date,
            "to ",
            validation_end_date,
        )
        val_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=validation,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    iteration=i,
                    model_name=self.model_name,
                    mode="validation",
                    print_verbosity=self.print_verbosity,
                )
            ]
        )
        val_obs = val_env.reset()
        self.DRL_validation(
            model=model,
            test_data=validation,
            test_env=val_env,
            test_obs=val_obs,
        )
        sharpe = self.get_validation_sharpe(i, model_name=self.model_name)
        print(f"{self.model_name} Sharpe Ratio: ", sharpe)
        sharpe_list.append(sharpe)
        return model, sharpe_list, sharpe

    def run_single_model_strategy(
        self,
        model_kwargs,
        timesteps,
    ):
        # Model Parameters
        sharpe_list = []
        sharpe = -1

        """Strategy for single model"""
        print(f"============Start {self.model_name} Strategy============")
        # for ensemble model, it's necessary to feed the last state
        # of the previous model to the current model as the initial state
        last_state_ensemble = []

        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []

        insample_turbulence = self.df[
            (self.df.date < self.train_period[1])
            & (self.df.date >= self.train_period[0])
        ]
        insample_turbulence_threshold = np.quantile(
            insample_turbulence.turbulence.values, 0.90
        ) # Generates threshold at 90% of turbulence values in training data during period

        start = time.time()
        for i in range(
            self.rebalance_window + self.validation_window, # 126
            len(self.unique_trade_date), # length of validation period (~ 3 years)
            self.rebalance_window, # intervals of 63 days
        ):
            validation_start_date = self.unique_trade_date[
                i - self.rebalance_window - self.validation_window
            ] # Starts at dates corresponding to index 0 (126-63-63) and increases by 63 days each iteration
            validation_end_date = self.unique_trade_date[i - self.rebalance_window] # Starts at 63rd day and increases by 63 days each iteration

            validation_start_date_list.append(validation_start_date) # Keeps track of dates for plotting
            validation_end_date_list.append(validation_end_date) # Keeps track of dates for plotting
            iteration_list.append(i) # Keeps track of iterations for plotting

            print("============================================")
            # initial state is empty
            if i - self.rebalance_window - self.validation_window == 0:
                # inital state
                initial = True
            else:
                # previous state
                initial = False

            end_date_index = self.df.index[
                self.df["date"]
                == self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ]
            ].to_list()[-1]
            start_date_index = end_date_index - 63 + 1

            historical_turbulence = self.df.iloc[
                start_date_index : (end_date_index + 1), :
            ]

            historical_turbulence = historical_turbulence.drop_duplicates(
                subset=["date"]
            )

            historical_turbulence_mean = np.mean(
                historical_turbulence.turbulence.values
            )

            # print(historical_turbulence_mean)

            if historical_turbulence_mean > insample_turbulence_threshold:
                turbulence_threshold = insample_turbulence_threshold
            else:
                turbulence_threshold = np.quantile(
                    insample_turbulence.turbulence.values, 1
                )

            turbulence_threshold = np.quantile(
                insample_turbulence.turbulence.values, 0.99
            )
            print("turbulence_threshold: ", turbulence_threshold)

            # Environment Setup starts
            # training env
            train = data_split(
                self.df,
                start=self.train_period[0], # Training always starts with 2010 date no matter the iteration
                end=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ], # Ends at the start of the validation period every iteration
            ) # Doesn't split data, just reorganizes on "date" column and reindexes
            self.train_env = DummyVecEnv( # DummyVecEnv is native to stablebaselines library that allows the StockTradingEnv to be compatible with stablebaselines A2C
                [ # I don't feel this is necessary if you are not using a model native to stablebaselines
                    lambda: StockTradingEnv( # This is our training environment using OpenAI gym (see env_stocktrading.py)
                        df=train,
                        stock_dim=self.stock_dim,
                        hmax=self.hmax,
                        initial_amount=self.initial_amount,
                        num_stock_shares=[0] * self.stock_dim,
                        buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                        sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                        reward_scaling=self.reward_scaling,
                        state_space=self.state_space,
                        action_space=self.action_space,
                        tech_indicator_list=self.tech_indicator_list,
                        print_verbosity=self.print_verbosity,
                    )
                ]
            )

            validation = data_split(
                self.df,
                start=self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
                end=self.unique_trade_date[i - self.rebalance_window],
            ) # Again, doesn't split data, just creates new dataset starting at validation start window and ending 63 days later

            # Training and Validation starts
            print(
                "======Model training from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )

            model, sharpe_list, sharpe = self._train_window( # Where actual training occurs, see above function
                model_kwargs,
                sharpe_list,
                validation_start_date,
                validation_end_date,
                timesteps,
                i,
                validation,
                turbulence_threshold,
            )

            print(
                "======Best Model Retraining from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[i - self.rebalance_window],
            )

            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.rebalance_window],
                "to ",
                self.unique_trade_date[i],
            )

            last_state_ensemble = self.DRL_prediction( # See function above
                model=model,
                last_state=last_state_ensemble, # This is especially important for ensembling because models needs access to other models final states
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
            ) 

        end = time.time()
        print("Strategy took: ", (end - start) / 60, " minutes")

        df_summary = pd.DataFrame(
            [
                iteration_list,
                validation_start_date_list,
                validation_end_date_list,
                sharpe_list
            ]
        ).T
        df_summary.columns = [
            "Iter",
            "Val Start",
            "Val End",
            f"{self.model_name} Sharpe"
        ]

        return df_summary # Returns sharpe and valuation information
