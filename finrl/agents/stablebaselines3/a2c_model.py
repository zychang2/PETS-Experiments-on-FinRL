# DRL models from Stable Baselines 3
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split


A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007} # Specific to A2C

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

class DRLA2CAgent:
    @staticmethod
    def get_model(
        env,
        policy="MlpPolicy", # Specific to stablebaselines, actor critic with 2 hidden layers, 64 neurons (need to define for PETS)
        policy_kwargs=None,
        model_kwargs=A2C_PARAMS, # calls the params defined above
        seed=None,
        verbose=1,
    ):
        return A2C(
            policy=policy,
            env=env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/A2C", # Used to display output of execution in notebook, no need to alter
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    @staticmethod
    def train_model(model, tb_log_name, iter_num, total_timesteps=5000):
        model = model.learn( # Function belonging to stablebaselines, returns a trained model
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        model.save(
            f"{config.TRAINED_MODEL_DIR}/A2C2_{total_timesteps // 1000}k_{iter_num}" # Saves to a model to a file so that it can be called during execution
            # I hardcoded A2C2, but you can call that model name specified by string in parameters
        )
        return model

    @staticmethod
    def get_validation_sharpe(iteration): # This function is only needed for ensembling, ignore during model development portion
        """Calculate Sharpe ratio based on validation results"""
        df_total_value = pd.read_csv(
            f"results/account_value_validation_A2C2_{iteration}.csv"
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
    ):
        self.df = df # Stock data
        self.train_period = train_period # tuple containing training start_date and end_date, starts 2010-2021
        self.val_test_period = val_test_period # tuple: 2021-2023

        self.unique_trade_date = df[
            (df.date > val_test_period[0]) & (df.date <= val_test_period[1])
        ].date.unique() # trims dataframe to test dates
        self.rebalance_window = rebalance_window # 63 days
        self.validation_window = validation_window # 63 days

        self.stock_dim = stock_dim # 29 stocks
        self.hmax = hmax # constrains buying and selling to 100 shares
        self.initial_amount = initial_amount # $10,000 
        self.buy_cost_pct = buy_cost_pct # transaction fee
        self.sell_cost_pct = sell_cost_pct # selling fee 
        self.reward_scaling = reward_scaling # trims dollar reward amount by 1e-4
        self.state_space = state_space # 175: 29 stocks (close price, stock values, MACD, RSI, CCI, ADX) + total portfolio balance = 175
        self.action_space = action_space # 29: 1 action per stock (must be less than 100 bc constrained by hmax)
        self.tech_indicator_list = tech_indicator_list # (MACD, RSI, CCI, ADX): finance indicators used in state space - good metrics to drive actions
        self.print_verbosity = print_verbosity
        self.train_env = None  # defined in train_validation() function

    # SKIP DOWN TO run_A2C_strategy() FUNCTION!!!!!!!!!!!!

    def DRL_validation(self, model, test_data, test_env, test_obs):
        """validation process"""
        for _ in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs) # Native to stablebaselines A2C, need to generate our own
            test_obs, rewards, dones, info = test_env.step(action) # Runs through validation timeframe (63 days) and records results in saved file (see env_stocktrading.py)

    def DRL_prediction(
        self, model, last_state, iter_num, turbulence_threshold, initial
    ):
        """make a prediction based on trained model"""

        # trading env
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[iter_num - self.rebalance_window], # Starts after validation period (training end date + 63)
            end=self.unique_trade_date[iter_num], # Ends training end date + 128
        )
        trade_env = DummyVecEnv( # Creates trading environment mirroring validation environment
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
                    model_name="A2C2",
                    mode="trade",
                    iteration=iter_num,
                    print_verbosity=self.print_verbosity,
                )
            ]
        )

        trade_obs = trade_env.reset()

        for i in range(len(trade_data.index.unique())): # Loops through each day in timeframe
            action, _states = model.predict(trade_obs) # Gets actions
            trade_obs, rewards, dones, info = trade_env.step(action) # Deploys actions and returns results (see env_stocktrading.py)
            if i == (len(trade_data.index.unique()) - 2): # Second to last trading day
                last_state = trade_env.envs[0].render() # Saves state from first environment in DummyVecEnv wrapper on that day, can use render() in ours

        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(f"results/last_state_A2C2_{i}.csv", index=False)
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

        print(f"======A2C2 Training========")
        model = self.get_model(
            self.train_env, policy="MlpPolicy", model_kwargs=model_kwargs
        ) # Generates model, see above function
        model = self.train_model( # See above function
            model,
            tb_log_name=f"A2C2_{i}",
            iter_num=i,
            total_timesteps=timesteps, # 10,000
        )  
        print(
            f"======A2C2 Validation from: ",
            validation_start_date,
            "to ",
            validation_end_date,
        )
        val_env = DummyVecEnv( # Creates duplicate environment for validation
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
                    iteration=i, # This is import for model retrieval - includes iteration in file path
                    model_name='A2C2', # This is also important for model retrieval - includes this string in file path
                    mode="validation",
                    print_verbosity=self.print_verbosity,
                )
            ]
        )
        val_obs = val_env.reset()
        self.DRL_validation( # See above function
            model=model,
            test_data=validation,
            test_env=val_env,
            test_obs=val_obs,
        )
        sharpe = self.get_validation_sharpe(i) # Gets sharpe ratio from validation iteration - only needed for ensembling so leave alone
        print(f"A2C2 Sharpe Ratio: ", sharpe)
        sharpe_list.append(sharpe)
        return model, sharpe_list, sharpe # Returns model and sharpe values, used for model selection in ensembling - leave alone

    def run_A2C_strategy(
        self,
        model_kwargs,
        timesteps,
    ):
        
        # Model Sharpe Ratios
        sharpe_list = []
        sharpe = -1

        """Strategy for A2C"""
        print("============Start A2C2 Strategy============")
        # For ensemble model, it's necessary to feed the last state of the previous model to the current model as the initial state
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

            # IGNORE TURBULENCE INDEX SECTION
            # -----------------------------------------------------------------------------------------------------

            # Tuning trubulence index based on historical data
            # Turbulence lookback window is one quarter (63 days)
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
                # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
                # then we assume that the current market is volatile,
                # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
                # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
                turbulence_threshold = insample_turbulence_threshold
            else:
                # if the mean of the historical data is less than the 90% quantile of insample turbulence data
                # then we tune up the turbulence_threshold, meaning we lower the risk
                turbulence_threshold = np.quantile(
                    insample_turbulence.turbulence.values, 1
                )

            turbulence_threshold = np.quantile(
                insample_turbulence.turbulence.values, 0.99
            )
            print("turbulence_threshold: ", turbulence_threshold)

            # -----------------------------------------------------------------------------------------------

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

            # Environment Setup ends

            # Training and Validation starts
            print(
                "======Model training from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )
            # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
            # print("==============Model Training===========")
            # Train Each Model

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
            # Training and Validation ends

            # Trading starts
            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.rebalance_window],
                "to ",
                self.unique_trade_date[i],
            )
            # print("Used Model: ", model_ensemble)
            last_state_ensemble = self.DRL_prediction( # See function above
                model=model,
                last_state=last_state_ensemble, # This is especially important for ensembling because models needs access to other models final states
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
            ) 
            # Trading ends

        end = time.time()
        print("Strategy took: ", (end - start) / 60, " minutes")

        df_summary = pd.DataFrame(
            [
                iteration_list,
                validation_start_date_list,
                validation_end_date_list,
                sharpe_list,
            ]
        ).T
        df_summary.columns = [
            "Iter",
            "Val Start",
            "Val End",
            "Model Used",
            "A2C Sharpe",
        ]

        return df_summary # Returns sharpe and valuation information
