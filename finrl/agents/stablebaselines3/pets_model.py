from __future__ import annotations

import time

import numpy as np
import pandas as pd
import random
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])
            except BaseException as inner_error:
                self.logger.record(key="train/reward", value=None)
                print("Original Error:", error)
                print("Inner Error:", inner_error)
        return True

class DynamicsModel(nn.Module):
    """Predicts the next state and reward given the current state and action."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DynamicsModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1),  # Predict next state and reward
        )

    def forward(self, state, action):
        # print('state: ', state.shape)
        # print('action: ', action.shape)
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)

class PETSAgent:   
    @staticmethod
    def collect_data(env, num_steps=10000):
        data = []
        state, = env.reset()
        # print('state reset: ', state)
        for _ in range(num_steps):
            action = env.action_space.sample()
            action = np.expand_dims(action, axis=0)
            # print('action space: ', action)
            next_state, reward, _, _ = env.step(action)
            next_state = next_state[0]
            reward = reward[0]
            action = action[0]
            # print('next state: ', next_state)
            # print('reward: ', reward)
            data.append((state, action, reward, next_state))
            state = next_state
        return data

    @staticmethod
    def train_dynamics_model(data, models, optimizers, criterion, num_epochs, batch_size=32):
        states, actions, rewards, next_states = zip(*data)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        dataset = TensorDataset(states, actions, rewards, next_states)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for batch in dataloader:
                s, a, r, ns = batch
                for model, optimizer in zip(models, optimizers):
                    pred = model(s, a)
                    pred_next_state, pred_reward = pred[:, :-1], pred[:, -1:]
                    loss = criterion(pred_next_state, ns) + criterion(pred_reward, r)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        return models

    @staticmethod
    def plan_action(state, models, horizon, action_dim, num_samples=100, num_elites=10, num_iterations=5):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        action_mean = torch.zeros(horizon, action_dim)
        action_std = torch.ones(horizon, action_dim)*0.5

        best_action = None
        best_reward = -float('inf')

        for _ in range(num_iterations):
            sampled_actions = torch.normal(action_mean.unsqueeze(0).repeat(num_samples, 1, 1), 
                                action_std.unsqueeze(0).repeat(num_samples, 1, 1)).clamp(-1, 1)
            rewards = []
            for actions in sampled_actions:
                cumulative_reward = 0
                simulated_state = state.squeeze(1)
                for t in range(horizon):
                    simulated_action = actions[t].unsqueeze(0)

                    model = random.choice(models)
                    pred = model(simulated_state, simulated_action)
                    pred_next_state, pred_reward = pred[:, :-1], pred[:, -1]
                    
                    cumulative_reward += pred_reward.item()
                    simulated_state = pred_next_state
                
                rewards.append(cumulative_reward)

            rewards = torch.tensor(rewards)
            elite_indices = rewards.argsort(descending=True)[:num_elites]
            elite_actions = sampled_actions[elite_indices]

            action_mean = elite_actions.mean(dim=0)
            action_std = elite_actions.std(dim=0)

            if rewards.max() > best_reward:
                best_reward = rewards.max()
                best_action = elite_actions[0]

        return best_action[0].detach().numpy()


    @staticmethod
    def get_validation_sharpe(iteration): # This function is only needed for ensembling, ignore during model development portion
        df_total_value = pd.read_csv(
            f"results/account_value_validation_PETS_{iteration}.csv"
        )
        if df_total_value["daily_return"].var() == 0:
            if df_total_value["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            return (
                (252**0.5)
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
        hidden_dim=64,
        lr = 1e-4,
        epochs=100,
        horizon=10,
        ensemble_size=5
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
        self.hidden_dim = hidden_dim
        self.learning_rate = lr
        self.num_epochs = epochs
        self.rollout_horizon = horizon
        self.ensemble_size = ensemble_size
        self.dynamics_models = [
            DynamicsModel(self.state_space, self.action_space, hidden_dim)
            for _ in range(ensemble_size)
        ]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for model in self.dynamics_models
        ]
        self.criterion = nn.MSELoss()

    def DRL_validation(self, models, test_data, test_env, test_obs):
        for _ in range(len(test_data.index.unique())):
            action = self.plan_action(test_obs, models, self.rollout_horizon, self.action_space)
            action = np.expand_dims(action, axis=0)
            test_obs, rewards, dones, info = test_env.step(action)
            test_obs = test_obs[0]
            if dones:
                break
    
    def DRL_prediction(
        self, models, last_state, iter_num, turbulence_threshold, initial
    ):
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[iter_num - self.rebalance_window], # Starts after validation period (training end date + 63)
            end=self.unique_trade_date[iter_num], # Ends training end date + 128
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
                    model_name="PETS",
                    mode="trade",
                    iteration=iter_num,
                    print_verbosity=self.print_verbosity,
                )
            ]
        )

        trade_obs = trade_env.reset()

        for i in range(len(trade_data.index.unique())): # Loops through each day in timeframe
            action = self.plan_action(trade_obs, models, self.rollout_horizon, self.action_space)
            action = np.expand_dims(action, axis=0)
            test_obs, rewards, dones, info = trade_env.step(action)
            test_obs = test_obs[0]
            if i == (len(trade_data.index.unique()) - 2): # Second to last trading day
                last_state = trade_env.envs[0].render() # Saves state from first environment in DummyVecEnv wrapper on that day, can use render() in ours

        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(f"results/last_state_PETS_{i}.csv", index=False)
        return last_state

    def _train_window(
        self,
        sharpe_list,
        validation_start_date,
        validation_end_date,
        timesteps,
        i,
        validation,
        turbulence_threshold,
    ):
        print(f"======PETS Training (PETS)========")
        train_data = self.collect_data(self.train_env)
        models = self.train_dynamics_model(
                train_data,
                self.dynamics_models,
                self.optimizers,
                self.criterion,
                self.num_epochs,
            )
        print(
            f"======PETS Validation from: ",
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
                    model_name='PETS',
                    mode="validation",
                    print_verbosity=self.print_verbosity,
                )
            ]
        )
        val_obs = val_env.reset()
        self.DRL_validation(
            models,
            test_data=validation,
            test_env=val_env,
            test_obs=val_obs,
        )
        sharpe = self.get_validation_sharpe(i)
        print(f"PETS Sharpe Ratio: ", sharpe)
        sharpe_list.append(sharpe)

        return models, sharpe_list, sharpe


    def run_strategy(
        self,
        timesteps,
    ):
        sharpe_list = []
        sharpe = -1
        """Strategy for PETS"""
        print("============Start PETS Strategy============")
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
            # -----------------------------------------------------------------------------------------------
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
            print(
                "======Model training from: ",
                self.train_period[0],
                "to ",
                self.unique_trade_date[
                    i - self.rebalance_window - self.validation_window
                ],
            )

            models, sharpe_list, sharpe = self._train_window( # Where actual training occurs, see above function
                sharpe_list,
                validation_start_date,
                validation_end_date,
                timesteps,
                i,
                validation,
                turbulence_threshold,
            )
            print(
                "======Trading from: ",
                self.unique_trade_date[i - self.rebalance_window],
                "to ",
                self.unique_trade_date[i],
            )
            last_state_ensemble = self.DRL_prediction( # See function above
                models=models,
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
            "PETS Sharpe"
        ]

        return df_summary