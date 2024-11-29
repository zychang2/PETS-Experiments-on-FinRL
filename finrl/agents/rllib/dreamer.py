# Rllib DreamerV3 integration
# https://github.com/ray-project/ray/blob/master/rllib/algorithms/dreamerv3/README.md
# hyperparameters: https://arxiv.org/pdf/2301.04104v1
# inference example: `dreamerv3_inference.py`
# https://docs.ray.io/en/latest/rllib/rllib-rlmodule.html#rlmodule-guide
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from pprint import pprint

class DRLAgent:
    """Implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class / registered env_creator name
        env_config: dict, configuration for setting up the environment
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, env_config):
        # self.price_array = price_array
        # self.tech_array = tech_array
        # self.turbulence_array = turbulence_array
        # env_config = {
        #     "price_array": self.price_array,
        #     "tech_array": self.tech_array,
        #     "turbulence_array": self.turbulence_array,
        #     "if_train": True,
        # }
        self.env = env
        self.config = (
            DreamerV3Config()
            # .api_stack( # Dreamer v3 is not compatible with old API stack
            #     enable_rl_module_and_learner=True,
            #     enable_env_runner_and_connector_v2=True,
            # )
            .environment(env=self.env, env_config=env_config)
            .training(
                model_size="S",
                training_ratio=4, # can tune this
            )
            .env_runners(
                num_env_runners=1,
            )
            # .learners(
            #     num_learners=0,
            #     num_gpus_per_learner=1,
            # )
        )
        self.algorithm = self.config.build()

    def get_model(self):
        return self.algorithm

    def train_model(self, model, total_episodes=5, verbose=False):
        # we can also use tuner API here
        # https://docs.ray.io/en/latest/rllib/rllib-training.html#using-the-python-api
        for _ in range(total_episodes):
            result = model.train()
            if verbose:
                pprint(result)
        return model

    @staticmethod
    def DRL_prediction(model, env):
        # test on the testing env
        state = env.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [env.initial_total_asset]
        done = False
        while not done:
            action = model.compute_single_action(state)
            state, reward, done, _ = env.step(action)

            total_asset = (
                env.amount
                + (env.price_ary[env.day] * env.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / env.initial_total_asset
            episode_returns.append(episode_return)

        print("episode return: " + str(episode_return))
        print("Test Finished!")
        return episode_total_assets
