# Leveraging Model-Based Reinforcement Learning in Stock Trading using FinRL framework

**Note:** Our code is built upon the FinRL framework, an open-source repository designed for stock trading. Link to the original repo [here](https://github.com/AI4Finance-Foundation/FinRL/).

We studied different algorithms not initially presented by the FinRL framework and their effectivness in Dow Jones Index (DJI) trading. We implemented a vanilla model-based reinforcement learning (MBRL) algorithm, as well as MBRL with uncertainty awareness, specifically PETS under stable-baselines specification. We also experimented HER Replay buffer on off-policy model-free algorithm and compared its performance to those without the replay buffer.

## Installation on Local machine

To set up the environment on a local machine, we used conda. The first step is to install it on a brand new environment:

```bash
conda create -n finrlenv python=3.10
conda activate finrlenv
pip install -e .  // At the root folder (/FinRL)
pip install jupyter notebook ipykernel
```

**Note:** There seems to be a problem if using python 3.12 (examples: [link](https://github.com/pygeos/pygeos/issues/463)). We used python 3.10 to work around this issue.

**Note:** This will install a none CUDA version pytorch, getting a CUDA version of pytorch with the following, directly from pytorch website:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

To run the hyperparameter tuning scripts, `optuna` needs to be installed:

```bash
pip install optuna
```

This concludes the environment setup on the local machine.

## Installation on PACE ICE

To setup Georgia Tech PACE environment on a local VSCode, follow this [guide](ssh_compute_node_login_setup.pdf). The rest of the set up is the same as how to install the conda environment on local machine.

## Modifications to files

Our modifications can be break down into two major types: agents implementation and experiment notebooks.

### Agent implementation

* We build agents upon the code in `finrl/agents/stablebaselines3/models.py`, all of the agents implemented by us is under this folder.

* In `a2c_model.py`, we separate the A2C agent out of the ensembel agents in `models.py`.

* In `her_1.py` and `her.py`, we introduced `HerReplayBuffer` from `stablebaselines3` on top of their `DDPG` agent.

* In `mbrl_model.py`, we implemented a vanilla model-based RL agent, where we rollout actions completely at random.

* In `pets_model.py`, we implemented a PETS agent.

* In `single_models.py`, we refractor the code in `models.py` so single agents can easily be called in our experiments notebooks.

* In `modified_pets_model.py`, we have a slightly different version of PETS. For this one, we consider the mean reward of the elite actions instead of argmax.

* In `pets_no_ensemble_model.py`, we implement a version of `pets_model.py`, where the number for dynamics model is 1.

### Experiment notebooks

* All of our experiments are in jupyter notebook format and they are in the `examples/` folder. Because the notebooks in the original repo are also in this folder, we will only list the notebooks created by us below.

* `DRL_A2C.ipynb`: this is a notebook created to run with single A2C agent from `a2c_model.py`, with and without optimized hyperparameters. This serves the purpose to familiarize us with the framework.

* `FinRL_OptunaTuning_A2C.ipynb`: In this notebook, we perform hyperparameter tuning on a A2C agent, using `optuna` library.

* `FinRL_PETS_StockTrading.ipynb` and `FinRL_PETS_StockTrading_pace.ipynb`: Notebooks to validate the PETS implementation.

* `FinRL_MBRL_StockTrading.ipynb`: Notebook to validate the MBRL implementation.

* `FinRL_Final_Project.ipynb`: Notebook where we conduct experiments on all models separately.

* `FinRL_Final_Project_DA.ipynb`: Notebook to calculate and analyze sharpe ratios. To run this notebook, unzip the data folders in `original_results.zip`, `pets_only_results.zip`, `new_pets_results.zip` to the `examples/` folder.

* `FinRL_Final_Project_Ablation.ipynb`: Notebook where we conduct ablation experiments on our implementation of PETS agents.