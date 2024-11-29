**NOTICE:**
Here, we provide several home-grown examples. For more tutorials with different topics, please check the repo at https://github.com/AI4Finance-Foundation/FinRL-Tutorials.

For example, in the [link](https://github.com/AI4Finance-Foundation/FinRL-Tutorials/tree/master/1-Introduction/Stock_NeurIPS2018), you can find not only the notebooks, but also the csv files and a trained agent we provide.


## Installing environment

To set up the environment on my local machine I used conda. I'm using windows 10. For the setup, I tried installing it on a brand new environment:

```
conda create -n finrlenv python=3.10
pip install -e .  // At the root folder (/FinRL)
```

**Note:** There seems to be a problem if using python 3.12 (examples: [link](https://github.com/pygeos/pygeos/issues/463)). My original environment was 3.10 and it worked fine. I have double checked it works with the new environment.

**Note:** This will install a none CUDA version pytorch, getting a CUDA version of pytorch with the following, directly from pytorch website:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Go to either `FinRL_A2C_StockTrading_ICAIF_2020.ipynb` or `FinRL_OptunaTuning_A2C.ipynb`. To run the tuning one, `optuna` need to be installed separately

```
pip install optuna
```

## Installation V2
```
conda create -n finrl python=3.10
conda activate finrl
pip install -e .
pip install optuna dm_tree tabulate
pip install ray
pip install --force-reinstall numpy scipy scikit-learn    
pip install tensorflow

# ray need to be reinstalled after reinstallation dm_tree!
# see https://github.com/ray-project/ray/issues/7645#issuecomment-719865669
# force reinstall is to solve conflicting numpy binary
```

## Notebook running

For `FinRL_A2C_StockTrading_ICAIF_2020.ipynb`, if running locally it would start from **Part 2.3**. I can confirm that the rest of the code can execute.

For `FinRL_OptunaTuning_A2C.ipynb`, start from third block with all the imports. Below I give the breakdown of this notebook:
* **"Collecting data and preprocessing"** section downloads the stock data from `YahooDownloader`. It then uses `FeatureEngineer` from `/finrl/meta/preprocessors/preprocessors.py` to break it down into a dataframe. Stock Dimension is 29, State Space is 291.

* When defining the environments, there were some complications with type errors, it is resolved by defining some of the arguments as list instead of 0. This is according to [this issue](https://github.com/AI4Finance-Foundation/FinRL/issues/540). For the agent I used A2C from stable-baselines3.

* 2 environments are created, `e_train_gym` uses the training dataset and `e_trade_gym` uses the testing dataset.

* After that is the **"Tuning hyperparameters using Optuna"** section. The `sample_..._params` basically defines the set of parameters and the values Optuna will try on. For time sake I only tuned 3 major ones.

* The `LoggingCallback` class will help the study to early stop:
    1. The callback will terminate if the improvement margin is below certain point
    2. It will terminate after certain number of trial_number are reached, not before that
    3. It will hold its patience to reach the threshold

* After the `LoggingCallback` class is the definition of `objective(trial)` for Optuna. The flow is select hyperparams -> get model and train model from DRLAgent -> save model -> do testing/prediction.

* Finally, create the study object and start the optimization, takes a long time even with just 3 hyperparams and GPU.