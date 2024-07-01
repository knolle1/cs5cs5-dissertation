# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:07:35 2024

@author: kimno
"""

from scipy.stats import uniform
from sklearn.model_selection import ParameterSampler
import train_eval

import pandas as pd
import os

env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : [(0, 3), (0, -4), (1, 3), (1, -4)],
        	  "collision_reward": -10,
        	  "reward_p" : 0.5,
           	  "collision_reward_factor" : 50,
        	  "success_goal_reward" : 0.12,
        	  "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              }
experiment_params = {"output_dir" : "./results/hyperparameter_tuning",
                     #"baseline_dir" : "./results/random_vertical",
                     "eval_envs" : {},
                     "render_eval" : False,
                     "plot" : True,
                     "n_runs" : 0
                     }
sweep_configuration = {
        'actor_lr': [1e-4, 3e-4, 1e-3],
        'critic_lr': [1e-4, 3e-4, 1e-3],
        'memory_size': [1024, 2048, 4096],
        'K_epochs': [5, 10, 20],
        'gamma': [0.95, 0.99],
        'lamb': [0.90, 0.95],
        'early_stop': [False],
        'cal_total_loss': [True],
        'parameters_hardshare': [False],
        #'seed': [12345],
        'c1': [0.1, 0.5, 1.0],
        'c2': [0, 0.01, 0.1],
        'minibatch_size': [32, 64, 128],
        'kl_threshold':  [0.15],
        'max_grad_norm': [0.5, 1.0],
        'eps_clip': [0.1, 0.2, 0.3],
        'num_cells': [64, 128, 256], # hidden layer size
        'layer_num': [2, 3, 4], # number of hidden layers
        'max_training_iter': uniform(loc=50_000, scale=250_000),
    }

param_id = 0

experiment_params["seed"] = param_id

for ppo_params in ParameterSampler(sweep_configuration, n_iter=50):

    df_params = pd.DataFrame(ppo_params, index=[param_id])
    
    experiment_params["output_dir"] = f"./results/hyperparameter_tuning/{param_id}"
    train_eval.main(env_params, ppo_params, experiment_params)
    metrics = ["episode_reward", "success", "crashed", "truncated"]
    for i in range(len(metrics)):
        df_metric = pd.read_csv(f"./results/hyperparameter_tuning/{param_id}/train/{metrics[i]}_results.csv")
        df_metric = df_metric.replace({True: 1, False: 0})
        df_metric = df_metric.dropna()
        if metrics[i] == "episode_reward":
            df_metric = df_metric.drop(columns="episode").rolling(100).mean()
        else:
            df_metric = df_metric.drop(columns="episode").rolling(100).sum()
        df_metric['mean'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].mean(axis=1)
        df_metric['std'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].std(axis=1)
        
        df_params[f"{metrics[i]}_mean"] = df_metric["mean"].iloc[-1]
        df_params[f"{metrics[i]}_std"] = df_metric["std"].iloc[-1]
        
    path = f"./results/hyperparameter_tuning/results.csv"
    df_params.to_csv(path, mode='a', header=not os.path.exists(path))
        
    param_id += 1