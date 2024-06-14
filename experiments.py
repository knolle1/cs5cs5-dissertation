# -*- coding: utf-8 -*-
"""
Code for running all experiments

Created on Tue Jun 11 11:14:22 2024

@author: kimno
"""
import gymnasium as gym
import highway_env
highway_env.register_highway_envs()

from agent.random import random_rollout
import train_eval

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from timeit import default_timer as timer

"""
# Run random baselines
# -----------------------------------------------------------------------------
env = gym.make('custom-parking-v0')
env.configure({"parking_angles" : [0, 0],
               "fixed_goal" : 2})
print("Running random_vertical")
for i in range(2,5):
    print(f"Run {i}")
    start_time = timer()
    random_rollout(1_000_000, env, "./results/random_vertical", run_id=i)
    end_time = timer()
    print("Time elapsed: " + str(end_time - start_time) + " s")

env = gym.make('custom-parking-v0')
env.configure({"parking_angles" : [-25, 25],
               "fixed_goal" : 2})
print("Running random_diagonal-25")
for i in range(2,5):
    print(f"Run {i}")
    start_time = timer()
    random_rollout(1_000_000, env, "./results/random_diagonal-25", run_id = i)
    end_time = timer()
    print("Time elapsed: " + str(end_time - start_time) + " s")
 

env = gym.make('custom-parking-v0')
env.configure({"parking_angles" : [-50, 50],
               "fixed_goal" : 2})
print("Running random_diagonal-50")
for i in range(1,5):
    print(f"Run {i}")
    start_time = timer()
    random_rollout(1_000_000, env, "./results/random_diagonal-50", run_id = i)
    end_time = timer()
    print("Time elapsed: " + str(end_time - start_time) + " s")

env = gym.make('custom-parking-v0')
env.configure({"parking_angles" : [90, 90],
               "fixed_goal" : 2})
print("Running random_parallel")
for i in range(1,5):
    print(f"Run {i}")
    start_time = timer()
    random_rollout(1_000_000, env, "./results/random_parallel", run_id = i)
    end_time = timer()
    print("Time elapsed: " + str(end_time - start_time) + " s")

    
"""
# Test vanilla PPO on parking environment
# -----------------------------------------------------------------------------
env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2}
ppo_params = {"gamma" : 0.99,
              "lamb" : 0.95,
              "eps_clip" : 0.2,
              "max_training_iter" : 1_000_000,
              "K_epochs" : 10,
              "num_cells" : 64,
              "actor_lr" : 1e-4,
              "critic_lr" : 1e-4,
              "memory_size" : 2048,
              "minibatch_size" : 64,
              "c1" : 0.5,
              "c2" : 0,
              "kl_threshold" : 0.15,
              "parameters_hardshare" : False,
              "early_stop" : False,
              "cal_total_loss" : False,
              "max_grad_norm" : 0.5
              }
experiment_params = {"output_dir" : "./results/eval_test_adj-weights_0.05_thresh_0.01",
                     "baseline_dir" : "./results/random_vertical",
                     "eval_envs" : {"vertical" : {"parking_angles" : [0, 0],
                                                  "fixed_goal" : 2},
                                    "diagonal-25" : {"parking_angles" : [-25, 25],
                                                     "fixed_goal" : 2},
                                    "diagonal-50" : {"parking_angles" : [-50, 50],
                                                     "fixed_goal" : 2},
                                    "parallel" : {"parking_angles" : [90, 90],
                                                  "fixed_goal" : 2},
                                    },
                     "render_eval" : True,
                     "plot" : True,
                     "n_runs" : 1}
train_eval.main(env_params, ppo_params, experiment_params)
