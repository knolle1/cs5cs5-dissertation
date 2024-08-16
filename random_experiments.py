# -*- coding: utf-8 -*-
"""
Code for running random experiments

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

# Run random baselines
# -----------------------------------------------------------------------------
env = gym.make('custom-parking-v0')
env_params = {"parking_angles" : [0, 0],
        	  "fixed_goal" : [[0, 3], [0, -4], [1, 3], [1, -4]],
		      "collision_reward": -10,
		      "reward_p" : 0.5,
		      "collision_reward_factor" : 50,
		      "success_goal_reward" : 0.12,
		      "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              }
print("Running random_vertical")
for i in range(2,5):
    print(f"Run {i}")
    start_time = timer()
    random_rollout(2_000_000, env, "./results/random_vertical", run_id=i)
    end_time = timer()
    print("Time elapsed: " + str(end_time - start_time) + " s")

env = gym.make('custom-parking-v0')
env.configure({"parking_angles" : [-25, 25],
        	  "fixed_goal" : [[0, 3], [0, -4], [1, 3], [1, -4]],
		      "collision_reward": -10,
		      "reward_p" : 0.5,
		      "collision_reward_factor" : 50,
		      "success_goal_reward" : 0.12,
		      "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              })
print("Running random_diagonal-25")
for i in range(5):
    print(f"Run {i}")
    start_time = timer()
    random_rollout(2_000_000, env, "./results/random_diagonal-25", run_id = i)
    end_time = timer()
    print("Time elapsed: " + str(end_time - start_time) + " s")


env = gym.make('custom-parking-v0')
env.configure({"parking_angles" : [-50, 50],
        	  "fixed_goal" : [[0, 3], [0, -4], [1, 3], [1, -4]],
		      "collision_reward": -10,
		      "reward_p" : 0.5,
		      "collision_reward_factor" : 50,
		      "success_goal_reward" : 0.12,
		      "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              })
print("Running random_diagonal-50")
for i in range(5):
    print(f"Run {i}")
    start_time = timer()
    random_rollout(2_000_000, env, "./results/random_diagonal-50", run_id = i)
    end_time = timer()
    print("Time elapsed: " + str(end_time - start_time) + " s")

env = gym.make('custom-parking-v0')
env.configure({"parking_angles" : [90, 90],
        	  "fixed_goal" : [[0, 3], [0, -4], [1, 3], [1, -4]],
		      "collision_reward": -10,
		      "reward_p" : 0.5,
		      "collision_reward_factor" : 50,
		      "success_goal_reward" : 0.12,
		      "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]})
print("Running random_parallel")
for i in range(5):
    print(f"Run {i}")
    start_time = timer()
    random_rollout(2_000_000, env, "./results/random_parallel", run_id = i)
    end_time = timer()
    print("Time elapsed: " + str(end_time - start_time) + " s")
