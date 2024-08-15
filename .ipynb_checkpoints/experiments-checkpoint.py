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
"""
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
for i in range(2,5):
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
for i in range(2,5):
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
for i in range(2,5):
    print(f"Run {i}")
    start_time = timer()
    random_rollout(2_000_000, env, "./results/random_parallel", run_id = i)
    end_time = timer()
    print("Time elapsed: " + str(end_time - start_time) + " s")
"""

# Test and tune parking environment using PPO
# -----------------------------------------------------------------------------

#train_eval.main(config="./config/tuning/0_default.json")
#train_eval.main(config="./config/tuning/1_p-norm_1.json")
#train_eval.main(config="./config/tuning/2_collision-reward_-100.json")
#train_eval.main(config="./config/tuning/3_success-threshold_0.03.json")
#train_eval.main(config="./config/tuning/4_angle-weights_0.05.json")
#train_eval.main(config="./config/tuning/5_hyperparams.json")

#train_eval.main(config="./config/tuning/6.1_reward-function.json")
#train_eval.main(config="./config/tuning/6.2_reward-function_p-0.5.json")
#train_eval.main(config="./config/tuning/6.3_reward-function_p-0.5_col-rwd--25.json")
#train_eval.main(config="./config/tuning/6.4_reward-function_p-0.5_success-0.12.json")

#train_eval.main(config="./config/tuning/7_hyperparameters_id-3.json")
#train_eval.main(config="./config/tuning/7_hyperparameters_id-6.json")
#train_eval.main(config="./config/tuning/7_hyperparameters_id-19.json")
#train_eval.main(config="./config/tuning/7_hyperparameters_id-26.json")
#train_eval.main(config="./config/tuning/7_hyperparameters_id-49.json")

#train_eval.main(config="./config/baseline/single_diagonal-25.json")
#train_eval.main(config="./config/baseline/single_diagonal-50.json")
#train_eval.main(config="./config/ewc_test.json")

"""
ppo_params = {"gamma" : 0.99,
              "lamb" : 0.95,
              "eps_clip" : 0.2,
              "max_training_iter" : 500_000,
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

# Default environment
env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -5,
              "reward_p" : 0.5,
              "collision_reward_factor" : 0,
              "success_goal_reward" : 0.12,
              "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02]
              }

experiment_params = {"output_dir" : "./results/test_envs/0_default",
                     "baseline_dir" : "./results/random_vertical",
                     "eval_envs" : {"vertical" : {"parking_angles" : [0, 0],
                                                  "fixed_goal" : 2},
                                    #"diagonal-25" : {"parking_angles" : [-25, 25],
                                    #                 "fixed_goal" : 2},
                                    #"diagonal-50" : {"parking_angles" : [-50, 50],
                                    #                 "fixed_goal" : 2},
                                    #"parallel" : {"parking_angles" : [90, 90],
                                    #              "fixed_goal" : 2},
                                    },
                     "render_eval" : True,
                     "plot" : True,
                     "seed" : 12345,
                     "n_runs" : 0}

train_eval.main(env_params, ppo_params, experiment_params)

# Change to p=1
env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -5,
              "reward_p" : 1,
              "collision_reward_factor" : 0,
              "success_goal_reward" : 0.12,
              "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02]
              }
experiment_params["output_dir"] = "./results/test_envs/1_p-norm_1"
train_eval.main(env_params, ppo_params, experiment_params)

# Increase collision penalty
env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -100,
              "reward_p" : 1,
              "collision_reward_factor" : 0,
              "success_goal_reward" : 0.12,
              "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02]
              }
experiment_params["output_dir"] = "./results/test_envs/2_collision-reward_-100"
train_eval.main(env_params, ppo_params, experiment_params)

# Reduce success threshold
env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -100,
              "reward_p" : 1,
              "collision_reward_factor" : 0,
              "success_goal_reward" : 0.03,
              "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02]
              }
experiment_params["output_dir"] = "./results/test_envs/3_success-threshold_0.03"
train_eval.main(env_params, ppo_params, experiment_params)

# Increase weights of angle observations
env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -100,
              "reward_p" : 1,
              "collision_reward_factor" : 0,
              "success_goal_reward" : 0.03,
              "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              }
experiment_params["output_dir"] = "./results/test_envs/4_angle-weights_0.05"
train_eval.main(env_params, ppo_params, experiment_params)

# Test tuned hyperparameters
ppo_params = {"gamma" : 0.99,
              "lamb" : 0.9,
              "eps_clip" : 0.3,
              "max_training_iter" : 500_000,
              "K_epochs" : 10,
              "num_cells" : 128,
              "actor_lr" : 1e-4,
              "critic_lr" : 1e-4,
              "memory_size" : 2048,
              "minibatch_size" : 32,
              "c1" : 0.5,
              "c2" : 0,
              "kl_threshold" : 0.15,
              "parameters_hardshare" : False,
              "early_stop" : False,
              "cal_total_loss" : False,
              "max_grad_norm" : 1
              }

experiment_params["output_dir"] = "./results/test_envs/5_hyperparams"
train_eval.main(env_params, ppo_params, experiment_params)

# Redefine reward function
env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -10,
              "reward_p" : 1,
              "collision_reward_factor" : 50,
              "success_goal_reward" : 0.03,
              "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              }
experiment_params["output_dir"] = "./results/test_envs/6.1_reward-function"
train_eval.main(env_params, ppo_params, experiment_params)

env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -10,
              "reward_p" : 0.5,
              "collision_reward_factor" : 50,
              "success_goal_reward" : 0.03,
              "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              }
experiment_params["output_dir"] = "./results/test_envs/6.2_reward-function_p-0.5"
train_eval.main(env_params, ppo_params, experiment_params)

env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -25,
              "reward_p" : 0.5,
              "collision_reward_factor" : 50,
              "success_goal_reward" : 0.03,
              "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              }
experiment_params["output_dir"] = "./results/test_envs/6.3_reward-function_p-0.5_col-rwd--25"
train_eval.main(env_params, ppo_params, experiment_params)

env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -10,
              "reward_p" : 0.7,
              "collision_reward_factor" : 50,
              "success_goal_reward" : 0.03,
              "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              }
experiment_params["output_dir"] = "./results/test_envs/6.4_reward-function_p-0.7"
train_eval.main(env_params, ppo_params, experiment_params)

env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -10,
              "reward_p" : 0.5,
              "collision_reward_factor" : 50,
              "success_goal_reward" : 0.12,
              "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              }
experiment_params["output_dir"] = "./results/test_envs/6.5_reward-function_p-0.5_success-0.12"
train_eval.main(env_params, ppo_params, experiment_params)

env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : 2,
              "collision_reward": -25,
              "reward_p" : 0.5,
              "collision_reward_factor" : 50,
              "success_goal_reward" : 0.20,
              "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              }
experiment_params["output_dir"] = "./results/test_envs/6.5_reward-function_p-0.5_success-0.2_seeded_remove-goalhit"
train_eval.main(env_params, ppo_params, experiment_params)


# Test vanilla PPO on parking environment
# -----------------------------------------------------------------------------
env_params = {"parking_angles" : [0, 0],
              "fixed_goal" : [(0, 3), (0, -4), (1, 3), (1, -4)],
        	  "collision_reward": -10,
        	  "reward_p" : 0.5,
           	  "collision_reward_factor" : 50,
        	  "success_goal_reward" : 0.12,
        	  "reward_weights": [1, 0.3, 0, 0, 0.05, 0.05]
              }
ppo_params = {"gamma" : 0.99,
              "lamb" : 0.9,
              "eps_clip" : 0.3,
              "max_training_iter" : 500_000,
              "K_epochs" : 10,
              "num_cells" : 128,
              "actor_lr" : 1e-4,
              "critic_lr" : 1e-4,
              "memory_size" : 2048,
              "minibatch_size" : 32,
              "c1" : 0.5,
              "c2" : 0,
              "kl_threshold" : 0.15,
              "parameters_hardshare" : False,
              "early_stop" : False,
              "cal_total_loss" : False,
              "max_grad_norm" : 1,
              "layer_num" : 2
              }
experiment_params = {"output_dir" : "./results/random_start_goal3",
                     "baseline_dir" : "./results/random_vertical",
                     "eval_envs" : {"vertical" : {"parking_angles" : [0, 0],
                                                  "fixed_goal" : [(0, 3), (0, -4), (1, 3), (1, -4)],
                                                  },
                                    },
                     "render_eval" : True,
                     "plot" : True,
                     "seed" : 12345,
                     "n_runs" : 1}
train_eval.main(env_params, ppo_params, experiment_params)
"""