# -*- coding: utf-8 -*-
"""
Code to test the PPO implementation with Gridworld, Cart-Pole and Parking environments

Created on Wed May 15 09:36:04 2024

@author: kimno
"""

from agent.ppo import PPO
from highway_env.gridworld import Gridworld
import gymnasium as gym

import highway_env
highway_env.register_highway_envs()

import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

gamma = 0.99
lamb = 0.95
eps_clip = 0.2
max_training_iter = 1_000_000
k_epochs = 10
num_cells = 64
actor_lr = 1e-4 #3e-4 
critic_lr = actor_lr
memory_size = 2048
minibatch_size = 64    
c1 = 0.5
c2 = 0
kl_threshold = 0.15
parameters_hardshare = False
early_stop = False
cal_total_loss = False
max_grad_norm = 0.5
seed = 123456

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
# -----------------------------------------------------------------------------
# Test gridworld

env = Gridworld(5, [4,4])

print("Gridworld")
for i in range(5):
    print("run", i)
    my_ppo = PPO(gamma, lamb, eps_clip, k_epochs, env.observation_space, env.action_space, num_cells,\
                     actor_lr, critic_lr, memory_size, minibatch_size, max_training_iter, \
                     cal_total_loss, c1, c2, early_stop, kl_threshold, parameters_hardshare, max_grad_norm, device)
        
    my_ppo.train(env, "./ppo_test/gridworld", run_id = i)
   
df_results = pd.read_csv("./ppo_test/gridworld/episode_reward_results.csv")
df_results = df_results.dropna()
df_results['mean'] = df_results[[x for x in df_results.columns if x.startswith("run_")]].mean(axis=1)
df_results['std'] = df_results[[x for x in df_results.columns if x.startswith("run_")]].std(axis=1)

fig, ax = plt.subplots()
ax.errorbar(df_results["episode"], df_results["mean"])#, yerr=df_results["std"])
ax.set_xlabel("episode")
ax.set_ylabel("cumulative reward")
fig.suptitle("Gridworld")
fig.savefig("./ppo_test/gridworld/episode_reward_results.png")

# -----------------------------------------------------------------------------
# Test cart-pole

env = gym.make("CartPole-v1", render_mode="rgb_array")
    
print("Cart-Pole")
for i in range(5):
    print("run", i)
    my_ppo = PPO(gamma, lamb, eps_clip, k_epochs, env.observation_space, env.action_space, num_cells,\
                     actor_lr, critic_lr, memory_size, minibatch_size, max_training_iter, \
                     cal_total_loss, c1, c2, early_stop, kl_threshold, parameters_hardshare, max_grad_norm, device)
        
    my_ppo.train(env, "./ppo_test/cart-pole", run_id = i)
   
df_results = pd.read_csv("./ppo_test/cart-pole/episode_reward_results.csv")
df_results = df_results.dropna()
df_results['mean'] = df_results[[x for x in df_results.columns if x.startswith("run_")]].mean(axis=1)
df_results['std'] = df_results[[x for x in df_results.columns if x.startswith("run_")]].std(axis=1)

fig, ax = plt.subplots()
ax.errorbar(df_results["episode"], df_results["mean"])#, yerr=df_results["std"])
ax.set_xlabel("episode")
ax.set_ylabel("cumulative reward")
fig.suptitle("Cart-Pole")
fig.savefig("./ppo_test/cart-pole/episode_reward_results.png")
"""

# -----------------------------------------------------------------------------
# Test parking environment

env = gym.make('custom-parking-v0')
env.configure({"parking_angles" : [0, 0],
               "fixed_goal" : 2,
               "reward_p" : 1})

directory = "./ppo_test/parking_obstacles"

#print("Parking")
#start_time = timer()
#for i in range(1):
#    print("run", i)
#    my_ppo = PPO(gamma, lamb, eps_clip, k_epochs, env.observation_space, env.action_space, num_cells,\
#                     actor_lr, critic_lr, memory_size, minibatch_size, max_training_iter, \
#                     cal_total_loss, c1, c2, early_stop, kl_threshold, parameters_hardshare, max_grad_norm, device)
#        
#    my_ppo.train(env, directory, run_id = i)
#end_time = timer()
#print("Time elapsed for all runs: " + str(end_time - start_time) + " s")


#print("recording evaluation...")
#env_rec = gym.make('custom-parking-v0', render_mode="rgb_array")
#env_rec.configure({"parking_angles" : [0, 0],
#               "fixed_goal" : 2,
#               "reward_p" : 1})
#my_ppo.evaluate_recording(env_rec, directory)
    
print("recording done!")
df_reward = pd.read_csv(f"{directory}/episode_reward_results.csv")
df_reward = df_reward.dropna()
df_reward['mean'] = df_reward[[x for x in df_reward.columns if x.startswith("run_")]].mean(axis=1)
df_reward['std'] = df_reward[[x for x in df_reward.columns if x.startswith("run_")]].std(axis=1)

fig, ax = plt.subplots()
ax.errorbar(df_reward["episode"], df_reward["mean"])#, yerr=df_reward["std"])
ax.set_xlabel("episode")
ax.set_ylabel("cumulative reward")
fig.suptitle("Parking")
fig.savefig(f"{directory}/episode_reward_results.png")

# Calculate rolling averages
metrics = ["episode_reward", "success", "crashed", "truncated"]
fig, ax = plt.subplots(nrows=len(metrics), figsize=(10, 10))
for i in range(len(metrics)):
    df = pd.read_csv(f"{directory}/{metrics[i]}_results.csv").replace({True: 1, False: 0})
    df = df.dropna()
    if metrics[i] == "episode_reward":
        df = df.drop(columns="episode").rolling(100).mean()
    else:
        df = df.drop(columns="episode").rolling(100).sum()
    df['mean'] = df[[x for x in df.columns if x.startswith("run_")]].mean(axis=1)
    df['std'] = df[[x for x in df.columns if x.startswith("run_")]].std(axis=1)
    ax[i].errorbar(df.index, df["mean"], label="PPO")#, yerr=df["std"])
    ax[i].set_ylabel(metrics[i])
ax[-1].set_xlabel("episode")

# Plot ideal value
ax[0].plot(range(len(df)), np.zeros(len(df)), label="Ideal")
ax[1].plot(range(len(df)), np.ones(len(df))*100, label="Ideal")
ax[2].plot(range(len(df)), np.zeros(len(df)), label="Ideal")
ax[3].plot(range(len(df)), np.zeros(len(df)), label="Ideal")
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(f"{directory}/rolling_metrics.png")
