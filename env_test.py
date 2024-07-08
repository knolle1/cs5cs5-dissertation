# -*- coding: utf-8 -*-
"""
Code to test the modified parking environment

Created on Sun May 12 13:39:30 2024

@author: kimno
"""
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import gymnasium as gym
import highway_env
highway_env.register_highway_envs()


from gymnasium.wrappers.record_video import RecordVideo

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from agent.random import random_rollout

env = gym.make('custom-parking-v0', render_mode="rgb_array")
env.configure({"parking_angles" : [-25, 25],
               "fixed_goal" : [(0, 3), (0, -4), (1, 3), (1, -4)],
               "vehicles_count" : 50,
               "reward_p" : 0.5,
               "env_change_config" : [{"parking_angles" : [0, 0]},
                                      {"parking_angles" : [-25, 25]},
                                      {"parking_angles" : [-50, 50]},
                                      {"parking_angles" : [90, 90]}],
               "env_change_frequency" : 100})

video_folder = "./video_env_test"
env = RecordVideo(env, video_folder, name_prefix=f"test")

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())


for i in range(500):
    #fig, ax = plt.subplots()
    #ax.imshow(env.render())
    #ax.axis('off')
    
    obs, reward, terminated, truncated, info = env.step([3, 0])
    
    if terminated or truncated:
        fig, ax = plt.subplots()
        ax.imshow(env.render())
        ax.axis('off')
        obs, info = env.reset()

env.close()
"""    
    x.append(obs[0])
    y.append(reward)
    
    print(info["is_success"], info["is_crashed"])
    
fig, ax = plt.subplots()
ax.plot(range(len(y)), y)

fig, ax = plt.subplots()
ax.plot(range(len(x)), x)
    
#random_rollout(1_000, env, "./results/random", 0)
"""