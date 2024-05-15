# -*- coding: utf-8 -*-
"""
Code to test the modified parking environment

Created on Sun May 12 13:39:30 2024

@author: kimno
"""

import gymnasium as gym
import highway_env
highway_env.register_highway_envs()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

env = gym.make('custom-parking-v0', render_mode="rgb_array")
env.configure({"parking_angles" : [90, 90],
               "fixed_goal" : 2})
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

for i in range(1):
    env.reset()
    fig, ax = plt.subplots()
    ax.imshow(env.render())
    ax.axis('off')
