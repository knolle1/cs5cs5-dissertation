# -*- coding: utf-8 -*-
"""
Code to test the modified parking environment using A2C

Created on Sun May 12 13:39:30 2024

@author: kimno
"""

import gymnasium as gym
import highway_env
highway_env.register_highway_envs()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import pandas as pd
import os

from a2c.a2c import A2CAgent

output_path = "./A2C/"
output_prefix = "04_const-start_collision-100_success-0.05_"

env = gym.make('custom-parking-v0', render_mode="rgb_array")
env.configure({"parking_angles" : [0, 0],
               "fixed_goal" : 2,
               "reward_p" : 1})

env.reset()

# Get number of actions from gym action space
n_actions = len(env.action_space.sample())

# Get the number of state observations
obs, info = env.reset()
n_observations = len(obs)

n_episodes = 5000
gamma = 0.99
lr = 1e-4
hidden_size = 256 #128

agent = A2CAgent(n_obs=n_observations, 
                 n_actions=n_actions, 
                 gamma=gamma, 
                 lr_actor=lr, lr_critic=lr, 
                 hidden_size=hidden_size,
                 continous_action=True)

cumul_reward_list = []

frames = []
ep_nr = []
eps_to_record = [1, 5, 10, 50, 100, 200, 500, 800, 900, 1000]

success_frames = []

crashed_episodes = []
success_episodes = []
truncated_episodes = []

df = pd.DataFrame()

for episode in range(n_episodes):
    print(f"Episode {episode}")
    
    done = False
    
    obs, info = env.reset()
    state = np.array(obs)
    
    total_reward = 0
    j = 0
    # Run episode
    while not done:
        if episode+1 in eps_to_record:
            frames.append(env.render())
            ep_nr.append(episode+1)
        
        # Select action
        action = agent.select_action(state)
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action.numpy(force=True))
        next_state = np.array(obs)
        
        # Optimise agent
        agent.optimise(state, action, reward, next_state, terminated, truncated)
        
        state = next_state
        done = terminated or truncated
        total_reward += reward
         
        j += 1
        
    if info['is_success']:
        success_frames.append(env.render())
    
    tmp = pd.DataFrame({"episode" : [episode],
                        "success" : [info['is_success']],
                        "crashed" : [info['is_crashed']],
                        "achieved" : [info['achieved']],
                        "goal" : [info['goal']],
                        "info_reward" : [info['reward']],
                        "last_reward" : [reward],
                        "total_reward" : [total_reward]})
    
    df = pd.concat([df, tmp])
            
    print("truncated", truncated, "success", info['is_success'], "steps", j, "total reward", total_reward)
      
    success_episodes.append(int(info['is_success']))
    crashed_episodes.append(int(info['is_crashed']))
    truncated_episodes.append(int(truncated))
    cumul_reward_list.append(total_reward)
    
    
df.to_csv(f"{output_path + output_prefix}_results.csv")
   
fig, ax = plt.subplots(figsize=(10, 10), nrows=4)
ax[0].plot(np.convolve(cumul_reward_list, np.ones(100)/100, mode="valid"))
ax[0].set_ylabel("Reward")
ax[1].plot(np.convolve(truncated_episodes, np.ones(100), mode="valid"))
ax[1].set_ylabel("Truncated Episodes")
ax[2].plot(np.convolve(success_episodes, np.ones(100), mode="valid"))
ax[2].set_ylabel("Success Episodes")
ax[3].plot(np.convolve(crashed_episodes, np.ones(100), mode="valid"))
ax[3].set_ylabel("Crashed Episodes")
fig.suptitle("Rolling averages of 100 episodes")
fig.tight_layout()
fig.savefig(f"{output_path + output_prefix}_rolling_avg_n-episodes-{n_episodes}_gamma-{gamma}_lr-{lr}_nn-size-{hidden_size}.png", format="png")
    
if not os.path.exists(f"{output_path}/{output_prefix}_sucess_frames"):
    os.makedirs(f"{output_path}/{output_prefix}_sucess_frames")
for i in range(1, 11):
    if len(success_frames) >= i:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(success_frames[-i])
        fig.savefig(f"{output_path}/{output_prefix}_sucess_frames/success_frame_{i}")

print("# episodes goal achieved: ", len(success_episodes))

# Plot total rewards
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.arange(1, len(cumul_reward_list)+1), cumul_reward_list)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.scatter(np.where(success_episodes)[0], np.ones(len(np.where(success_episodes)[0]))*10, color='green')
ax.scatter(np.where(crashed_episodes)[0], np.ones(len(np.where(crashed_episodes)[0]))*10, color='red')
fig.suptitle(f"Gamma: {gamma}; Learning rate: {lr};  Neural net size: {hidden_size}\nSuccessful Episodes: {sum(success_episodes)}")
fig.tight_layout()
fig.savefig(f"{output_path + output_prefix}_n-episodes-{n_episodes}_gamma-{gamma}_lr-{lr}_nn-size-{hidden_size}.png", format="png")

fig, ax = plt.subplots(figsize=(10, 5))
patch = ax.imshow(frames[0])
label = ax.text(0, 0, f"Episode 1", ha='center', va='center', fontsize=20, color="Red")
plt.axis('off')
    
def animate(i):
    patch.set_data(frames[i])
    label.set_text(f"Episode {ep_nr[i]}")
        
ani = animation.FuncAnimation(fig, animate, repeat=True, frames=len(frames) - 1, interval=50)

# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=60,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
ani.save(f"{output_path + output_prefix}_animation.gif", writer=writer)
