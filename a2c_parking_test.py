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

from a2c.a2c import A2CAgent

output_path = "./A2C/"
output_prefix = "fixed-goal_p-1_"

env = gym.make('custom-parking-v0', render_mode="rgb_array")
env.configure({"parking_angles" : [0, 0],
               "fixed_goal" : 2,
               "reward_p" : 1})

env.reset()

# Get number of actions from gym action space
n_actions = len(env.action_space.sample())

# Get the number of state observations
obs, info = env.reset()
n_observations = len(np.concatenate((obs['observation'], obs['desired_goal'])))

n_episodes = 1000
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

success_episodes = []

for episode in range(n_episodes):
    print(f"Episode {episode}")
    
    done = False
    
    obs, info = env.reset()
    state = np.concatenate((obs['observation'], obs['desired_goal']))
    
    total_reward = 0
    
    # Run episode
    while not done:
        if episode+1 in eps_to_record:
            frames.append(env.render())
            ep_nr.append(episode+1)
        
        # Select action
        action = agent.select_action(state)
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = np.concatenate((obs['observation'], obs['desired_goal']))
        
        # Optimise agent
        agent.optimise(state, action, reward, next_state, terminated, truncated)
        
        state = next_state
        done = terminated or truncated
        total_reward += reward
        
        if info['is_success']:
            success_episodes.append(episode)
        
    cumul_reward_list.append(total_reward)
    

print("# episodes goal achieved: ", len(success_episodes))

# Plot total rewards
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.arange(1, len(cumul_reward_list)+1), cumul_reward_list)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.scatter(success_episodes, np.ones(len(success_episodes))*10, color='green')
fig.suptitle(f"Gamma: {gamma}; Learning rate: {lr};  Neural net size: {hidden_size}\nSuccessful Episodes: {len(success_episodes)}")
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
