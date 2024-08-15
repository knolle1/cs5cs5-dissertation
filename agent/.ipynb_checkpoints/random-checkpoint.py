# -*- coding: utf-8 -*-
"""
Random agent as a baseline
Created on Tue Jun 11 10:42:30 2024

@author: kimno
"""

import numpy as np
import os
import pandas as pd

def random_rollout(n_steps, env, output_dir, run_id):
        
    episode_log = {'episode_reward' : [],
                   'success' : [],
                   'crashed' : [],
                   'truncated' : []}
        
    low = env.action_space.low
    high = env.action_space.high
    shape = env.action_space.shape
        
    env.reset()
    episode_reward = 0
        
    for i in range(n_steps):
            
        action = np.random.uniform(low, high, shape)
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
            
        if terminated or truncated:
            episode_log["episode_reward"].append(episode_reward)
            episode_log["success"].append('is_success' in info.keys() and info['is_success'])
            episode_log["crashed"].append('is_crashed' in info.keys() and info['is_crashed'])
            episode_log["truncated"].append(truncated)

            episode_reward = 0
            
            env.reset()
            
    # Save results
    if output_dir is not None:
        
        if not os.path.exists(output_dir):
            print(f"Creating output directory {output_dir}")
            os.makedirs(output_dir)
        
        for metric in ["episode_reward", "success", "crashed", "truncated"]:
            
            df = pd.DataFrame(episode_log)[[metric]]
            df = df.reset_index(names="episode")
            df["episode"] = pd.to_numeric(df["episode"])
            join_var = "episode"
                
            df = df.rename(columns={metric : f"run_{run_id}"})
                
            filename = os.path.join(output_dir, f"{metric}_results.csv")
            
            if os.path.isfile(filename):
                df_existing = pd.read_csv(filename)
                df = pd.merge(df_existing, df, on=join_var, how="outer")
                
            df.to_csv(filename, index=False)
                
        
                
        
        