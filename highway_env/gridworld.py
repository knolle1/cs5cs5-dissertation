# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:18:37 2024

@author: kimno
"""
import numpy as np
import random
import gymnasium
from gymnasium import spaces

class Gridworld:
    """
    Gridworld environment
    """
    def __init__(self, n, goal, max_steps = 200):
        # Setup environment (nxn grid world with start at [0, 0] and end square specified by goal)
        self.goal = goal
        self.n = n
        self.max_steps = max_steps
        self.n_actions = 4
        
        self.grid = np.full((self.n, self.n), -1)
        self.grid[goal[0]][goal[1]] = 100
        
        self.observation_space = spaces.Box(0, n - 1, shape=(2,), dtype=int)
        self.action_space = gymnasium.spaces.Discrete(n)
        
        self.reset()
        
    def reset(self):
        self.agent_xpos = random.randint(0, self.n - 1)
        self.agent_ypos = random.randint(0, self.n - 1)
        self.steps = 0
        return [self.agent_xpos, self.agent_ypos], None
    
    def step(self, direction):
        # Take action
        # Note: bottom left corner is coordinates [0, 0]
        if direction == 0 and self.agent_xpos < (self.n - 1):  # right
            self.agent_xpos += 1
        elif direction == 1 and self.agent_ypos < (self.n - 1):  # up
            self.agent_ypos += 1
        elif direction == 2 and self.agent_xpos > 0:  # left
            self.agent_xpos -= 1
        elif direction == 3 and self.agent_ypos > 0:    # down
            self.agent_ypos -= 1
        
        self.steps += 1
        
        # Get reward
        reward = self.grid[self.agent_xpos][self.agent_ypos]
        
        # Determine if done
        if (self.agent_xpos == self.goal[0] and self.agent_ypos == self.goal[1]):
            terminated = True
            truncated = False
        elif self.steps >= self.max_steps:
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False
        
        # For compatibility with gymnasium envs
        info = None
        
        return np.array([self.agent_xpos, self.agent_ypos]), reward, terminated, truncated, info