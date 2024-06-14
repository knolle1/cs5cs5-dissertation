# -*- coding: utf-8 -*-
"""
A2C implementation

@author: kimno
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np

class A2CAgent:
    
    def __init__(self, n_obs, n_actions, max_steps=1, gamma=0.99, lr_actor=1e-4, lr_critic=1e-4, hidden_size=128,
                 continous_action = False, log_std_init=0.0):
        
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        
        self.continous_action = continous_action
        
        if self.continous_action:
            log_std = log_std_init * np.ones(self.n_actions, dtype=np.float32)
            # Add it to the list of parameters
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)
        
        self.actor = nn.Sequential(nn.Linear(self.n_obs, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.n_actions)
                                  ).double()

        self.critic = nn.Sequential(nn.Linear(self.n_obs, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_size, 1)
                                   ).double()
        
        # Init optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Init lists and gradients for optimisation
        self.state_values = []
        self.action_logprobs = []
        self.rewards = []
        
        
        
    def get_policy(self, state):
        """
        Returns distribution of actions from policy given the state
        """
        state = torch.Tensor(state).double()
        return self.actor(state)
    
    def get_state_value(self, state):
        """
        Returns value of the state
        """
        state = torch.Tensor(state).double()
        return self.critic(state)
        
        
    def select_action(self, state):
        """
        Select action based on policy
        """
        state = torch.Tensor(state).double()
        
        """
        # Get distribution
        probs = self.get_policy(state)
        
        # Sample action from distribution
        # "logits argument will be interpreted as unnormalized log probabilities 
        # and can therefore be any real number. It will likewise be normalized 
        # so that the resulting probabilities sum to 1 along the last dimension."
        action = Categorical(logits=probs).sample().item()
        """
        if self.continous_action:
            mu = self.get_policy(state)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)
        else:
            probs = self.get_policy(state)
            dist = Categorical(logits=probs)
    
        action = dist.sample()
        
        return action
    
    def optimise(self, state, action, reward, next_state, terminated, truncated):
        # Get state value
        state_val = torch.squeeze(self.critic(torch.Tensor(state).double()).view(-1))
        #print(state, state_val)
        self.state_values.append(state_val)
        
        # Get log probability of action given the state
        if self.continous_action:
            mu = self.get_policy(state)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)
            action_logprob = dist.log_prob(action).sum(axis=-1)
        else:
            probs = self.get_policy(state)
            dist = Categorical(logits=probs)
            action_logprob = dist.log_prob(action)
        self.action_logprobs.append(action_logprob)
        
        # Get reward
        #print(reward)
        self.rewards.append(reward)
        
        
        # Update weights when terminal state reached 
        if terminated or truncated or (len(self.state_values) >= self.max_steps):
            
            # Init actual rewards
            if terminated:
                R = 0
            else:
                # Estimate using value of next state
                next_state = torch.Tensor(next_state).double()
                R = self.critic(next_state).item()
                
            # Calculate actual rewards by iterating backwards through the transitions
            G = []
            for i in range(len(self.state_values)-1, -1, -1):
                R = self.rewards[i] + self.gamma * R
                G = [torch.Tensor(np.array(R)).double()] + G
                
            # Turn array of tensors into single tensor
            self.action_logprobs = torch.stack(tuple(self.action_logprobs), 0)
            G = torch.stack(tuple(G), 0)
            self.state_values = torch.stack(tuple(self.state_values), 0)
            
            # Reset optimisers
            self.critic_optim.zero_grad()
            self.actor_optim.zero_grad()
            
            # Calculate actor gradient
            # Detach state values so that this gradient isn't backpropagated through the critic
            advantage = G - self.state_values.detach()
            # Use negative loss because optimiser does gradient descent and 
            # here we want to do gradient ascent
            actor_loss = -(torch.sum(self.action_logprobs * advantage))
            
            # Calculate critic loss
            critic_loss = torch.nn.MSELoss()(G, self.state_values)
            
            # Backpropagate losses and optimise parameters
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
            
            # Reset lists and gradients
            self.state_values = []
            self.action_logprobs = []
            self.rewards = []