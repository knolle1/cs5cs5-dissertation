# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:46:08 2024

PPO implementation from: https://github.com/realwenlongwang/PPO-Single-File-Notebook-Implementation

@author: kimno
"""

from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical, Normal
#import gymnasium as gym
#from tqdm.notebook import tnrange
import numpy as np
import scipy
#import wandb
from gymnasium.spaces import Box, Discrete
import os
import random
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.wrappers import NormalizeObservation
import pandas as pd
import copy 

from line_profiler import profile

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length, shape=None):
    """
    Helper function makes sure the shape of experience is correct for the buffer

    Args:
        length (int): _description_
        shape (tuple[int,int], optional): _description_. Defaults to None.

    Returns:
        tuple[int,int]: correct shape
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

# TODO: This buffer cannot recompute GAE. Maybe change it in the future
class PPOBuffer():
    """
    A buffer to store the rollout experience from OpenAI spinningup
    """
    def __init__(self, observation_dim, action_dim, capacity, gamma, lam):
        self.obs_buf = np.zeros(combined_shape(capacity, observation_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(capacity, action_dim), dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.rtg_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.idx = 0
        self.path_idx = 0
        self.gamma = gamma
        self.lam = lam

    def push(self, obs, act, rew, val, logp):
        assert self.idx < self.capacity
        self.obs_buf[self.idx] = obs
        self.act_buf[self.idx] = act
        self.rew_buf[self.idx] = rew
        self.val_buf[self.idx] = val
        self.logp_buf[self.idx] = logp

        self.idx += 1

    def GAE_cal(self, last_val):
        """Calculate the GAE when an episode is ended

        Args:
            last_val (int): last state value, it is zero when the episode is terminated.
            it's v(s_{t+1}) when the state truncate at t.
        """
        path_slice = slice(self.path_idx, self.idx)
        # to make the deltas the same dim
        rewards = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rewards[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        ### OpenAI spinning up implemetation comment: No ideal, big value loss when episode rewards are large
        # self.rtg_buf[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]

        ### OpenAI stable_baseline3 implementation
        ### in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        ### TD(lambda) estimator, see "Telescoping in TD(lambda)"
        self.rtg_buf[path_slice] = self.adv_buf[path_slice] + self.val_buf[path_slice]
        
        self.path_idx = self.idx

                
    def sample(self, minibatch_size, device):
        """This method sample a list of minibatches from the memory

        Args:
            minibatch_size (int): size of minibatch, usually 2^n
            device (object): CPU or GPU

        Returns:
            list: a list of minibatches
        """
        assert self.idx == self.capacity, f'The buffer is not full, \
              self.idx:{self.idx} and self.capacity:{self.capacity}'
        # normalise advantage
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / (np.std(self.adv_buf) + 1e-8)
        
        inds = np.arange(self.capacity)
        
        np.random.shuffle(inds)
        
        data = []
        for start in range(0, self.capacity, minibatch_size):
            end = start + minibatch_size
            minibatch_inds = inds[start:end]
            minibatch = dict(obs=self.obs_buf[minibatch_inds], act=self.act_buf[minibatch_inds], \
                             rtg=self.rtg_buf[minibatch_inds], adv=self.adv_buf[minibatch_inds], \
                             logp=self.logp_buf[minibatch_inds])
            data.append({k: torch.as_tensor(v, dtype=torch.float32, device=device) for k,v in minibatch.items()})
        
        return data
    
    def reset(self):
        # reset the index
        self.idx, self.path_idx = 0, 0
        
def layer_init(layer, std=np.sqrt(2)):
    """Init the weights as the stable baseline3 so the performance is comparable.
       But it is not the ideal way to initialise the weights.

    Args:
        layer (_type_): layers
        std (_type_, optional): standard deviation. Defaults to np.sqrt(2).

    Returns:
        _type_: layers after init
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer

class Actor_Net(nn.Module):
    def __init__(self, n_observations, n_actions, num_cells, continous_action, log_std_init=0.0, layer_num=3):
        super(Actor_Net, self).__init__()
        
        self.continous_action = continous_action
        self.action_dim = n_actions
        
        layers = []
        input_dim = n_observations
        for _ in range(layer_num):
            layers.append(layer_init(nn.Linear(input_dim, num_cells)))
            layers.append(nn.LayerNorm(num_cells))
            layers.append(nn.Tanh())
            input_dim = num_cells
        layers.append(layer_init(nn.Linear(num_cells, n_actions), std=0.01))
        
        self.network = nn.Sequential(*layers)

        if self.continous_action:
            log_std = log_std_init * np.ones(self.action_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)

    def forward(self, x):
        return self.network(x)
    
    def act(self, x):
        if self.continous_action:
            mu = self.forward(x)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)
        else:
            log_probs = F.log_softmax(self.forward(x), dim=1)
            dist = Categorical(log_probs)
    
        action = dist.sample()
        if self.continous_action:
            action_logprob = dist.log_prob(action).sum(axis=-1)
        else:
            action_logprob = dist.log_prob(action)

        return action, action_logprob
    
    def logprob_ent_from_state_acton(self, x, act):
        if self.continous_action:
            mu = self.forward(x)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)
            act_logp = dist.log_prob(act).sum(axis=-1) 
        else:
            dist = Categorical(F.softmax(self.forward(x), dim=1))
            act_logp = dist.log_prob(act)
        entropy = dist.entropy()
        
        return entropy, act_logp
    
   
class Critic_Net(nn.Module):
    def __init__(self, n_observations, num_cells, layer_num=3):
        super(Critic_Net, self).__init__()
        
        layers = []
        input_dim = n_observations
        for _ in range(layer_num):
            layers.append(layer_init(nn.Linear(input_dim, num_cells)))
            layers.append(nn.LayerNorm(num_cells))
            layers.append(nn.Tanh())
            input_dim = num_cells
        layers.append(layer_init(nn.Linear(num_cells, 1), std=1.0))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Actor_Critic_net(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, continous_action, continous_observation, parameters_hardshare, log_std_init=0.0, layer_num=3):
        super(Actor_Critic_net, self).__init__()

        self.parameters_hardshare = parameters_hardshare
        self.continous_action = continous_action
        self.continous_observation = continous_observation
        self.action_dim = act_dim
        self.obs_dim = obs_dim

        if self.parameters_hardshare:
            layers = []
            input_dim = obs_dim
            for _ in range(layer_num):
                layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Tanh())
                input_dim = hidden_dim

            self.shared_network = nn.Sequential(*layers)
            self.actor_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
            self.critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
            
            if self.continous_action:
                log_std = log_std_init * np.ones(self.action_dim, dtype=np.float32)
                self.log_std = torch.nn.Parameter(torch.as_tensor(log_std), requires_grad=True)
        else:
            self.actor = Actor_Net(obs_dim, act_dim, hidden_dim, continous_action, log_std_init, layer_num)
            self.critic = Critic_Net(obs_dim, hidden_dim, layer_num)

    def forward(self, x):
        if not self.continous_observation:
            x = F.one_hot(x.long(), num_classes=self.obs_dim).float()
        if self.parameters_hardshare:
            x = self.shared_network(x)
            actor_logits = self.actor_head(x)
            value = self.critic_head(x)
        else:
            actor_logits = self.actor(x)
            value = self.critic(x)
        return actor_logits, value

    def get_value(self, x):
        if not self.continous_observation:
            x = F.one_hot(x.long(), num_classes=self.obs_dim).float()
        return self.critic(x).item()

    def act(self, x, deterministic=False):
        if self.continous_action:
            mu, value = self.forward(x)
            log_std = self.log_std if self.parameters_hardshare else self.actor.log_std
            std = torch.exp(log_std)
            dist = Normal(mu, std)
        else:
            actor_logits, value = self.forward(x)
            log_probs = F.log_softmax(actor_logits, dim=-1)
            dist = Categorical(log_probs)

        if deterministic:
            action = dist.mode
        else:
            action = dist.sample()
            
        if self.continous_action:
            action_logprob = dist.log_prob(action).sum(axis=-1)
        else:
            action_logprob = dist.log_prob(action)

        return action, action_logprob, value

    def logprob_ent_from_state_acton(self, x, action):
        if self.continous_action:
            mu, value = self.forward(x)
            log_std = self.log_std if self.parameters_hardshare else self.actor.log_std
            std = torch.exp(log_std)
            dist = Normal(mu, std)
            action_logp = dist.log_prob(action).sum(axis=-1)
        else:
            actor_logits, value = self.forward(x)
            log_probs = F.log_softmax(actor_logits, dim=-1)
            dist = Categorical(log_probs)
            action_logp = dist.log_prob(action)
        entropy = dist.entropy().sum(axis=-1)

        return entropy, action_logp, value
    
class PPO():
    def __init__(self, gamma, lamb, eps_clip, K_epochs, \
                 observation_space, action_space, num_cells, layer_num,\
                 actor_lr, critic_lr, memory_size , minibatch_size,\
                 max_training_iter, cal_total_loss, c1, c2, \
                 early_stop, kl_threshold, parameters_hardshare, \
                 max_grad_norm , device, ewc_lambda=0, ewc_discount=0.9
                 ):
        """Init

        Args:
            gamma (float): discount factor of future value
            lamb (float): lambda factor from GAE from 0 to 1
            eps_clip (float): clip range, usually 0.2
            K_epochs (in): how many times learn from one batch
            action_space (tuple[int, int]): action space of environment
            num_cells (int): how many cells per hidden layer
            critic_lr (float): learning rate of the critic
            memory_size (int): the size of rollout buffer
            minibatch_size (int): minibatch size
            cal_total_loss (bool): add entropy loss to the actor loss or not
            c1 (float): coefficient for value loss
            c2 (float): coefficient for entropy loss
            kl_threshold (float): approx kl divergence, use for early stop
            parameters_hardshare (bool): whether to share the first two layers of actor and critic
            device (_type_): tf device
            ewc_lambda (float) : weight of EWC penalty
            ewc_discount (float) : discount factor for updating online EWC

        """
        self.gamma = gamma
        self.lamb = lamb
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.max_training_iter = int(max_training_iter) # Cast to int incase it is float

        self.observation_space = observation_space
        self.action_space = action_space
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        
        self.cal_total_loss = cal_total_loss
        self.c1 = c1
        self.c2 = c2
        self.early_stop = early_stop
        self.kl_threshold = kl_threshold

        self.parameters_hardshare = parameters_hardshare
        self.episode_count = 1
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

        self._last_obs = None
        self._episode_reward = 0
        self._early_stop_count = 0

        if isinstance(action_space, Box):
            self.continous_action = True
        elif isinstance(action_space, Discrete):
            self.continous_action = False
        else:
            raise AssertionError(f"action space is not valid {action_space}")

        if isinstance(observation_space, Box):
            self.continous_observation = True
        elif isinstance(observation_space, Discrete):
            self.continous_observation = False
        else:
            raise AssertionError(f"observation space is not valid {observation_space}")        

        self.actor_critic = Actor_Critic_net(
            observation_space.shape[0] if self.continous_observation else observation_space.n, 
            action_space.shape[0] if self.continous_action else action_space.n, 
            num_cells, self.continous_action, self.continous_observation, parameters_hardshare, layer_num=layer_num).to(device)

        if parameters_hardshare:
            ### eps=1e-5 follows stable-baseline3
            self.actor_critic_opt = torch.optim.Adam(self.actor_critic.parameters(), lr=actor_lr, eps=1e-5)
            
        else:
            self.actor_critic_opt = torch.optim.Adam([ 
                {'params': self.actor_critic.actor.parameters(), 'lr': actor_lr, 'eps' : 1e-5},
                {'params': self.actor_critic.critic.parameters(), 'lr': critic_lr, 'eps' : 1e-5} 
            ])


        self.memory = PPOBuffer(observation_space.shape, action_space.shape, memory_size, gamma, lamb)

        self.device = device
        
        # Init EWC variables
        self.ewc_lambda = ewc_lambda
        self.ewc_discount = ewc_discount
        self.ewc_saved_params = {n: p.data.clone() for n, p in 
                                 self.actor_critic.actor.named_parameters()}
        self.ewc_importance = {n: torch.zeros_like(p).to(p.device) for n, p in 
                               self.actor_critic.actor.named_parameters()}
        
        # These two lines monitor the weights and gradients
        #wandb.watch(self.actor_critic.actor, log='all', log_freq=1000, idx=1)
        #wandb.watch(self.actor_critic.critic, log='all', log_freq=1000, idx=2)
        # wandb.watch(self.actor_critic, log='all', log_freq=1000)

    def roll_out(self, env, eval_frequ=None, eval_render=False, eval_envs={}, 
                 output_dir=None, run_id=None):
        """rollout for experience

        Args:
            env (gymnasium.Env): environment from gymnasium
        """
        
        assert self._last_obs is not None, "No previous observation"
        
        action_shape = env.action_space.shape
        # Run the policy for T timestep
        for i in range(self.memory_size):
            with torch.no_grad():
                obs_tensor = torch.tensor(self._last_obs, \
                                        dtype=torch.float32, device=self.device).unsqueeze(0)
            
                action, action_logprob, value = self.actor_critic.act(obs_tensor)
            if self.continous_action:
                action = action.cpu().numpy().reshape(action_shape)
            else:
                action = action.item()

            action_logprob = action_logprob.item()

            value = value.item()

            ### Clipping actions when they are reals is important
            clipped_action = action

            if self.continous_action:
                clipped_action = np.clip(action, self.action_space.low, self.action_space.high)

            next_obs, reward, terminated, truncated, info = env.step(clipped_action)

            self.global_step += 1

            self.memory.push(self._last_obs, action, reward, value, action_logprob)

            self._last_obs = next_obs

            self._episode_reward += reward

            if terminated or truncated:
                if truncated:
                    with torch.no_grad():
                        input_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device) if self.continous_observation else torch.tensor(next_obs, dtype=torch.long, device=self.device)
                        last_value = self.actor_critic.get_value(input_tensor)
                else:
                    last_value = 0
                
                self.memory.GAE_cal(last_value)

                self._last_obs, _ = env.reset()
                
                self.episode_count += 1

                #wandb.log({'episode_reward' : self._episode_reward}, step=self.global_step)
                self.episode_log["episode_reward"].append(self._episode_reward)
                self.episode_log["success"].append('is_success' in info.keys() and info['is_success'])
                self.episode_log["crashed"].append('is_crashed' in info.keys() and info['is_crashed'])
                self.episode_log["truncated"].append(truncated)

                self._episode_reward = 0
                
            if eval_frequ is not None and (self.global_step % eval_frequ) == 0:
                # Evaluate current agent
                self.evaluate(output_dir=output_dir, run_id=run_id, 
                              render=eval_render, eval_envs=eval_envs, deterministic=True)

        with torch.no_grad():
            input_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device) if self.continous_observation else torch.tensor(next_obs, dtype=torch.long, device=self.device)
            last_value = self.actor_critic.get_value(input_tensor)
        self.memory.GAE_cal(last_value)


    def evaluate_recording(self, env, directory, run_id):
        self.actor_critic.eval()

        video_folder = os.path.join(directory, 'videos')

        env = RecordVideo(env, video_folder, name_prefix=f"run-{run_id}")

        obs, _ = env.reset()

        done = False

        action_shape = env.action_space.shape

        while not done:
            obs_tensor = torch.tensor(obs, \
                                    dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = self.actor_critic.act(obs_tensor)

            if self.continous_action:
                action = action.cpu().numpy().reshape(action_shape)
            else:
                action = action.item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = next_obs

        
        #mp4_files = [file for file in os.listdir(video_folder) if file.endswith(".mp4")]
        #
        #for mp4_file in mp4_files:
        #    wandb.log({'Episode_recording': wandb.Video(os.path.join(video_folder, mp4_file))})

        env.close()
        


    def compute_loss(self, data):
        """compute the loss of state value, policy and entropy

        Args:
            data (List[Dict]): minibatch with experience

        Returns:
            actor_loss : policy loss
            critic_loss : value loss
            entropy_loss : mean entropy of action distribution
        """
        observations, actions, logp_old = data['obs'], data['act'], data['logp']
        advs, rtgs = data['adv'], data['rtg']

        # Calculate the pi_theta (a_t|s_t)
        entropy, logp, values = self.actor_critic.logprob_ent_from_state_acton(observations, actions)
        ratio = torch.exp(logp - logp_old)
        # Kl approx according to http://joschu.net/blog/kl-approx.html
        kl_apx = ((ratio - 1) - (logp - logp_old)).mean()
    
        clip_advs = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advs
        # Torch Adam implement tation mius the gradient, to plus the gradient, we need make the loss negative
        actor_loss = -(torch.min(ratio*advs, clip_advs)).mean()

        values = values.flatten() # I used squeeze before, maybe a mistake

        critic_loss = F.mse_loss(values, rtgs)
        # critic_loss = ((values - rtgs) ** 2).mean()

        entropy_loss = entropy.mean()

        return actor_loss, critic_loss, entropy_loss, kl_apx        

    def optimise(self):

        entropy_loss_list = []
        actor_loss_list = []
        critic_loss_list = []
        kl_approx_list = []
        ewc_penalty_list = []
        
        # for _ in tnrange(self.K_epochs, desc=f"epochs", position=1, leave=False):
        for _ in range(self.K_epochs):
            
            # resample the minibatch every epochs
            data = self.memory.sample(self.minibatch_size, self.device)
            
            for minibatch in data:
            
                actor_loss, critic_loss, entropy_loss, kl_apx = self.compute_loss(minibatch)

                entropy_loss_list.append(-entropy_loss.item())
                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                kl_approx_list.append(kl_apx.item())
                
                # Calculate EWC penalty
                if self.ewc_lambda > 0:
                    ewc_penalty = torch.tensor(0).float().to(self.device)
                    for n, p in self.actor_critic.actor.named_parameters():
                        ewc_penalty += (self.ewc_importance[n] * (p - self.ewc_saved_params[n]).pow(2)).sum()
                    ewc_penalty_list.append(ewc_penalty.item())

                if self.cal_total_loss:
                    total_loss = (actor_loss + 
                                  self.c1 * critic_loss - 
                                  self.c2 * entropy_loss + 
                                  self.ewc_lambda * ewc_penalty)

                ### If this update is too big, early stop and try next minibatch
                if self.early_stop and kl_apx > self.kl_threshold:
                    self._early_stop_count += 1
                    ### OpenAI spinning up uses break as they use fullbatch instead of minibatch
                    ### Stable-baseline3 uses break, which is questionable as they drop the rest
                    ### of minibatches.
                    continue
                
                self.actor_critic_opt.zero_grad()
                if self.cal_total_loss:
                    total_loss.backward()
                    # Used by stable-baseline3, maybe more important for RNN
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.actor_critic_opt.step()

                else:
                    actor_loss.backward()
                    critic_loss.backward()
                    # Used by stable-baseline3, maybe more important for RNN
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.actor_critic_opt.step()
                    
        if self.ewc_lambda > 0:
            # Compute Fisher information for EWC loss
            data = self.memory.sample(1, self.device)
            fisher = {n: torch.zeros_like(p).to(p.device) for n, p in 
                          self.actor_critic.actor.named_parameters()} 
            for d in data:
                self.actor_critic_opt.zero_grad()
                actor_loss, _, _, _ = self.compute_loss(d)
                actor_loss.backward()
                
                for n, p in self.actor_critic.actor.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.data.clone().pow(2) / len(data)
            
            # Update EWC importance
            with torch.no_grad():
                for n, imp in self.ewc_importance.items():
                    self.ewc_importance[n] = self.ewc_discount * imp + fisher[n]
            
            # Update saved optimised parameters
            self.ewc_saved_params = {n: p.data.clone() for n, p in 
                                     self.actor_critic.actor.named_parameters()}
            
            self.loss_log['ewc_penalty'].append(np.mean(ewc_penalty_list))
        
        
        self.memory.reset()    
        # Logging, use the same metric as stable-baselines3 to compare performance
        #with torch.no_grad():
        #    if self.continous_action:
        #        mean_std = np.exp(self.actor_critic.actor.log_std.mean().item())
        #        wandb.log({'mean_std': mean_std})
        #
        #wandb.log(
        #    {
        #        'actor_loss': np.mean(actor_loss_list),
        #        'critic_loss' : np.mean(critic_loss_list),
        #        'entropy_loss' : np.mean(entropy_loss_list),
        #        'KL_approx' : np.mean(kl_approx_list)
        #    }, step=self.global_step
        #)
        #if self.early_stop:
        #    wandb.run.summary['early_stop_count'] = self._early_stop_count 
        
        self.loss_log['actor_loss'].append(np.mean(actor_loss_list)),
        self.loss_log['critic_loss'].append(np.mean(critic_loss_list)),
        self.loss_log['entropy_loss'].append(np.mean(entropy_loss_list)),
        self.loss_log['KL_approx'].append(np.mean(kl_approx_list)),
        self.loss_log['step'].append(self.global_step)

                
    def train(self, env, output_dir=None, run_id=None, 
              eval_frequ=10_000, eval_render=False, eval_envs={}, save_results=True):
        self.actor_critic.train()
        
        # Reset logging dictionaries
        self.episode_log = {'episode_reward' : [],
                            'success' : [],
                            'crashed' : [],
                            'truncated' : []}
        self.loss_log = {'step' : [],
                         'actor_loss': [],
                         'critic_loss' : [],
                         'entropy_loss' : [],
                         'KL_approx' : [],
                         "ewc_penalty" : [],
                         }
        self.eval_log = {}
        for label in eval_envs.keys():
            self.eval_log[label] = {'step' : [],
                                    'episode_reward' : [],
                                    'success' : [],
                                    'crashed' : [],
                                    'truncated' : []}

        env.reset_global_step()
        self._last_obs, _ = env.reset()

        #for i in tnrange(self.max_training_iter // self.memory_size):
        for i in range(self.max_training_iter // self.memory_size):
            print("Iteration", i)
            self.roll_out(env, eval_frequ=eval_frequ, eval_render=eval_render, 
                          output_dir=output_dir, run_id=run_id, eval_envs=eval_envs)

            self.optimise()
            
        if save_results:
            # Save results
            if output_dir is not None:
                
                train_dir = os.path.join(output_dir, "train")
                
                if not os.path.exists(train_dir):
                    print(f"Creating output directory {train_dir}")
                    os.makedirs(train_dir)
                    
                metric_list = ["episode_reward", "success", "crashed", "truncated",
                               "actor_loss", "critic_loss", "entropy_loss", "KL_approx"]
                if self.ewc_lambda > 0:
                    metric_list.append("ewc_penalty")
                
                for metric in metric_list:
                    
                    if metric in ["episode_reward", "success", "crashed", "truncated"]:
                        df = pd.DataFrame(self.episode_log)[[metric]]
                        df = df.reset_index(names="episode")
                        df["episode"] = pd.to_numeric(df["episode"])
                        join_var = "episode"
                    else:
                        df = pd.DataFrame(self.loss_log)[["step", metric]]
                        join_var = "step"
                        
                    df = df.rename(columns={metric : f"run_{run_id}"})
                        
                    filename = os.path.join(train_dir, f"{metric}_results.csv")
                    
                    if os.path.isfile(filename):
                        df_existing = pd.read_csv(filename)
                        df = pd.merge(df_existing, df, on=join_var, how="outer")
                        
                    df.to_csv(filename, index=False)
                
                for label in eval_envs.keys():
                    
                    eval_dir = os.path.join(output_dir, "evaluate", label)
                        
                    if not os.path.exists(eval_dir):
                        print(f"Creating output directory {eval_dir}")
                        os.makedirs(eval_dir)
                    
                    for metric in ["episode_reward", "success", "crashed", "truncated"]:
                        
                        df = pd.DataFrame(self.eval_log[label])[["step", metric]]
                        join_var = "step"
                            
                        df = df.rename(columns={metric : f"run_{run_id}"})
                            
                        filename = os.path.join(eval_dir, f"{metric}_results.csv")
                        
                        if os.path.isfile(filename):
                            df_existing = pd.read_csv(filename)
                            df = pd.merge(df_existing, df, on=join_var, how="outer")
                            
                        df.to_csv(filename, index=False)
    
            # save the model to the wandb run folder
            # PATH = os.path.join(wandb.run.dir, "actor_critic.pt")
            # torch.save(self.actor_critic.state_dict(), PATH)
    
    
            #wandb.run.summary['total_episode'] = self.episode_count
        
    def evaluate(self, output_dir=None, run_id=None, render=False, eval_envs={},
                 deterministic=False):
        
        print("evaluating")
        for label, env in eval_envs.items():
            if render:
                video_folder = os.path.join(output_dir, 'evaluate', label, 'videos')
                eval_env = RecordVideo(env, video_folder, name_prefix=f"run-{run_id}_step-{self.global_step}_")
            else:
                eval_env = copy.deepcopy(env)
    
            obs, _ = eval_env.reset()
            done = False
            action_shape = eval_env.action_space.shape
            total_reward = 0
    
            while not done:
                obs_tensor = torch.tensor(obs, \
                                        dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = self.actor_critic.act(obs_tensor, deterministic=True)
    
                action = action.cpu().numpy()
                action = action.reshape(action_shape)
                next_obs, reward, terminated, truncated, info = eval_env.step(action)
                
                #filename = os.path.join(output_dir, 'evaluate', label, f"run-{run_id}.csv").replace("/", "\\")
                #if not os.path.isfile(filename):
                #    with open(filename, 'w') as f:
                #        f.write("step;success;crashed;goal;achieved;goal hit;distance;test success\n")
                #with open(filename, 'a') as f:
                #    f.write(f"{self.global_step};{info['is_success']};{info['is_crashed']};{info['goal']};{info['achieved']};{info['goal_hit']};{info['reward']};{info['test_success']}".replace("\n", "") + "\n")
                    
                total_reward += reward
                done = terminated or truncated
                obs = next_obs
    
            self.eval_log[label]["step"].append(self.global_step)
            self.eval_log[label]["episode_reward"].append(total_reward)
            self.eval_log[label]["success"].append('is_success' in info.keys() and info['is_success'])
            self.eval_log[label]["crashed"].append('is_crashed' in info.keys() and info['is_crashed'])
            self.eval_log[label]["truncated"].append(truncated)
    
            eval_env.close()
            
            #if label == "vertical" and self.global_step >= 400_000:
            #    a = 1