# -*- coding: utf-8 -*-
"""
Code for training and evaluating PPO on the parking environment

Created on Mon Jun 10 17:00:06 2024

@author: kimno
"""

from agent.ppo import PPO
import gymnasium as gym

import highway_env
highway_env.register_highway_envs()

import torch

import argparse
from timeit import default_timer as timer
import tracemalloc
import sys
import os
import json
import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import math

import matplotlib.pyplot as plt

# Format plot labels
# Code from https://stackoverflow.com/questions/59969492/how-to-print-10k-20k-1m-in-the-xlabel-of-matplotlib-plot
def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else math.floor (math.log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 2)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

def create_plots(experiment_params):
    # Set font size for all plots
    plt.rcParams['font.size'] = '16'
        
    # Create plot for episode rewards
    df_results = pd.read_csv(os.path.join(experiment_params["output_dir"], 
                                         "train", 
                                         "episode_reward_results.csv"))
    df_results = df_results.dropna()
    df_results['mean'] = df_results[[x for x in df_results.columns if x.startswith("run_")]].mean(axis=1)
    df_results['std'] = df_results[[x for x in df_results.columns if x.startswith("run_")]].std(axis=1)
    
    if 'baseline_dir' in experiment_params.keys() and experiment_params['baseline_dir'] is not None:
        df_baseline = pd.read_csv(os.path.join(experiment_params["baseline_dir"],
                                             "episode_reward_results.csv"))
        df_baseline = df_baseline.dropna()
        df_baseline['mean'] = df_baseline[[x for x in df_baseline.columns if x.startswith("run_")]].mean(axis=1)
        df_baseline['std'] = df_baseline[[x for x in df_baseline.columns if x.startswith("run_")]].std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_results["episode"], df_results["mean"], label="PPO") # plot mean
    ax.fill_between(df_results["episode"], df_results["mean"]-df_results["std"], 
                    df_results["mean"]+df_results["std"], alpha=0.3) # plot std dev
    xlim = ax.get_xlim()
    if 'baseline_dir' in experiment_params.keys() and experiment_params['baseline_dir'] is not None: # plot random baseline
        ax.plot(df_baseline["episode"], df_baseline["mean"], label="Random")
        ax.fill_between(df_baseline["episode"], df_baseline["mean"]-df_baseline["std"], 
                        df_baseline["mean"]+df_baseline["std"], alpha=0.3)
    ax.plot(df_results["episode"], np.zeros(len(df_results)), label="Ideal") # plot ideal rewards
    ax.set_xlabel("episode")
    ax.set_ylabel("cumulative reward")
    ax.set_xlim(xlim)
    fig.suptitle("Parking")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(experiment_params["output_dir"], "train", "episode_reward_results.png"))
    
    # Create plot for KL approx.
    df_results = pd.read_csv(os.path.join(experiment_params["output_dir"], 
                                         "train", 
                                         "KL_approx_results.csv"))
    df_results = df_results.dropna()
    df_results['mean'] = df_results[[x for x in df_results.columns if x.startswith("run_")]].mean(axis=1)
    df_results['std'] = df_results[[x for x in df_results.columns if x.startswith("run_")]].std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_results["step"], df_results["mean"], label="PPO") # plot mean
    ax.fill_between(df_results["step"], df_results["mean"]-df_results["std"], 
                    df_results["mean"]+df_results["std"], alpha=0.3) # plot std dev
    ax.set_xlabel("step")
    ax.set_ylabel("KL approx")
    #fig.suptitle("Parking")
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(experiment_params["output_dir"], "train", "KL_approx_results.png"))
    

    # Calculate rolling averages
    metrics = ["episode_reward", "success", "crashed", "truncated"]
    fig, ax = plt.subplots(nrows=len(metrics), figsize=(10, 10))
    for i in range(len(metrics)):
        df_metric = pd.read_csv(os.path.join(experiment_params["output_dir"], 
                                             "train",
                                             f"{metrics[i]}_results.csv")).replace({True: 1, False: 0})
        df_metric = df_metric.dropna()
        if metrics[i] == "episode_reward":
            df_metric = df_metric.drop(columns="episode").rolling(100).mean()
        else:
            df_metric = df_metric.drop(columns="episode").rolling(100).sum()
        df_metric['mean'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].mean(axis=1)
        df_metric['std'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].std(axis=1)
        
        if 'baseline_dir' in experiment_params.keys() and experiment_params['baseline_dir'] is not None:
            df_baseline = pd.read_csv(os.path.join(experiment_params["baseline_dir"], 
                                                 f"{metrics[i]}_results.csv")).replace({True: 1, False: 0})
            df_baseline = df_baseline.dropna()
            if metrics[i] == "episode_reward":
                df_baseline = df_baseline.drop(columns="episode").rolling(100).mean()
            else:
                df_baseline = df_baseline.drop(columns="episode").rolling(100).sum()
            df_baseline['mean'] = df_baseline[[x for x in df_baseline.columns if x.startswith("run_")]].mean(axis=1)
            df_baseline['std'] = df_baseline[[x for x in df_baseline.columns if x.startswith("run_")]].std(axis=1)
        
        ax[i].plot(df_metric.index, df_metric["mean"], label="PPO") # plot mean
        ax[i].fill_between(df_metric.index, df_metric["mean"]-df_metric["std"],
                           df_metric["mean"]+df_metric["std"], alpha=0.3) # plot min and max
        xlim = ax[i].get_xlim()
        if 'baseline_dir' in experiment_params.keys() and experiment_params['baseline_dir'] is not None:
            ax[i].plot(df_baseline.index, df_baseline["mean"], label="Random") # plot mean
            ax[i].fill_between(df_baseline.index, df_baseline["mean"]-df_baseline["std"], 
                            df_baseline["mean"]+df_baseline["std"], alpha=0.3) # plot min and max
        if metrics[i] == "success":
            ax[i].plot(range(len(df_metric)), np.ones(len(df_metric))*100, label="Ideal")
        else:
            ax[i].plot(range(len(df_metric)), np.zeros(len(df_metric)), label="Ideal")
        ax[i].set_ylabel(metrics[i])
        ax[i].set_xlim(xlim)
    ax[-1].set_xlabel("episode")
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(experiment_params["output_dir"], "train", "rolling_metrics.png"))
    
    # Create rolling averages for deterministic evaluation
    metrics = ["episode_reward", "success", "crashed", "truncated"]
    fig, ax = plt.subplots(nrows=len(metrics), figsize=(10, 10))
    for i in range(len(metrics)):
        for label in experiment_params["eval_envs"].keys():
            df_metric = pd.read_csv(os.path.join(experiment_params["output_dir"], 
                                                 "evaluate", label,
                                                 f"{metrics[i]}_results.csv")).replace({True: 1, False: 0})
            df_metric = df_metric.dropna()
            #if metrics[i] == "episode_reward":
            #    df_metric = df_metric.drop(columns="step").rolling(100).mean()
            #else:
            #    df_metric = df_metric.drop(columns="step").rolling(100).sum()
            df_metric['mean'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].mean(axis=1)
            df_metric['std'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].std(axis=1)
            
            ax[i].plot(df_metric["step"], df_metric["mean"], label=label) # plot mean
            #ax[i].fill_between(df_metric["step"], df_metric["mean"]-df_metric["std"],
            #                   df_metric["mean"]+df_metric["std"], alpha=0.3) # plot min and max
            #ax[i].errorbar(df_metric["step"], df_metric["mean"], yerr=df_metric["std"], label=label) # plot mean
        if metrics[i] == "success":
            ax[i].plot(df_metric["step"], np.ones(len(df_metric)), label="Ideal")
        else:
            ax[i].plot(df_metric["step"], np.zeros(len(df_metric)), label="Ideal")
        ax[i].set_ylabel(metrics[i])
        ax[i].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax[-1].set_xlabel("step")
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(experiment_params["output_dir"], "evaluate", "rolling_metrics_compare.png"))
    
    # Create rolling averages for deterministic evaluation
    metrics = ["episode_reward", "success", "crashed", "truncated"]
    fig, ax = plt.subplots(nrows=len(metrics), figsize=(10, 10))
    label = "vertical"
    for i in range(len(metrics)):
        df_metric = pd.read_csv(os.path.join(experiment_params["output_dir"], 
                                                 "evaluate", label,
                                                 f"{metrics[i]}_results.csv")).replace({True: 1, False: 0})
        df_metric = df_metric.dropna()
        #if metrics[i] == "episode_reward":
        #    df_metric = df_metric.drop(columns="step").rolling(100).mean()
        #else:
        #    df_metric = df_metric.drop(columns="step").rolling(100).sum()
        df_metric['mean'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].mean(axis=1)
        df_metric['std'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].std(axis=1)
        
        ax[i].plot(df_metric["step"], df_metric["mean"], label=label) # plot mean
        ax[i].fill_between(df_metric["step"], df_metric["mean"]-df_metric["std"],
                               df_metric["mean"]+df_metric["std"], alpha=0.3) # plot min and max
        if metrics[i] == "success":
            ax[i].plot(df_metric["step"], np.ones(len(df_metric)), label="Ideal")
        else:
            ax[i].plot(df_metric["step"], np.zeros(len(df_metric)), label="Ideal")
        ax[i].set_ylabel(metrics[i])
        ax[i].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax[-1].set_xlabel("step")
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(experiment_params["output_dir"], "evaluate", "rolling_metrics.png"))
    

def main(env_params=None, ppo_params=None, experiment_params=None, config=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON file with environment, PPO and experiment parameters', type=str)
    args = parser.parse_args()

    # Check manditory parameters were passed
    if args.config is None and config is None and env_params is None and ppo_params is None and experiment_params is None:
        print("ERROR: Please specify environment, PPO and experiment parameters")
        #sys.exit(1)
        
    # Overwrite parameters with cmd arguments if they were specified
    if args.config is not None:
        config = args.config
        
    # Get parameters from config JSON file
    if config is not None:
        with open(config) as file:
            params = json.load(file)
            ppo_params = params["ppo"]
            env_params = params["env"]
            experiment_params = params["experiment"]    
       
    # Init environment
    env = gym.make('custom-parking-v0')
    env.configure(env_params)
    
    # Init evaluation environments
    eval_envs = {}
    for label, params in experiment_params["eval_envs"].items():
        if experiment_params["render_eval"]:
            eval_envs[label] = gym.make('custom-parking-v0', render_mode="rgb_array")
        else:
            eval_envs[label] = gym.make('custom-parking-v0')
        eval_envs[label].configure(params)
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run training and evaluation
    for i in range(experiment_params["n_runs"]):
        print(f"Run {i}")
        agent = PPO(**ppo_params, device=device, observation_space=env.observation_space, 
                     action_space=env.action_space)
        
        agent.train(env, experiment_params["output_dir"], run_id=i, 
                    eval_frequ=10_000, eval_render=experiment_params["render_eval"], eval_envs=eval_envs)
        
        #agent.evaluate_recording(env_rec, os.path.join(experiment_params["output_dir"], "train"))
    
    if 'plot' in experiment_params.keys() and experiment_params['plot'] is True:
        create_plots(experiment_params)

      
if __name__ == '__main__':
    main()