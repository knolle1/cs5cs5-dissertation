import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import json
import os
import numpy as np
import pandas as pd
import math

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Format plot labels
# Code from https://stackoverflow.com/questions/59969492/how-to-print-10k-20k-1m-in-the-xlabel-of-matplotlib-plot
def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else math.floor (math.log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 2)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

def plot_importance(in_dir, out_dir):
    files = os.listdir(os.path.join(in_dir, "train", "ewc_importances"))
    files = [f for f in files if f.endswith(".json")]
    steps = list(dict.fromkeys([s.split(".")[0].split("step_")[1] for s in files]))
    
    for step in steps:
        weights = None
        n = 0
        for file in files:
            if f"step_{step}" in file:
                f = open(os.path.join(in_dir, "train", "ewc_importances", file))
                data = json.load(f)
                f.close()
    
                # Normalise importances (min max scaling to [0, 1])
                vmax = 0
                for name in data.keys():
                    vmax = max(vmax, np.max(data[name]))
                for name in data.keys():
                    data[name] = data[name] / vmax
    
                # Aggregate across runs
                if weights is None:
                    weights = data
                else:
                    for (weights_n, weights_v), (data_n, data_v) in zip(weights.items(), data.items()):
                        assert (weights_n == data_n)
                        weights[weights_n] = np.array(weights_v) + np.array(data_v)
                n += 1
    
        width_ratios = []
        vmax = 0
        #Get mean and reshape
        for name in weights.keys():
            weights[name] = weights[name] / n
            vmax = max(vmax, np.max(weights[name]))
            if len(weights[name].shape) == 1:
                weights[name] = weights[name].reshape((weights[name].shape[0],1))
            width_ratios.append(weights[name].shape[1])
        width_ratios.append(2)
    
        fig, ax = plt.subplots(ncols = len(weights)+1, figsize=(20, 5), gridspec_kw=dict(width_ratios=width_ratios))
        i=0
        for name in weights.keys():
            sns.heatmap(weights[name], ax=ax[i], vmax=vmax, vmin=0, cbar=False, xticklabels=False, yticklabels=False)
            ax[i].set_title(name, y=1.05, rotation = 90)
            i+=1
        fig.colorbar(ax[1].collections[0], cax=ax[-1])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0)

        prefix = in_dir[2:].replace("/", "_")
        fig.savefig(os.path.join(out_dir, f"{prefix}_ewc-importances-step-{step}.png")) 

def plot_training_curve(in_dir, baseline_dir, out_dir):
    # Set font size for all plots
    plt.rcParams['font.size'] = '16'
    
    # Calculate rolling averages
    metrics = ["episode_reward"]
    fig, ax = plt.subplots(nrows=len(metrics), figsize=(10, 5))
    for i in range(len(metrics)):
        df_metric = pd.read_csv(os.path.join(in_dir, 
                                             "train",
                                             f"{metrics[i]}_results.csv")).replace({True: 1, False: 0})
        df_metric = df_metric.dropna()
        df_metric = df_metric.drop(columns="episode").rolling(100).mean()
        df_metric['mean'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].mean(axis=1)
        df_metric['std'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].std(axis=1)
        
        df_baseline = pd.read_csv(os.path.join(baseline_dir, 
                                                 f"{metrics[i]}_results.csv")).replace({True: 1, False: 0})
        df_baseline = df_baseline.dropna()
        df_baseline = df_baseline.drop(columns="episode").rolling(100).mean()
        df_baseline['mean'] = df_baseline[[x for x in df_baseline.columns if x.startswith("run_")]].mean(axis=1)
        df_baseline['std'] = df_baseline[[x for x in df_baseline.columns if x.startswith("run_")]].std(axis=1)
        
        ax.plot(df_metric.index, df_metric["mean"], label="PPO") # plot mean
        ax.fill_between(df_metric.index, df_metric["mean"]-df_metric["std"],
                           df_metric["mean"]+df_metric["std"], alpha=0.3) # plot min and max
        xlim = ax.get_xlim()
        ax.plot(df_baseline.index, df_baseline["mean"], label="Random") # plot mean
        ax.fill_between(df_baseline.index, df_baseline["mean"]-df_baseline["std"], 
                            df_baseline["mean"]+df_baseline["std"], alpha=0.3) # plot min and max
        ax.plot(range(len(df_metric)), np.zeros(len(df_metric)), label="Ideal")
        ax.set_ylabel("total episode reward")
        ax.set_xlim(xlim)
        ax.minorticks_on()
        ax.grid(which="major")
        ax.grid(which='minor', linestyle=':')
    ax.set_xlabel("episode")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    
    prefix = in_dir[2:].replace("/", "_")
    fig.savefig(os.path.join(out_dir, f"{prefix}_avg-training-curve.png")) 

def plot_outcomes(in_dir, baseline_dir, out_dir):
    # Set font size for all plots
    plt.rcParams['font.size'] = '16'

    # Calculate rolling averages
    metrics = ["success", "crashed", "truncated"]
    fig, ax = plt.subplots(nrows=len(metrics), figsize=(10, 10))
    for i in range(len(metrics)):
        df_metric = pd.read_csv(os.path.join(in_dir, 
                                             "train",
                                             f"{metrics[i]}_results.csv")).replace({True: 1, False: 0})
        df_metric = df_metric.dropna()
        df_metric = df_metric.drop(columns="episode").rolling(100).sum()
        df_metric['mean'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].mean(axis=1)
        df_metric['std'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].std(axis=1)
        
        df_baseline = pd.read_csv(os.path.join(baseline_dir, 
                                                 f"{metrics[i]}_results.csv")).replace({True: 1, False: 0})
        df_baseline = df_baseline.dropna()
        df_baseline = df_baseline.drop(columns="episode").rolling(100).sum()
        df_baseline['mean'] = df_baseline[[x for x in df_baseline.columns if x.startswith("run_")]].mean(axis=1)
        df_baseline['std'] = df_baseline[[x for x in df_baseline.columns if x.startswith("run_")]].std(axis=1)
        
        ax[i].plot(df_metric.index, df_metric["mean"], label="PPO") # plot mean
        ax[i].fill_between(df_metric.index, df_metric["mean"]-df_metric["std"],
                           df_metric["mean"]+df_metric["std"], alpha=0.3) # plot min and max
        xlim = ax[i].get_xlim()
        ax[i].plot(df_baseline.index, df_baseline["mean"], label="Random") # plot mean
        ax[i].fill_between(df_baseline.index, df_baseline["mean"]-df_baseline["std"], 
                            df_baseline["mean"]+df_baseline["std"], alpha=0.3) # plot min and max
        if metrics[i] == "success":
            ax[i].plot(range(len(df_metric)), np.ones(len(df_metric))*100, label="Ideal")
        else:
            ax[i].plot(range(len(df_metric)), np.zeros(len(df_metric)), label="Ideal")
        ax[i].set_ylabel(metrics[i])
        ax[i].set_xlim(xlim)
        ax[i].minorticks_on()
        ax[i].grid(which="major")
        ax[i].grid(which='minor', linestyle=':')
    ax[-1].set_xlabel("episode")
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()

    prefix = in_dir[2:].replace("/", "_")
    fig.savefig(os.path.join(out_dir, f"{prefix}_avg-outcomes.png"))

def plot_deterministic_outcomes(in_dir, label, baseline_dir, out_dir):
    # Set font size for all plots
    plt.rcParams['font.size'] = '16'

    # Create rolling averages for deterministic evaluation
    metrics = ["episode_reward", "success", "crashed", "truncated"]
    fig, ax = plt.subplots(nrows=len(metrics), figsize=(10, 10))

    metrics = ["success", "crashed", "truncated"]
    fig, ax = plt.subplots(nrows=len(metrics), figsize=(10, 10))
    for i in range(len(metrics)):
        df_metric = pd.read_csv(os.path.join(in_dir, "evaluate", label,
                                             f"{metrics[i]}_results.csv")).replace({True: 1, False: 0})
        df_metric = df_metric.dropna()
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
        ax[i].minorticks_on()
        ax[i].grid(which="major")
        ax[i].grid(which='minor', linestyle=':')
    ax[-1].set_xlabel("step")
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()

    prefix = in_dir[2:].replace("/", "_")
    fig.savefig(os.path.join(out_dir, f"{prefix}_avg-deterministic-outcomes.png"))

def plot_compare_scenarios(in_dir, out_dir, scenario_labels=[], scenario_interval=500000):
    # Set font size for all plots
    plt.rcParams['font.size'] = '16'

    # Create rolling averages for deterministic evaluation
    for metric in ["episode_reward", "success"]:
        labels = ["vertical", "diagonal-25", "diagonal-50", "parallel"]
        colours = ["C0", "C2", "C3", "C4"]
        fig, ax = plt.subplots(nrows=len(labels), figsize=(15, 10), sharex='col', sharey='col')
        for i in range(len(labels)):
            df_metric = pd.read_csv(os.path.join(in_dir, 
                                                         "evaluate", labels[i],
                                                         f"{metric}_results.csv")).replace({True: 1, False: 0})
            df_metric = df_metric.dropna()
            df_metric['mean'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].mean(axis=1)
            df_metric['std'] = df_metric[[x for x in df_metric.columns if x.startswith("run_")]].std(axis=1)
                    
            ax[i].plot(df_metric["step"], df_metric["mean"], label=labels[i], color=colours[i]) # plot mean
            ax[i].fill_between(df_metric["step"], df_metric["mean"]-df_metric["std"],
                                   df_metric["mean"]+df_metric["std"], alpha=0.3, color=colours[i]) # plot min and max
            #ax[i].errorbar(df_metric["step"], df_metric["mean"], yerr=df_metric["std"], label=labels[i]) # plot mean

            if metric == "success":
                ax[i].plot(df_metric["step"], np.ones(len(df_metric)), label="Ideal", color="C1")
            else:
                ax[i].plot(df_metric["step"], np.zeros(len(df_metric)), label="Ideal", color="C1")
            #ax[i].set_ylabel(metric)
            ax[i].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            ax[i].minorticks_on()
            ax[i].grid(which="major")
            ax[i].grid(which='minor', linestyle=':')

        ax[-1].set_xlabel("step")
    
        if metric == "episode_reward":
            fig.text(0, 0.5, "total episode reward", va='center', rotation='vertical', fontsize=18)
            #ymin, ymax = ax[0].get_ylim()
            #ax[0].set_ylim([-100, ymax])
        else:
            fig.text(0, 0.5, f"{metric}", va='center', rotation='vertical', fontsize=18)
    
        lines = [Line2D([0], [0], color=c, ) for c in colours+["C1"]]
        ax[0].legend(handles=lines, labels=labels+["Ideal"], loc='upper left', bbox_to_anchor=(1, 1))

        for i in range(len(scenario_labels)):
            ymin, ymax = ax[0].get_ylim()
            y = ymax + abs(ymax - ymin) * 0.1
            x = scenario_interval* (0.5+i)
            ax[0].text(x, y, f"{scenario_labels[i]}\n training", horizontalalignment="center")

        for i in range(1, len(scenario_labels)):
            for j in range(len(labels)):
                ax[j].axvline(x=scenario_interval*i, ymin=0, ymax=1.2, c='black', linestyle="--", lw=2, zorder=0, clip_on=False)
            
        fig.tight_layout()
        fig.subplots_adjust(wspace=0, hspace=0.05)
        fig.subplots_adjust(left=0.08)

        prefix = in_dir[2:].replace("/", "_")
        fig.savefig(os.path.join(out_dir, f"{prefix}_compare-deterministic-{metric}.png"))

def plot_all(in_dir, baseline_dir, out_dir, label=None, scenario_labels=[], scenario_interval=500000):
    if not os.path.exists(out_dir):
        print(f"Creating output directory {out_dir}")
        os.makedirs(out_dir)
        
    plot_training_curve(in_dir, baseline_dir, out_dir)
    plot_outcomes(in_dir, baseline_dir, out_dir)
    if label is not None:
        plot_deterministic_outcomes(in_dir, label, baseline_dir, out_dir)
    plot_compare_scenarios(in_dir, out_dir,scenario_labels, scenario_interval)

# Single Scenario Training
plot_all("./results/baselines/single_vertical", "./results/random_vertical", "./plots", label="vertical")
plot_all("./results/baselines/single_diagonal-25", "./results/random_diagonal-25", "./plots", label="diagonal-25")
plot_all("./results/baselines/single_diagonal-50", "./results/random_diagonal-50", "./plots", label="diagonal-50")
plot_all("./results/baselines/single_parallel", "./results/random_parallel", "./plots", label="parallel")

# Sequential training
plot_all("./results/baselines/sequential_in-order", "./results/random_vertical", "./plots", 
         scenario_labels=["vertical", "diagonal-25", "diagonal-50", "parallel"])

"""
plot_importance("./results/ppo-ewc/single_vertical")
plot_importance("./results/ppo-ewc/single_diagonal-25")
plot_importance("./results/ppo-ewc/single_diagonal-50")
plot_importance("./results/ppo-ewc/single_parallel")
plot_importance("./results/ppo-ewc/sequential_in-order")
plot_importance("./results/ppo-ewc/sequential_in-order_num_cells-128")
plot_importance("./results/ppo-ewc/sequential_in-order_num_cells-256")
"""