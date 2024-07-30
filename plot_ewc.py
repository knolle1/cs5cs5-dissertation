import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

def plot_importance(directory):
    files = os.listdir(os.path.join(directory, "train", "ewc_importances"))
    files = [f for f in files if f.endswith(".json")]
    steps = list(dict.fromkeys([s.split(".")[0].split("step_")[1] for s in files]))
    
    for step in steps:
        weights = None
        n = 0
        for file in files:
            if f"step_{step}" in file:
                f = open(os.path.join(directory, "train", "ewc_importances", file))
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
        fig.savefig(os.path.join(directory, "train", f"ewc_importances_step_{step}.png")) 


plot_importance("./results/ppo-ewc/single_vertical")
plot_importance("./results/ppo-ewc/single_diagonal-25")
plot_importance("./results/ppo-ewc/single_diagonal-50")
plot_importance("./results/ppo-ewc/single_parallel")