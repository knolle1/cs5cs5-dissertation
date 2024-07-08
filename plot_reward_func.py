# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:13:11 2024

@author: kimno
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '16'

fig, ax = plt.subplots(figsize=(10, 7))
for p in [1, 0.5, 0.1]:
    for k in [1]:
        x = np.arange(2, step=0.01).reshape((-1, 1))
        
        y=-k *np.power(
            np.dot(
                np.abs(x),
                np.array([1]),
            ),
            p,
        )
        
        ax.plot(x, y, label=f"p={p}")
    
ax.legend()
ax.set_xlabel("Distance between state and desired goal")
ax.set_ylabel("Value of reward function")