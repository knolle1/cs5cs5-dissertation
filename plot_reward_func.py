# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:13:11 2024

@author: kimno
"""

import numpy as np
import matplotlib.pyplot as plt

for p in [1, 0.5, 0.1]:
    for k in [1]:
        x = np.arange(5, step=0.01).reshape((-1, 1))
        
        y=-k *np.power(
            np.dot(
                np.abs(x),
                np.array([1]),
            ),
            p,
        )
        
        plt.plot(x, y, label=f"p={p} k={k}")
    
plt.legend()
plt.xlabel(r"$x = \sum \ W_i * |s_i - s_{g,i} |$")