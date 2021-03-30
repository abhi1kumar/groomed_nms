import copy
import os, sys
sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions(suppress=True)

import matplotlib
import matplotlib.ticker
import matplotlib.patches as mpatches
import plot.plotting_params as params
from plot.common_operations import *
from matplotlib import pyplot as plt

data = np.array([
    [18.9,14.08,11.01,27.15,19.69,15.96,55.71,41.11,32.76,61.94,44.93,36.22],
    [19.67,14.32,11.27,27.38,19.75,15.92,55.62,41.07,32.89,61.83,44.98,36.29],
    [18.5,13.89,11.05,26.4,19.27,15.64,54.5,40.48,32.42,60.67,44.25,35.71],
    [17.97,13.57,10.85,27.97,20.43,16.05,57.11,41.37,33.17,61.62,46.14,36.25],
    ])

key = 1 # 0 --> easy, 1 --> mod, 2--> hard
columns_to_use = 3*np.arange(4) + key
group_sizes = np.array([50, 100, 200, 500])
colors     = [params.color2, params.color1]
linestyles = ['-', '--']
markers    = ['o', 's']
texts_to_use = [r"$3D$", r"$BEV$", r"$3D$", r"$BEV$"]
num_group_sizes = data.shape[0]
assert group_sizes.shape[0] == num_group_sizes

ms         = params.ms
lw         = params.lw
shift      = 2

plt.figure(figsize= params.size, dpi= params.DPI)
ax1 = plt.gca()
ax1.tick_params(axis='y', labelcolor=colors[0])
for i in range(num_group_sizes//2):
    text_label    = texts_to_use[i]
    ax1.plot(group_sizes, data[:, columns_to_use[i]], lw= lw, markersize= ms, marker= markers[i], linestyle= linestyles[i], label= text_label, color=colors[0])

ax1.set_xscale('log')
ax1.set_xticks(group_sizes)
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.set_xlim(48.5, 518)
ax1.set_xlabel('Group size')
ax1.set_ylim(13, 21)
ax1.set_ylabel(r"AP (IoU= 0.7)" )
ax1.yaxis.label.set_color(colors[0])
plt.grid(True)

ax2 = ax1.twinx()
ax2.tick_params(axis='y', labelcolor=colors[1])
handles = []
for i in range(num_group_sizes//2):
    columns_index = 6+columns_to_use[i]
    text_label    = texts_to_use[i+2]
    h, = ax2.plot(group_sizes, data[:, columns_index], lw= lw, markersize= ms, marker= markers[i], linestyle= linestyles[i], label= text_label, color=colors[1] )
    handles.append(copy.copy(h))

for h in handles:
    h.set_color("black")

ax2.set_ylim(40, 50)
ax2.set_ylabel(r"AP (IoU= 0.5)" )
ax2.yaxis.label.set_color(colors[1])

plt.legend(handles=handles, loc= (0.7, 0.18))
savefig(plt, "images/sensitivity_to_group_size.png")
plt.close()