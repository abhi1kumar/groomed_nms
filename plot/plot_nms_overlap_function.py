

"""
    Sample Run:
    python plot/plot_nms_overlap_function.py

    Plots the overlap functions of different functions.
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions(suppress=True)

import matplotlib
import plot.plotting_params as params
from plot.common_operations import *
from matplotlib import pyplot as plt

from lib.groomed_nms import pruning_function

nms_threshold = 0.4
pad = 0.03
N = 1000
plot_lw = params.lw
matplotlib.rcParams.update({"font.size": 34})
temperature_list = np.array([0.03, 0.1, 0.3])
colors_list      = ["red", "orange", "dodgerblue", "purple", "limegreen"]

delta = 1.0/N
iou = np.arange(0, 1+delta, delta)

classic = np.ones(iou.shape)
classic[iou > nms_threshold] = 0


plt.figure(figsize= (15, 9.6), dpi= params.DPI)
plt.plot(iou, classic, color= "green"    , label= "Classic", lw= plot_lw)
plt.plot(iou, 1-iou  , color= "limegreen", label= "Linear" , lw= plot_lw)

# Plot soft NMS
soft_nms    = 1 - pruning_function(iou, nms_threshold= nms_threshold, temperature= 0.5, pruning_method= "soft_nms")
soft_label  = r"Exponential $\tau= $" + "{:.2f}".format(0.5)
plt.plot(iou, soft_nms, color= "k", label= soft_label, lw= plot_lw, linestyle= "--")
for i in range(temperature_list.shape[0]):
    temperature = temperature_list[i]
    soft_nms    = 1 - pruning_function(iou, nms_threshold= nms_threshold, temperature= temperature, pruning_method= "soft_nms")
    soft_label  = r"Exponential $\tau= $" + "{:.2f}".format(temperature)
    plt.plot(iou, soft_nms, color= colors_list[i], label= soft_label, lw= plot_lw, linestyle= "--")

# Plot sigmoidal NMS
for i in range(temperature_list.shape[0]):
    temperature = temperature_list[i]
    diff_nms    = 1 - pruning_function(iou, nms_threshold= nms_threshold, temperature= temperature, pruning_method= "sigmoidal")
    diff_label  = r"Sigmoidal $\tau= $" + "{:.2f}".format(temperature)
    plt.plot(iou, diff_nms, color= colors_list[i], label= diff_label, lw= plot_lw)

plt.grid(True)
plt.xlim((0, 1))
plt.ylim((0-pad, 1+pad))
plt.legend(bbox_to_anchor=(0.99, 1.02))
plt.xlabel(r"IOU Overlap $(o)$")
plt.ylabel("Acceptance")

savefig(plt, "images/nms_overlap_function.png")
plt.close()

#=====================================================================
# Plot pruning function
#=====================================================================
classic = 1-classic

plt.figure(figsize= (15, 9.6), dpi= params.DPI)
plt.plot(iou, classic , color= "black", label=r"Classic" , lw= plot_lw)
plt.plot(iou, iou     , color= "green", label=r"Linear"  , lw= plot_lw)

# Plot soft NMS
soft_nms    = pruning_function(iou, nms_threshold= nms_threshold, temperature= 0.5, pruning_method= "soft_nms")
soft_label  = r"Exponential  $\tau= $" + "{:.2f}".format(0.5)
plt.plot(iou, soft_nms, color= "purple", label= soft_label, lw= plot_lw, linestyle= "--")

for i in range(temperature_list.shape[0]):
    temperature = temperature_list[i]
    soft_nms    = pruning_function(iou, nms_threshold= nms_threshold, temperature= temperature, pruning_method= "soft_nms")
    soft_label  = r"Exponential  $\tau= $" + "{:.2f}".format(temperature)
    plt.plot(iou, soft_nms, color= colors_list[i], label= soft_label, lw= plot_lw, linestyle= "--")

# Plot sigmoidal NMS
for i in range(temperature_list.shape[0]):
    temperature = temperature_list[i]
    diff_nms    = pruning_function(iou, nms_threshold= nms_threshold, temperature= temperature, pruning_method= "sigmoidal")
    diff_label  = r"Sigmoidal $\tau= $" + "{:.2f}".format(temperature)
    plt.plot(iou, diff_nms, color= colors_list[i], label= diff_label, lw= plot_lw)

plt.grid(True)
plt.xlim((0, 1))
plt.ylim((0-pad, 1+pad))
plt.xlabel(r"IOU Overlap  $(o)$")
plt.ylabel(r"Pruned Output $( p(o) )$")
#plt.legend(bbox_to_anchor=(1.05, -0.13), ncol=3, fontsize=24) #(x, y)
plt.legend(bbox_to_anchor=(0.99, 1.02), fontsize=24)

savefig(plt, "images/nms_pruning_function_2.png")
plt.close()