

import os, sys
sys.path.append(os.getcwd())

import plot.plotting_params as params
from plot.common_operations import *

from matplotlib import pyplot as plt
from easydict import EasyDict as edict

import torch.nn as nn
import torch.optim as optim
from lib.core import *

def get_lr_val(conf): 
    lr_val = []

    cnt_batch = 0
    for iteration in range(conf.max_iter):
        cnt_batch += 1
        # learning rate
        adjust_lr(conf, optimizer, iteration)
        lr_val.append(get_lr(optimizer))

    ind_array = np.arange(cnt_batch)

    return lr_val, ind_array

lw = params.lw+1
max_iter_list = [60000, 60000, 60000, 60000]
lr_power_list = [0.99 , 0.9  , 0.7  , 0.5]
color_list    = [params.color1,  params.color2, "orange", "purple", "violet", "yellow", "black"]

model = nn.Sequential(nn.Linear(5,4))
conf = edict()
# solver settings
conf.solver_type = "sgd"
conf.lr          = 0.004
conf.momentum    = 0.9

# sgd parameters
conf.lr_policy = "poly"
conf.lr_steps  = None
conf.lr_target = conf.lr * 0.00001

conf.max_iter            = 50000
conf.lr_power            = 0.9
optimizer                = optim.Adadelta(model.parameters(), lr= conf.lr)
lr_val_1, ind_array_1    = get_lr_val(conf)

plt.figure(figsize= params.size, dpi= params.DPI)
plt.semilogy(ind_array_1, lr_val_1, lw= lw, color= "limegreen", label= "Self balancing")

for i in range(len(max_iter_list)): 
    conf.max_iter            = max_iter_list[i]
    conf.lr_power            = lr_power_list[i]
    optimizer                = optim.Adadelta(model.parameters(), lr= conf.lr)
    lr_val_2, ind_array_2    = get_lr_val(conf)
    plt.semilogy(ind_array_2, lr_val_2, lw= lw, color= color_list[i], label= "({:}k, {:.2f})".format(int(max_iter_list[i]/1000), lr_power_list[i]))

plt.grid(True)
plt.legend()
plt.xlabel("Mini-batches")
plt.ylabel("learning rate (log)")

savefig(plt, os.path.join(params.IMAGE_DIR, "lr_policy.png"))
plt.show()