
import os,sys
sys.path.append(os.getcwd())

import numpy as np
import torch

from lib.math_3d import get_corners_of_cuboid

np.random.seed(0)
m  = 5
x3d = 30*np.random.uniform(size=(m ))
y3d = 10*np.random.uniform(size=(m ))
z3d = 15*np.random.uniform(size=(m ))
l3d = 4 *np.random.uniform(size=(m ))
w3d = 5 *np.random.uniform(size=(m ))
h3d = 6 *np.random.uniform(size=(m ))
r3d = 1 *np.random.uniform(low= -1.57, high= 1.57, size=(m ))

out = get_corners_of_cuboid(x3d, y3d, z3d, w3d, h3d, l3d, r3d)

x3d_torch = torch.from_numpy(x3d).float()
y3d_torch = torch.from_numpy(y3d).float()
z3d_torch = torch.from_numpy(z3d).float()
l3d_torch = torch.from_numpy(l3d).float()
w3d_torch = torch.from_numpy(w3d).float()
h3d_torch = torch.from_numpy(h3d).float()
r3d_torch = torch.from_numpy(r3d).float()

out_torch = get_corners_of_cuboid(x3d_torch, y3d_torch, z3d_torch, w3d_torch, h3d_torch, l3d_torch, r3d_torch)

out_torch_np = out_torch.numpy()

print(out)
print(out_torch_np)
print(np.sum(np.abs(out-out_torch_np)))