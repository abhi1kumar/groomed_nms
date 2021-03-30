

import torch

# y_min_b1 = torch.Tensor([-1.5, -2.5])
# y_max_b1 = torch.Tensor([ 1.5,  2.5])
# y_min_b2 = torch.Tensor([-1, -2])
# y_max_b2 = torch.Tensor([ 2,  3])
#
# y_intersect_min = torch.max(torch.cat((y_min_b1.unsqueeze(1), y_min_b2.unsqueeze(1)), dim=1), dim=1)[0]
# y_intersect_max = torch.min(torch.cat((y_max_b1.unsqueeze(1), y_max_b2.unsqueeze(1)), dim=1), dim=1)[0]
# y_intersect = torch.max(torch.zeros((1,)), y_intersect_max - y_intersect_min)

# print(y_intersect_min, y_intersect_max)

y_min_b1 = torch.Tensor([-1.5, -2.5])
y_max_b1 = torch.Tensor([ 1.5,  2.5])
y_min_b2 = torch.Tensor([-1, -2, -3])
y_max_b2 = torch.Tensor([ 2,  3,  5])
y_intersect_min = torch.max(y_min_b1.unsqueeze(1), y_min_b2.unsqueeze(0))
y_intersect_max = torch.min(y_max_b1.unsqueeze(1), y_max_b2.unsqueeze(0))
y_intersect = torch.max(torch.zeros((1,)), y_intersect_max - y_intersect_min)

print(y_intersect)

vol_1 = torch.Tensor([1, 2])
vol_2 = torch.Tensor([10, 20, 30])

vol = vol_1.unsqueeze(1) + vol_2.unsqueeze(0)
print(vol)