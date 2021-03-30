

import os, sys
sys.path.append(os.getcwd())

import torch
from lib.loss.custom_loss import *

def test_grad_with_weighing(classifications, targets, wt= 0.0):
    loss_func = torch.nn.MSELoss()
    loss = wt * loss_func(classifications, targets)
    print("Loss = {}".format(loss))
    loss.backward()
    print(classifications.grad)
    classifications.grad.data.zero_()

    loss_func = CustomLoss()
    loss = wt * loss_func(classifications, targets)
    print("Loss = {}".format(loss))
    loss.backward()
    print(classifications.grad)
    classifications.grad.data.zero_()

classifications = torch.Tensor([1.00, 0.6, 0.7, 0.0, 0.0])
targets         = torch.Tensor([0   , 1   , 0  , 0.0, 0.0])
classifications.requires_grad = True

print("\nWhen weights are one")
test_grad_with_weighing(classifications, targets, wt= 1)

print("\nWhen weights are not one")
test_grad_with_weighing(classifications, targets, wt= 0.1)


