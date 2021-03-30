import numpy as np
import itertools
import torch

from lib.groomed_nms import indices_copy

torch.manual_seed(0)

print("\n================= dim=2, both  2D indA,  2D indB given ===============")
A = torch.zeros(3, 4)
A.requires_grad = True
B = torch.rand(3, 3)

indA = torch.LongTensor([[0, 0], [0, 1], [1, 1]])
indB = torch.LongTensor([[1, 1], [2, 1], [2, 2]])

print("Inputs")
print(A)
print(B)
print(indA)
print(indB)

A= indices_copy(A, B, indA, indB)
print("Output inplace")
print(A)
A = torch.zeros(3, 4)
A.requires_grad = True
new_A = indices_copy(A, B, indA, indB, inplace=False)
print("Output out of place")
print(new_A)
print("Unmodified A")
print(A)

print("\n================= dim= 2, With only 2D indA given ===============")
A = torch.zeros(3, 4)
A.requires_grad = True
B = torch.rand(3, 1)

print("Inputs")
print(A)
print(B)

A= indices_copy(A, B, indA)
print("Output inplace")
print(A)

A = torch.zeros(3, 4)
A.requires_grad = True
new_A = indices_copy(A, B, indA, inplace=False)
print("Output out of place")
print(new_A)
print("Unmodified A")
print(A)

print("\n================= dim= 2, With only 1D indA given ===============")
A = torch.zeros(5, 5)
A.requires_grad = True
B = torch.rand(3, 3)
# indA = torch.LongTensor([[1,1], [1,2], [1,4], [2,1], [2,2], [2,4], [4,1], [4,2], [4,4]])
indA = torch.LongTensor([1, 2, 4])
print("Inputs")
print(A)
print(B)

A= indices_copy(A, B, indA)

print("Output inplace")
print(A)

A = torch.zeros(5, 5)
A.requires_grad = True
new_A = indices_copy(A, B, indA, inplace=False)
print("Output out of place")
print(new_A)
print("Unmodified A")
print(A)

print("\n================= dim> 2, both indA, indB given ===============")
A = torch.zeros(3, 4, 6)
A.requires_grad = True
B = torch.rand(3, 4, 6)

indA = torch.LongTensor([[0, 0], [0, 1], [1, 1]])
indB = torch.LongTensor([[1, 1], [2, 1], [2, 2]])

A= indices_copy(A, B, indA, indB)
print("Output inplace")
print(A)

A = torch.zeros(3, 4, 6)
A.requires_grad = True
new_A = indices_copy(A, B, indA, indB, inplace=False)
print("Output out of place")
print(new_A)
print("Unmodified A")
print(A)