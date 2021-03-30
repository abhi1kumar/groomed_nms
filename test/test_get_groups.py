

import os,sys
sys.path.append(os.getcwd())

import torch
import numpy as np

from lib.groomed_nms import get_groups

torch.manual_seed(0)
num_boxes = 10

iou = torch.rand(num_boxes,num_boxes)
for i in range(num_boxes):
    iou[i,i] = 1
iou = 0.5*(iou.transpose(1, 0) + iou)

scores = torch.sort(torch.rand(10), descending= True)[0]

groups = get_groups(scores_unsorted= scores, iou_unsorted= iou, group_threshold= 0.4)
print(groups)