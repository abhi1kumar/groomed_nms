import math
import os, sys

from torch.autograd import Variable
import torch.nn.functional as F

sys.path.append(os.getcwd())

import numpy as np
import torch
torch.set_printoptions(profile="full")
torch.set_printoptions(precision=4)

from lib.groomed_nms import differentiable_nms, get_groups
from lib.loss.aploss import APLoss
from lib.loss.ranknetloss import RankNetLoss
loss_1_wt = 0.05

def check_backward(acceptance_prob, targets, scores= None, ious_2d_for_nms= None, w= None, loss_to_use= "ce", example_weights= None, fg_index_for_nms= None, groups= None, groups_in_loss= False, print_in_single_line= False):
    loss_function_2 = torch.nn.BCELoss(reduction= 'none')
    loss_2_wt = 0.0
    if loss_2_wt > 0.0:
        print("Using sum of {} loss + {} * celoss".format(loss_to_use, loss_2_wt))
    else:
        print("Using {} loss".format(loss_to_use))
    if groups_in_loss:
        print("Using groups in loss")
    if loss_to_use == "ce":
        loss_function = torch.nn.BCELoss(reduction= 'none')
        # acceptance_prob = F.sigmoid(acceptance_prob)
    elif loss_to_use == "ap":
        loss_function = APLoss()
    elif loss_to_use == "mse":
        loss_function = torch.nn.L1Loss(reduction= 'none')
    elif loss_to_use == "ranknet":
        loss_function = RankNetLoss()

    if not print_in_single_line:
        print("After NMS probabilities  = ", acceptance_prob)
        print("Targets                  = ", targets)
    if not groups_in_loss:
        loss_1 = loss_1_wt * loss_function(acceptance_prob, targets)
        if loss_to_use == "ce" or loss_to_use == "mse":
            if example_weights is not None:
                loss_1 = loss_1 * example_weights
            loss_1 = loss_1.mean()
    else:
        num_groups = len(groups)
        for grp_index in range(num_groups):
            grp_box_index = fg_index_for_nms[groups[grp_index]]
            loss_group    = loss_1_wt * loss_function(acceptance_prob[grp_box_index], targets[grp_box_index])
            if loss_to_use == "ce" or loss_to_use == "mse":
                if example_weights is not None:
                    loss_group = loss_group * example_weights[grp_box_index]
                loss_group = loss_group.mean()
            if grp_index == 0:
                loss_1 = loss_group
            else:
                loss_1 += loss_group
        loss_1 /= num_groups

    if loss_2_wt > 0.0:
        loss_2 = loss_2_wt*loss_function_2(acceptance_prob, targets)
        if example_weights is not None:
            loss_2 = loss_2 * example_weights
    else:
        loss_2 = 0
    loss   = loss_1 + loss_2
    loss = loss.mean()
    print("Loss                     = ", loss)
    loss.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_value_(scores, 1)

    if print_in_single_line:
        print("           Index,   Bef NMS,  Aft NMS,   Relev,   Grad")
        disp_tensor = torch.cat([torch.arange(num_boxes_display).float().cuda().unsqueeze(1), scores[:num_boxes_display].unsqueeze(1), acceptance_prob[:num_boxes_display].unsqueeze(1), targets[:num_boxes_display]. unsqueeze(1)], dim= 1)
        if example_weights is not None:
            disp_tensor = torch.cat([disp_tensor, example_weights[:num_boxes_display]. unsqueeze(1)], dim= 1)
        disp_tensor = torch.cat([disp_tensor, scores.grad[:num_boxes_display].unsqueeze(1)], dim= 1)
        print(disp_tensor)
        print("No of negative grad = {}\n".format(torch.sum(scores.grad < 0)))

    if scores is not None:
        if not print_in_single_line:
            print("Scores gradient          = ", scores.grad)
        # retain_graph allows but do not accumulate the gradients again
        scores.grad.data.zero_()
    if not print_in_single_line:
        if ious_2d_for_nms is not None:
            print("IOU_overlaps gradient  = ", ious_2d_for_nms.grad)
        if w is not None:
            print("w gradient = ", w.grad)

def testing_with_loss(scores_to_nms, ious_2d_for_nms, fg_index_for_nms, num_boxes_for_nms, loss_to_use, scoring_method, sorting_method, temperature, sorting_temperature, cuda_testing= True, fg_index_gt_all_cases= None, group_boxes = False, groups_in_loss= False):
    groups = get_groups(iou_unsorted= ious_2d_for_nms, group_threshold= 0.4, scores_unsorted= scores_to_nms[fg_index_for_nms])
    #====================================================================
    # When the top rank is a failure
    #====================================================================
    fg_index_gt                  = fg_index_gt_all_cases[0]
    targets_for_nms              = torch.zeros((num_boxes, )).float()
    targets_for_nms[fg_index_gt] = 1
    loss_weights                 = torch.ones((num_boxes, )).float()
    loss_weights[fg_index_gt]    = np.power(num_boxes/len(fg_index_gt), weighing_power)

    if cuda_testing:
        targets_for_nms = targets_for_nms.cuda()
        loss_weights    = loss_weights.cuda()

    _, _, scores_after_nms_img = differentiable_nms(scores_unsorted= scores_to_nms[fg_index_for_nms], iou_unsorted= ious_2d_for_nms, temperature= temperature, return_sorted_prob= False, pruning_method= scoring_method, sorting_method= sorting_method, sorting_temperature= sorting_temperature, group_boxes = group_boxes)

    scores_after_nms[fg_index_for_nms] = scores_after_nms_img
    check_backward(acceptance_prob= scores_after_nms[:num_boxes_for_nms], targets= targets_for_nms[:num_boxes_for_nms], scores= scores_to_nms, ious_2d_for_nms= ious_2d_for_nms, loss_to_use= loss_to_use, example_weights=loss_weights[:num_boxes_for_nms], fg_index_for_nms= fg_index_for_nms, groups= groups, groups_in_loss= groups_in_loss, print_in_single_line= True)

    #====================================================================
    # When the top rank is a success
    #====================================================================
    fg_index_gt                  = fg_index_gt_all_cases[1]
    targets_for_nms              = torch.zeros((num_boxes, )).float()
    targets_for_nms[fg_index_gt] = 1
    loss_weights                 = torch.ones((num_boxes, )).float()
    loss_weights[fg_index_gt]    = np.power(num_boxes/len(fg_index_gt), weighing_power)

    if cuda_testing:
        targets_for_nms = targets_for_nms.cuda()
        loss_weights    = loss_weights.cuda()

    _, _, scores_after_nms_img = differentiable_nms(scores_unsorted= scores_to_nms[fg_index_for_nms], iou_unsorted= ious_2d_for_nms, temperature= temperature, return_sorted_prob= False, pruning_method= scoring_method, sorting_method= sorting_method, sorting_temperature= sorting_temperature, group_boxes = group_boxes)

    scores_after_nms[fg_index_for_nms] = scores_after_nms_img
    check_backward(acceptance_prob= scores_after_nms[:num_boxes_for_nms], targets= targets_for_nms[:num_boxes_for_nms], scores= scores_to_nms, ious_2d_for_nms= ious_2d_for_nms, loss_to_use= loss_to_use, example_weights=loss_weights[:num_boxes_for_nms], fg_index_for_nms= fg_index_for_nms, groups= groups, groups_in_loss= groups_in_loss, print_in_single_line= True)

    #====================================================================
    # When only backgrounds
    #====================================================================
    fg_index_gt                  = []
    targets_for_nms              = torch.zeros((num_boxes, )).float()
    targets_for_nms[fg_index_gt] = 1
    loss_weights                 = torch.ones((num_boxes, )).float()
    if len(fg_index_gt) > 0:
        loss_weights[fg_index_gt]= np.power(num_boxes/len(fg_index_gt), weighing_power)

    if cuda_testing:
        targets_for_nms = targets_for_nms.cuda()
        loss_weights    = loss_weights.cuda()

    _, _, scores_after_nms_img = differentiable_nms(scores_unsorted= scores_to_nms[fg_index_for_nms], iou_unsorted= ious_2d_for_nms, temperature= temperature, return_sorted_prob= False, pruning_method= scoring_method, sorting_method= sorting_method, sorting_temperature= sorting_temperature, group_boxes = group_boxes)

    scores_after_nms[fg_index_for_nms] = scores_after_nms_img
    check_backward(acceptance_prob= scores_after_nms[:num_boxes_for_nms], targets= targets_for_nms[:num_boxes_for_nms], scores= scores_to_nms, ious_2d_for_nms= ious_2d_for_nms, loss_to_use= loss_to_use, example_weights=loss_weights[:num_boxes_for_nms], fg_index_for_nms= fg_index_for_nms, groups= groups, groups_in_loss= groups_in_loss, print_in_single_line= True)

#==============================================================================
# Main starts here
#==============================================================================
# torch.manual_seed(0)
# nms_overlap_threshold    = 0.4
# valid_box_prob_threshold = 0.2
temperature              = 0.1
shift= 1
cuda_testing = True
num_boxes_display = 5
'''
print("")
targets_for_nms = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0,  0, 1, 0, 1, 0])
scores_to_nms   = torch.Tensor([0.1, 0.1 , 0.1, 0.1, 0.1, 0.94449151, 0.98317957, 0.94365776, 0.92837799, 0.91277248])
ious_2d_for_nms    = torch.Tensor([[1., 0.94648953, 0.94525505, 0., 0.93143163],
                                   [0.94648953, 1.,         0.91186984, 0.,         0.95791224],
                                   [0.94525505, 0.91186984, 1.,         0.,         0.88044045],
                                   [0.,         0.,         0.,         1.,         0.        ],
                                   [0.93143163, 0.95791224, 0.88044045, 0.,         1.        ]])

scores_after_nms = torch.zeros(scores_to_nms.shape)

if cuda_testing:
    scores_to_nms   = scores_to_nms.cuda()
    ious_2d_for_nms    = ious_2d_for_nms.cuda()
    targets_for_nms = targets_for_nms.cuda()
    scores_after_nms= scores_after_nms.cuda()

scores_to_nms.requires_grad= True
ious_2d_for_nms.requires_grad= True

_, sorted_index = torch.sort(scores_to_nms, descending= True)
num_boxes_for_nms       = min(5, sorted_index.shape[0])
fg_index_for_nms        = sorted_index[:num_boxes_for_nms]

if scores_to_nms.is_cuda:
    fg_index_np = fg_index_for_nms.cpu()
else:
    fg_index_np = fg_index_for_nms
fg_index_np     = fg_index_np.clone().numpy()
bg_index_for_nms= torch.from_numpy(np.setdiff1d(np.arange(scores_to_nms.shape[0]), fg_index_np))
if cuda_testing:
    bg_index_for_nms = bg_index_for_nms.cuda()

_, invalid_indexes, scores_after_nms_img = differentiable_nms(scores_unsorted= scores_to_nms[fg_index_for_nms], iou_unsorted= ious_2d_for_nms, temperature= temperature, return_sorted_prob= False, scoring_method= "basic")

scores_after_nms[fg_index_for_nms] = scores_after_nms_img
# scores_after_nms[fg_index_for_nms[invalid_indexes] ] = torch.min(scores_to_nms[bg_index_for_nms]) + 1e-3
# scores_after_nms[bg_index_for_nms] = scores_to_nms[bg_index_for_nms]
print("Before NMS probabilities = ", scores_to_nms)
check_backward(acceptance_prob= scores_after_nms, targets= targets_for_nms, scores= scores_to_nms, ious_2d_for_nms= ious_2d_for_nms)
'''

#====================================================================
# Testing gradients
#====================================================================
losses_to_use       = ["ap"]#, "mse", "ce"]
num_boxes         = 500
num_boxes_select  = 250
num_boxes_display = 10
temperature       = 0.1
weighing_power    = 1.0
scoring_method    = "linear"
sorting_method    = "hard"
sorting_temperature = 0.0007
groups_in_loss     = False
special_box = 7
torch.manual_seed(0)
np.random.seed(0)

data = -np.sort(-np.random.uniform(low=0, high=1, size=(num_boxes, )) )
data = data.tolist()
scores_to_nms = torch.Tensor(data).float()
scores_after_nms = torch.zeros(scores_to_nms.shape)

if cuda_testing:
    scores_to_nms   = scores_to_nms.cuda()
    scores_after_nms= scores_after_nms.cuda()

scores_to_nms.requires_grad= True

_, sorted_index = torch.sort(scores_to_nms, descending= True)
num_boxes_for_nms       = min(num_boxes_select, sorted_index.shape[0])
fg_index_for_nms        = sorted_index[:num_boxes_for_nms]

fg_index_gt_all_cases        = np.zeros((2, 2))
fg_index_gt_all_cases[0]     = np.array([3, special_box+1])#[0, special_box, 4]
fg_index_gt_all_cases[1]     = np.array([0, special_box  ])#[1, special_box, 4]

#==============================================================
# Original testing
#==============================================================
data = np.random.uniform(low=0, high=1, size=(num_boxes_for_nms, num_boxes_for_nms ))
data[num_boxes_display-1] = np.zeros((num_boxes_for_nms, ))
data[:,num_boxes_display-1] = np.zeros((num_boxes_for_nms, ))
data[num_boxes_display-1,num_boxes_display-1] = 1
data = 0.5*(data.transpose(1, 0) + data)
data = data.tolist()
ious_2d_for_nms = torch.Tensor(data).float()
if cuda_testing:
    ious_2d_for_nms = ious_2d_for_nms.cuda()
ious_2d_for_nms.requires_grad= True

# group_boxes = False
# for loss_to_use in losses_to_use:
#     testing_with_loss(scores_to_nms= scores_to_nms, ious_2d_for_nms= ious_2d_for_nms, fg_index_for_nms= fg_index_for_nms, num_boxes_for_nms= num_boxes_for_nms, loss_to_use= loss_to_use, scoring_method= scoring_method, sorting_method= sorting_method, temperature= temperature, sorting_temperature= sorting_temperature, cuda_testing= True, fg_index_gt_all_cases= fg_index_gt_all_cases, group_boxes = group_boxes)


#==============================================================
# Group/Non-group testing for 1 box
#==============================================================
special_box = [5, 9]
group_boxes = True
print("\n======================== Group Testing for 1 object ======================")
fg_index_gt_all_cases        = np.zeros((2, 1))
fg_index_gt_all_cases[0]     = np.array([3])
fg_index_gt_all_cases[1]     = np.array([0])

data = np.random.uniform(low=0.8, high=1, size=(num_boxes_for_nms, num_boxes_for_nms ))
np.fill_diagonal(data, 1)
data = 0.5*(data.transpose(1, 0) + data)
data = data.tolist()

ious_2d_for_nms = torch.Tensor(data).float()
if cuda_testing:
    ious_2d_for_nms = ious_2d_for_nms.cuda()
ious_2d_for_nms.requires_grad= True

for loss_to_use in losses_to_use:
    testing_with_loss(scores_to_nms= scores_to_nms, ious_2d_for_nms= ious_2d_for_nms, fg_index_for_nms= fg_index_for_nms, num_boxes_for_nms= num_boxes_for_nms, loss_to_use= loss_to_use, scoring_method= scoring_method, sorting_method= sorting_method, temperature= temperature, sorting_temperature= sorting_temperature, cuda_testing= True, fg_index_gt_all_cases= fg_index_gt_all_cases, group_boxes = group_boxes, groups_in_loss= groups_in_loss)

#==============================================================
# Group/Non-group testing for 2 boxes
#==============================================================
print("\n======================== Group Testing for 2 objects ======================")
fg_index_gt_all_cases        = np.zeros((2, 2))
fg_index_gt_all_cases[0]     = np.array([3, special_box[0]+1])
fg_index_gt_all_cases[1]     = np.array([0, special_box[0]  ])

data = np.random.uniform(low=0.8, high=1, size=(num_boxes_for_nms, num_boxes_for_nms ))
data[:special_box[0], special_box[0]:] = 0
data[special_box[0]:, :special_box[0]] = 0
np.fill_diagonal(data, 1)
data = 0.5*(data.transpose(1, 0) + data)
data = data.tolist()

ious_2d_for_nms = torch.Tensor(data).float()
if cuda_testing:
    ious_2d_for_nms = ious_2d_for_nms.cuda()
ious_2d_for_nms.requires_grad= True

# group_boxes = False
# for loss_to_use in losses_to_use:
#     testing_with_loss(scores_to_nms= scores_to_nms, ious_2d_for_nms= ious_2d_for_nms, fg_index_for_nms= fg_index_for_nms, num_boxes_for_nms= num_boxes_for_nms, loss_to_use= loss_to_use, scoring_method= scoring_method, sorting_method= sorting_method, temperature= temperature, sorting_temperature= sorting_temperature, cuda_testing= True, fg_index_gt_all_cases= fg_index_gt_all_cases, group_boxes = group_boxes)

# group_boxes = True
for loss_to_use in losses_to_use:
    testing_with_loss(scores_to_nms= scores_to_nms, ious_2d_for_nms= ious_2d_for_nms, fg_index_for_nms= fg_index_for_nms, num_boxes_for_nms= num_boxes_for_nms, loss_to_use= loss_to_use, scoring_method= scoring_method, sorting_method= sorting_method, temperature= temperature, sorting_temperature= sorting_temperature, cuda_testing= True, fg_index_gt_all_cases= fg_index_gt_all_cases, group_boxes = group_boxes, groups_in_loss= groups_in_loss)

#==============================================================
# Group testing for 3 boxes
#==============================================================
print("\n======================== Group Testing for 3 objects ======================")

fg_index_gt_all_cases        = np.zeros((2, 3))
fg_index_gt_all_cases[0]     = np.array([3, special_box[0]+1, special_box[1]+1])
fg_index_gt_all_cases[1]     = np.array([0, special_box[0]  , special_box[1]  ])

data = np.random.uniform(low=0.8, high=1, size=(num_boxes_for_nms, num_boxes_for_nms ))
data[:special_box[0], special_box[0]:] = 0
data[special_box[0]:, :special_box[0]] = 0
data[special_box[0]:special_box[1], special_box[1]:] = 0
data[special_box[1]:, special_box[0]:special_box[1]] = 0
np.fill_diagonal(data, 1)
data = 0.5*(data.transpose(1, 0) + data)
data = data.tolist()

ious_2d_for_nms = torch.Tensor(data).float()
if cuda_testing:
    ious_2d_for_nms = ious_2d_for_nms.cuda()
ious_2d_for_nms.requires_grad= True
for loss_to_use in losses_to_use:
    testing_with_loss(scores_to_nms= scores_to_nms, ious_2d_for_nms= ious_2d_for_nms, fg_index_for_nms= fg_index_for_nms, num_boxes_for_nms= num_boxes_for_nms, loss_to_use= loss_to_use, scoring_method= scoring_method, sorting_method= sorting_method, temperature= temperature, sorting_temperature= sorting_temperature, cuda_testing= True, fg_index_gt_all_cases= fg_index_gt_all_cases, group_boxes = group_boxes, groups_in_loss= groups_in_loss)