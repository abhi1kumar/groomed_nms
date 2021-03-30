import os, sys
sys.path.append(os.getcwd())

import torch
from lib.nms.gpu_nms import gpu_nms

torch.set_printoptions(precision=3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from lib.groomed_nms import *
from lib.core import iou
from lib.nms_others import *
import time

def get_scores_iou(num_boxes, nms_overlap_threshold= 0.4):
    # Generate some random scores
    scores = torch.FloatTensor(num_boxes).uniform_(nms_overlap_threshold, 1.0)
    #scores, indices = torch.sort(scores, descending=True)

    # Generate some IOU stats
    iou = torch.rand(num_boxes, num_boxes) #torch.eye(num_boxes)
    mask = torch.eye(num_boxes, num_boxes).byte()
    iou.masked_fill_(mask, 1)
    #iou = torch.tril(iou)

    return scores, iou

def run_sign_inverse(scores, iou, temp, print_flag= True):
    print("")
    if print_flag:
        print(scores)
        print(iou)

    start = time.time()
    sorted_indices, nms_scores = differentiable_nms(scores, iou, nms_threshold= nms_overlap_threshold,
                                                     temperature=temp, valid_box_prob_threshold= valid_box_prob_threshold, method="matrix_mult")
    # sorted_indices, nms_scores = get_approximate_nms(scores, iou, temp=temp, method="sign_reverse")
    end = time.time()
    print("Time needed = {:.2f}".format(end - start))
    if print_flag:
        print(nms_scores)
        #print(nms_scores[sorted_indices])

#==============================================================================
# Main starts here
#==============================================================================
nms_overlap_threshold    = 0.4
valid_box_prob_threshold = 0.3
temp                     = 0.1
torch.manual_seed(0)
shift= 1

# num_boxes      = 5
# scores, iou    = get_scores_iou(num_boxes, nms_overlap_threshold)
# scores = scores[:3]
#
# iou = torch.Tensor([[1.0, 0.3 , 0.04], [0.3, 1.0 , 0.04], [0.04, 0.04, 1.0]])
# run_sign_inverse(scores, iou, temp)
#
# iou = torch.Tensor([[1.0, 0.3 , 0.04], [0.3, 1.0 , 0.6], [0.04, 0.6, 1.0]])
# run_sign_inverse(scores, iou, temp)
#
# iou = torch.Tensor([[1.0, 0.5 , 0.04], [0.5, 1.0 , 0.04], [0.04, 0.04, 1.0]])
# run_sign_inverse(scores, iou, temp)
#
# iou = torch.Tensor([[1.0, 0.5 , 0.3], [0.5, 1.0 , 0.6], [0.3, 0.6, 1.0]])
# run_sign_inverse(scores, iou, temp)
#
# iou = torch.Tensor([[1.0, 0.5 , 0.5], [0.5, 1.0 , 0.6], [0.5, 0.6, 1.0]])
# run_sign_inverse(scores, iou, temp)
#
# iou = torch.Tensor([[1.0, 0.5 , 0.0], [0.5, 1.0 , 0.6], [0.0, 0.6, 1.0]])
# run_sign_inverse(scores, iou, temp)

# num_boxes      = 500
# scores, iou    = get_scores_iou(num_boxes, nms_overlap_threshold)
# run_sign_inverse(scores, iou, temp, print_flag= True)

#==============================================================================
# Compare with all different versions of the NMS
#==============================================================================
print("")
sorting_method = "hard"
scoring_method = "linear"
group_boxes  = "True"
num_boxes    = 5
constant     = 10

scores, _    = get_scores_iou(num_boxes, nms_overlap_threshold)
scores.requires_grad= True
print("Scores")
print(torch.round(scores.clone()*1000)/1000)
w            = torch.rand(num_boxes)*constant
w.requires_grad= True
aboxes       = torch.zeros((num_boxes, 5))
aboxes[:, 2] = w
aboxes[:, 3] = w
aboxes[:, 4] = scores
print(aboxes)

iou_overlaps = iou(aboxes[:, :4], aboxes[:, :4], mode='combinations')#, shift= shift)
print("IOU overlaps")
print(torch.round(iou_overlaps.clone()*1000)/1000)
print(iou_overlaps.requires_grad)

# Remember to make a clone of the aboxes otherwise somehow iou values get changed
keep_inds     = gpu_nms(aboxes.clone().detach().numpy().astype(np.float32), nms_overlap_threshold, device_id=0)
print("Indices from Garrick et al", keep_inds)

keep_inds     = navneeth_soft_nms(aboxes.clone().detach().numpy(), Nt= nms_overlap_threshold, shift= shift)
print("Indices from Navneet et al", keep_inds.tolist())

keep_inds     = girshick_nms(aboxes.clone().detach().numpy(), nms_overlap_threshold, shift= shift)
print("Indices from Girsick et al", keep_inds)

# keep_inds     = dpp_nms(aboxes.clone().detach())
# print("Indices from DPP-NMS      ", keep_inds.tolist())

keep_inds, _, non_suppression_prob = differentiable_nms(aboxes[:, 4], iou_overlaps, nms_threshold= nms_overlap_threshold,
                                                        temperature=temp, valid_box_prob_threshold= valid_box_prob_threshold, pruning_method= scoring_method, sorting_method= sorting_method, group_boxes= group_boxes, debug= True)
print("Indices from Ours         ", keep_inds.clone().detach().numpy().tolist())

print("\n=========================== More Tests ===================================")
debug = False

iou_overlaps = torch.Tensor([[1.00, 0.00, 0.00, 0.00], [0.00, 1.00, 0.00, 0.00], [0.9, 0.9, 1.00, 0], [0, 0, 0, 1.00]])
score = torch.Tensor([0.99, 0.98, 0.8, 0.7])
print("\nOut id=[0.990, 0.980, 0.000, 0.700]")
keep_inds, _, non_suppression_prob = differentiable_nms(score, iou_overlaps, nms_threshold= nms_overlap_threshold,
                                                        temperature=temp, valid_box_prob_threshold= valid_box_prob_threshold, pruning_method= scoring_method, sorting_method= sorting_method, return_sorted_prob= False, group_boxes= group_boxes, debug= debug)
print(non_suppression_prob)


iou_overlaps = torch.Tensor([[1.00, 0.00, 0.00, 0.00, 0.00], [0.00, 1.00, 0.00, 0.00, 0.00], [0.9, 0.9, 1.00, 0, 0.00], [0.9, 0.9, 0, 1.00, 0.00], [0., 0., 0.9, 0.9, 1.00]])
score = torch.Tensor([0.99, 0.98, 0.8, 0.7, 0.6])
print("\nOut id=[0.990, 0.980, 0.000, 0.000, 0.600]")
keep_inds, _, non_suppression_prob = differentiable_nms(score, iou_overlaps, nms_threshold= nms_overlap_threshold,
                                                        temperature=temp, valid_box_prob_threshold= valid_box_prob_threshold, pruning_method= scoring_method, sorting_method= sorting_method, return_sorted_prob= False, group_boxes= group_boxes, debug= debug)
print(non_suppression_prob)