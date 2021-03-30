

import os, sys
sys.path.append(os.getcwd())

import torch

from lib.loss.aploss import APLoss

def calc_iou(a, b):

    a=a.type(torch.cuda.DoubleTensor)
    b=b.type(torch.cuda.DoubleTensor)

    area = (b[:, 2] - b[:, 0]+1) * (b[:, 3] - b[:, 1]+1)

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])+1
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])+1
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]+1) * (a[:, 3] - a[:, 1]+1), dim=1) + area - iw * ih
    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def AP_loss_old(logits, targets):
    delta = 1.0

    grad = torch.zeros(logits.shape).cuda()
    metric = torch.zeros(1).cuda()

    if torch.max(targets) <= 0:
        return grad, metric

    labels_p = (targets == 1)
    fg_logits = logits[labels_p]
    threshold_logit = torch.min(fg_logits) - delta

    ######## Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n  = ((targets == 0) & (logits >= threshold_logit))
    valid_bg_logits = logits[valid_labels_n]
    valid_bg_grad   = torch.zeros(len(valid_bg_logits)).cuda()
    ########

    fg_num = len(fg_logits)
    prec = torch.zeros(fg_num).cuda()
    _,order = torch.sort(fg_logits)
    max_prec = 0

    for ii in order:
        # Compute the rank as the interpolated precision
        tmp1 = fg_logits - fg_logits[ii]
        tmp1 = torch.clamp(tmp1 / (2 * delta) + 0.5, min=0, max=1)

        tmp2 = valid_bg_logits - fg_logits[ii]
        tmp2 = torch.clamp(tmp2 / (2 * delta) + 0.5, min=0, max=1)
        # rank wrt foreground
        a = torch.sum(tmp1) + 0.5
        # rank wrt background
        b = torch.sum(tmp2)
        tmp2 /= (a + b)
        current_prec = a / (a + b)
        if (max_prec <= current_prec):
            max_prec = current_prec
        else:
            tmp2 *= ((1 - max_prec) / (1 - current_prec))
        valid_bg_grad += tmp2
        prec[ii] = max_prec

    grad[valid_labels_n] = valid_bg_grad
    grad[labels_p] = -(1 - prec)

    fg_num = max(fg_num, 1)

    grad /= (fg_num)

    metric = torch.sum(prec, dim=0, keepdim=True) / fg_num

    return grad, 1 - metric


#===============================================================================
# Main starts here
#===============================================================================
torch.manual_seed(0)

p_num=torch.zeros(1)
labels_b=[]
# All in x1y1x2y2 format
annotations     = torch.zeros((1, 2, 5))
annotations[0, 0, 0:2] = 10
annotations[0, 0, 2:4] = 12
annotations[0, 0, 4]   = 2

annotations[0, 1, 2:4] = 2
annotations[0, 1,   4] = 1

anchors         = torch.zeros((1, 7, 4))
anchors[0, 0, 2:4] = 1
anchors[0, 1, 2:4] = 0.9
anchors[0, 2, 2:4] = 1.5

anchors[0, 3:6, 0:2] = 10
anchors[0, 3, 2:4] = 11
anchors[0, 4, 2:4] = 12
anchors[0, 5, 2:4] = 12.5

classifications = torch.zeros((1, 7, 3))
classifications[0] = torch.rand(7,3)

anchors = anchors.cuda()
annotations = annotations.cuda()
classifications = classifications.cuda()

for j in range(anchors.shape[0]):
    classification = classifications[j, :, :]
    bbox_annotation = annotations[j, :, :]
    bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

    IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
    IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

    # print(IoU)
    ######
    _, gt_IoU_argmax = torch.max(IoU, dim=0) # 1 x num_annotations
    # positive_indices = torch.ge(torch.zeros(IoU_max.shape).cuda(),1).type(torch.ByteTensor) #  bools of shape num_anchors
    positive_indices = torch.zeros(IoU_max.shape).type(torch.ByteTensor).cuda()

    # Assign the max overlaps (even when it is less than 0.5) as valid positive indices
    positive_indices[gt_IoU_argmax.long()] = 1

    ######
    # And requires conversion to long() in old Pytorch
    positive_indices = positive_indices.long() | torch.ge(IoU_max, 0.5).long()
    positive_indices = positive_indices.type(torch.cuda.ByteTensor)
    negative_indices = torch.lt(IoU_max, 0.4)

    assigned_annotations = bbox_annotation[IoU_argmax, :]

    # compute the loss for classification
    targets = torch.ones(classification.shape) * -1
    targets = targets.cuda()

    targets[negative_indices, :] = 0
    targets[positive_indices, :] = 0
    targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
    labels_b.append(targets)

# print("=================================================")
# print("With anchors and stuff")
# print("=================================================")
# print(anchors)
# print(annotations)
# print(classifications)
#
# print(classifications)
# print(labels_b)
# print("From inbuilt function")
# labels_b=torch.stack(labels_b)
# classification_grads,classification_losses=AP_loss_old(classifications,labels_b)
# print("From our loss based function")
# print(classification_losses)
# print(classification_grads)
#
# print("From inbuilt function")
# apLoss = APLoss()
# print(apLoss(classifications,labels_b))
#
# print("\n=================================================")
# print("When there are only backgrounds")
# print("=================================================")
# classifications = torch.Tensor([0.1402, 0.0049, 0.0069, 0.0064, 0.0108, 0.0059, 0.0073, 0.0025, 0.0023,
#         0.0064, 0.0043, 0.0058, 0.0072, 0.0049, 0.0080, 0.0044, 0.0047, 0.0081,
#         0.0102, 0.0101, 0.0213, 0.0229, 0.0072, 0.0116]).cuda()
#
# labels_b = torch.zeros(classifications.shape).cuda()
# print(AP_loss_old(classifications,labels_b)[1])
# print(apLoss(classifications,labels_b))
#
# print("\n=================================================")
# print("When there are only ignores")
# print("=================================================")
# classifications = torch.Tensor([0.1402, 0.0049, 0.0069]).cuda()
# labels_b        = -torch.ones(classifications.shape).cuda()
# print(AP_loss_old(classifications,labels_b)[1])
# print(apLoss(classifications,labels_b))

print("\n=================================================")
print("Effect of reshaping")
print("=================================================")
classifications = torch.Tensor([0.1402, 0.0049, 0.0069, 0.0064, 0.0108, 0.0059, 0.0073, 0.0025, 0.0023, 0.0064]).cuda()
labels_b = torch.Tensor([1, 0 , 0, 0, 0, 0, 1, 1, 1, 0]).cuda()
# print(AP_loss_old(classifications,labels_b))
#
# print(AP_loss_old(classifications.reshape((2, 5)),labels_b.reshape((2, 5))))

classifications = torch.Tensor([[1.00, 0.00, 0.0069, 0.0064, 0.0108], [0.0059, 0.8, 0.0025, 0.0023, 0.0064]]).cuda()
labels_b = torch.Tensor([[1, 0 , 0, 0, 0], [0, 0, 1, 0, 0]]).cuda()
classifications.requires_grad = True
loss_func = APLoss()
loss = loss_func(classifications, labels_b)
print("Loss = {}".format(loss))
loss.backward()
print(classifications.grad)
classifications.grad.data.zero_()

loss = loss_func(classifications[0], labels_b[0]) + loss_func(classifications[1], labels_b[1])
loss /= classifications.shape[0]
print("Loss = {}".format(loss))
loss.backward()
print(classifications.grad)
classifications.grad.data.zero_()

print("\n=================================================")
print("Effect of weighing the loss")
print("=================================================")
classifications = torch.Tensor([[1.00, 0.00, 0.0069, 0.0064, 0.0108], [0.0059, 0.8, 0.0025, 0.0023, 0.0064]]).cuda()
labels_b = torch.Tensor([[1, 0 , 0, 0, 0], [0, 0, 1, 0, 0]]).cuda()
classifications.requires_grad = True
loss_func = APLoss()

loss = loss_func(classifications[0], labels_b[0]) + loss_func(classifications[1], labels_b[1])
loss /= classifications.shape[0]
print("Loss = {}".format(loss))
loss.backward()
print(classifications.grad)
classifications.grad.data.zero_()

loss = loss_func(classifications[0], labels_b[0]) + loss_func(classifications[1], labels_b[1])
loss /= classifications.shape[0]
loss *= 0.1
print("Loss = {}".format(loss))
loss.backward()
print(classifications.grad)
classifications.grad.data.zero_()