

"""
    APLoss Implementation
    Towards accurate one-stage object detection with AP-loss, Chen et al, CVPR 2019

    The code is taken from their official implementation AP_loss(logits,targets):
    https://github.com/cccorn/AP-loss/blob/master/lib/model/aploss.py
"""

import torch
import torch.nn as nn

class backpropAPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta= 1.0, positive_label= 1, negative_label= 0):
        delta = 1.0

        grad = torch.zeros(logits.shape)
        metric = torch.zeros(1)

        convert_to_cuda = logits.is_cuda and (not grad.is_cuda)
        if convert_to_cuda:
            grad = grad.cuda()
            metric = metric.cuda()

        if torch.max(targets) <= 0:
            ctx.grad = grad
            return metric

        labels_p = (targets == positive_label)
        fg_logits = logits[labels_p]
        threshold_logit = torch.min(fg_logits) - delta

        ######## Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
        valid_labels_n = ((targets == negative_label) & (logits >= threshold_logit))
        valid_bg_logits = logits[valid_labels_n]
        valid_bg_grad = torch.zeros(len(valid_bg_logits))
        if convert_to_cuda:
            valid_bg_grad= valid_bg_grad.cuda()
        ########

        fg_num = len(fg_logits)
        prec = torch.zeros(fg_num)
        if convert_to_cuda:
            prec = prec.cuda()
        _, order = torch.sort(fg_logits)
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
        ctx.grad = grad

        metric = torch.sum(prec, dim=0, keepdim=True) / fg_num

        return 1 - metric.squeeze()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.grad * grad_output
        # backpropAPLoss takes 5 parameters and should return 5 gradients.
        # Return 4 None since we do not require gradients wrt others
        return grad_input, None, None, None, None

class APLoss(nn.Module):
    def __init__(self, delta= 1.0, positive_label= 1, negative_label= 0):
        super(APLoss, self).__init__()
        self.delta          = delta
        self.positive_label = positive_label
        self.negative_label = negative_label

    def forward(self, logits, targets):
        return backpropAPLoss.apply(logits, targets, self.delta, self.positive_label, self.negative_label)