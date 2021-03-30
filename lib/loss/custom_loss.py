import torch
import torch.nn as nn

class backpropCustomLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        grad = 2*(logits-targets)/logits.shape[0]
        ctx.grad = grad
        return torch.mean((logits-targets)**2)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.grad * grad_output
        # backpropCustomLoss takes 2 parameters and should return 2 gradients.
        # Return 1 None since we do not require gradients wrt targets
        return grad_input, None

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, logits, targets):
        return backpropCustomLoss.apply(logits, targets)