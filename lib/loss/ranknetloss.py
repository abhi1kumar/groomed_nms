
"""
    RankNet loss taken from
    https://github.com/allegro/allRank/blob/master/allrank/models/losses/rankNet.py
"""

from itertools import product

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from lib.groomed_nms import cast_to_cpu_cuda_tensor

PADDED_Y_VALUE = -100

class RankNetLoss(nn.Module):
    def __init__(self):
        super(RankNetLoss, self).__init__()

    def forward(self, y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, weight_by_diff=False, weight_by_diff_powed=False):
        """
        RankNet loss introduced in "Learning to Rank using Gradient Descent".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
        :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
        :return: loss value, a torch.Tensor
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)

        mask = y_true == padded_value_indicator
        y_pred[mask] = float('-inf')
        y_true[mask] = float('-inf')

        # here we generate every pair of indices from the range of document length in the batch
        document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

        pairs_true = y_true[:, document_pairs_candidates]
        selected_pred = y_pred[:, document_pairs_candidates]

        # here we calculate the relative true relevance of every candidate pair
        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

        # here we filter just the pairs that are 'positive' and did not involve a padded instance
        # we can do that since in the candidate pairs we had symetric pairs so we can stick with
        # positive ones for a simpler loss function formulation
        if torch.sum(torch.abs(true_diffs)) <= 0:
            output = torch.Tensor([0.0]).type(pred_diffs.dtype)
            output = cast_to_cpu_cuda_tensor(output, pred_diffs)
            output.requires_grad = True
            return output
        else:
            the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

            pred_diffs = pred_diffs[the_mask]

            weight = None
            if weight_by_diff:
                abs_diff = torch.abs(true_diffs)
                weight = abs_diff[the_mask]
            elif weight_by_diff_powed:
                true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
                abs_diff = torch.abs(true_pow_diffs)
                weight = abs_diff[the_mask]

            # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
            # whether one document is better than the other and not about the actual difference in
            # their relevancy levels
            true_diffs = (true_diffs > 0).type(torch.float32)
            true_diffs = true_diffs[the_mask]

            return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)