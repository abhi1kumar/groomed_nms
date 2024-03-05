"""
This file is meant to contain all functions of the detective framework
which are "specific" to the framework but generic among experiments.

For example, all the experiments need to initialize configs, training models,
log stats, display stats, and etc. However, these functions are generally fixed
to this framework and cannot be easily transferred in other projects.
"""

# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from copy import copy
import importlib
import random
import torch
import shutil
import sys
import os
import cv2
import math

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.util import *


def init_config(conf_name):
    """
    Loads configuration file, by checking for the conf_name.py configuration file as
    ./config/<conf_name>.py which must have function "Config".

    This function must return a configuration dictionary with any necessary variables for the experiment.
    """

    conf = importlib.import_module('config.' + conf_name).Config()

    return conf


def init_training_model(conf, cache_folder):
    """
    This function is meant to load the training model and optimizer, which expects
    ./model/<conf.model>.py to be the pytorch model file.

    The function copies the model file into the cache BEFORE loading, for easy reproducibility.
    """

    src_path = os.path.join('.', 'models', conf.model + '.py')
    dst_path = os.path.join(cache_folder, conf.model + '.py')

    # (re-) copy the pytorch model file
    if os.path.exists(dst_path): os.remove(dst_path)
    shutil.copyfile(src_path, dst_path)

    # load and build
    network = absolute_import(src_path)
    network = network.build(conf, 'train')

    # multi-gpu
    network = torch.nn.DataParallel(network)

    # load SGD
    if conf.solver_type.lower() == 'sgd':

        lr = conf.lr
        mo = conf.momentum
        wd = conf.weight_decay

        optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=mo, weight_decay=wd)

    # load adam
    elif conf.solver_type.lower() == 'adam':

        lr = conf.lr
        wd = conf.weight_decay

        optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=wd)

    # load adamax
    elif conf.solver_type.lower() == 'adamax':

        lr = conf.lr
        wd = conf.weight_decay

        optimizer = torch.optim.Adamax(network.parameters(), lr=lr, weight_decay=wd)


    return network, optimizer


def loss_backprop(loss, net, optimizer, conf=None, iteration=None):

    # backprop
    if loss > 0:

        loss.backward()

        clip_value = 50
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
        if grad_norm > clip_value:
           logging.info("iter: {} Skipping gradient update due to high norm {:.2f}".format(iteration, grad_norm))
           optimizer.zero_grad()
           return

        torch.nn.utils.clip_grad_value_(net.parameters(), 1)

        # batch skip, simulates larger batches by skipping gradient step
        if (conf is not None) and (iteration is not None) and \
                ((not 'batch_skip' in conf) or ((iteration + 1) % conf.batch_skip) == 0):

            optimizer.step()
            optimizer.zero_grad()

def adjust_lr(conf, optimizer, iter):
    """
    Adjusts the learning rate of an optimizer according to iteration and configuration,
    primarily regarding regular SGD learning rate policies.

    Args:
        conf (dict): configuration dictionary
        optimizer (object): pytorch optim object
        iter (int): current iteration
    """

    if 'batch_skip' in conf and ((iter + 1) % conf.batch_skip) > 0: return

    if conf.solver_type.lower() == 'sgd' or conf.solver_type.lower() == 'adam':

        lr = conf.lr
        lr_steps = conf.lr_steps
        max_iter = conf.max_iter
        lr_policy = conf.lr_policy
        lr_target = conf.lr_target

        if lr_steps:
            steps = np.array(lr_steps) * max_iter
            total_steps = steps.shape[0]
            step_count = np.sum((steps - iter) <= 0)

        else:
            total_steps = max_iter
            step_count = iter

        warmup = None if 'warmup' not in conf else conf.warmup
        if warmup is not None and step_count <= warmup:
            scale = (step_count+1) / float(warmup)
            lr *= scale

        # perform the exact number of steps needed to get to lr_target
        elif lr_policy.lower() == 'step':
            scale = (lr_target / lr) ** (1 / total_steps)
            lr *= scale ** step_count

        # compute the scale needed to go from lr --> lr_target
        # using a polynomial function instead.
        elif lr_policy.lower() == 'poly':

            power = 0.9 if 'lr_power' not in conf else conf.lr_power
            scale = total_steps / (1 - (lr_target / lr) ** (1 / power))
            lr *= (1 - step_count / scale) ** power

        else:
            raise ValueError('{} lr_policy not understood'.format(lr_policy))

        # update the actual learning rate
        # https://stackoverflow.com/a/48324389
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def intersect(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of intersect between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the intersect in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        # np.ndarray
        if data_type == np.ndarray:
            # print("Using combinations in intersect function on numpy")
            # Calculates the coordinates of the overlap box
            # eg if the box x-coords is at 4 and 5, then the overlap will be minimum
            # of the two which is 4
            # np.maximum is to take two arrays and compute their element-wise maximum.
            # Here, 'compatible' means that one array can be broadcast to the other.
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        elif data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:4], box_b[:, 2:4].unsqueeze(1))
            min_xy = torch.max(box_a[:, 0:2], box_b[:, 0:2].unsqueeze(1))
            inter = torch.clamp((max_xy - min_xy), 0)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = torch.max(box_a[:, :2], box_b[:, :2])
            inter = torch.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.maximum(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))


def iou3d(corners_3d_b1, corners_3d_b2, vol= None):
    """
        Calculates IOU3D of the two set of axis aligned cuboids
        The points in the cuboids should be numpy of the shape (3, 8)
        Axes convention

                Z
               /
              /
             /
            /______ X
            |
            |
            |
            V
            Y

             4 ___________________ 5
              /|                 /|
             / |              1 / |
          0 /__|_______________/  |
            |  |---------------|--|6
            |  /7              |  /
            | /                | /
           2|/_________________|/ 3

    """
    corners_3d_b1 = copy.copy(corners_3d_b1)
    corners_3d_b2 = copy.copy(corners_3d_b2)

    if vol is None:
        vol = get_volume(corners_3d_b1) + get_volume(corners_3d_b2)

    corners_3d_b1 = corners_3d_b1.T
    corners_3d_b2 = corners_3d_b2.T

    y_min_b1 = np.min(corners_3d_b1[:, 1])
    y_max_b1 = np.max(corners_3d_b1[:, 1])
    y_min_b2 = np.min(corners_3d_b2[:, 1])
    y_max_b2 = np.max(corners_3d_b2[:, 1])
    y_intersect = np.max([0, np.min([y_max_b1, y_max_b2]) - np.max([y_min_b1, y_min_b2])])

    # set Z as Y
    corners_3d_b1[:, 1] = corners_3d_b1[:, 2]
    corners_3d_b2[:, 1] = corners_3d_b2[:, 2]

    polygon_order = [7, 2, 3, 6, 7]
    box_b1_bev = Polygon([list(corners_3d_b1[i][0:2]) for i in polygon_order])
    box_b2_bev = Polygon([list(corners_3d_b2[i][0:2]) for i in polygon_order])

    intersect_bev = box_b2_bev.intersection(box_b1_bev).area
    intersect_3d = y_intersect * intersect_bev

    iou_bev = intersect_bev / (box_b2_bev.area + box_b1_bev.area - intersect_bev)
    iou_3d = intersect_3d / (vol - intersect_3d)

    return iou_bev, iou_3d


def iou3d_approximate(corners_3d_b1, corners_3d_b2, mode= "list", method= "normal"):
    """
    :param corners_3d_b1: tensors of shape Nx3x8
    :param corners_3d_b2: tensors of shape Nx3x8
    :param mode: either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
                    combinations: box_a = M x 8 x 3, box_b = N x 8 x 3 then the output is M x N
                    list:         box_a = M x 8 x 3, box_b = M x 8 x 3 then the output is M
    :param method: either 'normal' or 'generalized'
                    normal = usual IOU3D.
                    generalized = generalized IOU3D used in
                        Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression
                        Rezatofighi et al, CVPR 2019
                        https://openaccess.thecvf.com/content_CVPR_2019/papers/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.pdf
                        Generalized IOU code is based on compute_iou() function of https://github.com/generalized-iou/Detectron.pytorch/blob/master/lib/utils/net.py
    :return: iou3d tensor of shape N

            Axes convention

                Z
               /
              /
             /
            /______ X
            |
            |
            |
            V
            Y

             4 ___________________ 5
              /|                 /|
             / |              1 / |
          0 /__|_______________/  |
            |  |---------------|--|6
            |  /7              |  /
            | /                | /
           2|/_________________|/ 3

    """
    if type(corners_3d_b1) == torch.Tensor:
        if corners_3d_b1.dim() > 2:
            N = corners_3d_b1.shape[0]
        else:
            N = 1
            corners_3d_b1 = corners_3d_b1.unsqueeze(0)
            corners_3d_b2 = corners_3d_b2.unsqueeze(0)

        # Calculate volume first
        vol_1 = get_volume(corners_3d_b1)
        vol_2 = get_volume(corners_3d_b2)
        if mode == "combinations":
            vol = vol_1.unsqueeze(1) + vol_2.unsqueeze(0)
        elif mode == "list":
            vol = vol_1 + vol_2

        # Bring them in the format N x 8 x 3
        corners_3d_b1 = corners_3d_b1.transpose(1, 2)
        corners_3d_b2 = corners_3d_b2.transpose(1, 2)

        y_min_b1 = torch.min(corners_3d_b1[:, :, 1], dim=1)[0]
        y_max_b1 = torch.max(corners_3d_b1[:, :, 1], dim=1)[0]
        y_min_b2 = torch.min(corners_3d_b2[:, :, 1], dim=1)[0]
        y_max_b2 = torch.max(corners_3d_b2[:, :, 1], dim=1)[0]

        if mode == "combinations":
            y_intersect_min = torch.max(y_min_b1.unsqueeze(1), y_min_b2.unsqueeze(0))
            y_intersect_max = torch.min(y_max_b1.unsqueeze(1), y_max_b2.unsqueeze(0))
        elif mode == "list":
            y_intersect_min = torch.max(torch.cat((y_min_b1.unsqueeze(1), y_min_b2.unsqueeze(1)), dim=1), dim=1)[0]
            y_intersect_max = torch.min(torch.cat((y_max_b1.unsqueeze(1), y_max_b2.unsqueeze(1)), dim=1), dim=1)[0]
        y_intersect = torch.max(torch.zeros((1,)), y_intersect_max - y_intersect_min)

        # Set Z as Y
        corners_3d_b1[:, :, 1] = corners_3d_b1[:, :, 2]
        corners_3d_b2[:, :, 1] = corners_3d_b2[:, :, 2]

        # Get boxes in X-Z plane
        corners_bev_b1 = corners_3d_b1[:, [2, 3, 6, 7], :2]
        corners_bev_b2 = corners_3d_b2[:, [2, 3, 6, 7], :2]

        # Remove rotation in boxes
        bev_b1 = remove_rotation_in_boxes(corners_bev_b1)
        bev_b2 = remove_rotation_in_boxes(corners_bev_b2)

        if method == "generalized":
            y_hull = get_hull(y_min_b1= y_min_b1, y_max_b1= y_max_b1, y_min_b2= y_min_b2, y_max_b2= y_max_b2, mode= mode)

            x_min_b1 = torch.min(corners_bev_b1[:, :, 0], dim=1)[0]
            x_max_b1 = torch.max(corners_bev_b1[:, :, 0], dim=1)[0]
            x_min_b2 = torch.min(corners_bev_b2[:, :, 0], dim=1)[0]
            x_max_b2 = torch.max(corners_bev_b2[:, :, 0], dim=1)[0]
            x_hull = get_hull(y_min_b1= x_min_b1, y_max_b1= x_max_b1, y_min_b2= x_min_b2, y_max_b2= x_max_b2, mode= mode)

            # Using index 1 for selecting since we have already swapped Z and Y axes above
            z_min_b1 = torch.min(corners_bev_b1[:, :, 1], dim=1)[0]
            z_max_b1 = torch.max(corners_bev_b1[:, :, 1], dim=1)[0]
            z_min_b2 = torch.min(corners_bev_b2[:, :, 1], dim=1)[0]
            z_max_b2 = torch.max(corners_bev_b2[:, :, 1], dim=1)[0]
            z_hull = get_hull(y_min_b1= z_min_b1, y_max_b1= z_max_b1, y_min_b2= z_min_b2, y_max_b2= z_max_b2, mode= mode)

            vol_hull = x_hull * y_hull * z_hull

        iou_bev = iou(bev_b1, bev_b2, mode= mode)

        intersect_area_bev = intersect(bev_b1, bev_b2, mode= mode)
        if mode == "combinations":
            # intersect gives in shape N x M. Bring it into M x N
            intersect_area_bev = intersect_area_bev.transpose(1, 0)

        intersect_3d = intersect_area_bev * y_intersect
        union_3d     = (vol - intersect_3d)
        iou_3d = intersect_3d / union_3d
        if method == "generalized":
            iou_3d = iou_3d - ((vol_hull - union_3d)/vol_hull)

    return iou_bev, iou_3d

def get_hull(y_min_b1, y_max_b1, y_min_b2, y_max_b2, mode= "list"):
    if mode == "combinations":
        y_hull_min = torch.min(y_min_b1.unsqueeze(1), y_min_b2.unsqueeze(0))
        y_hull_max = torch.max(y_max_b1.unsqueeze(1), y_max_b2.unsqueeze(0))
    elif mode == "list":
        y_hull_min = torch.min(torch.cat((y_min_b1.unsqueeze(1), y_min_b2.unsqueeze(1)), dim=1), dim=1)[0]
        y_hull_max = torch.max(torch.cat((y_max_b1.unsqueeze(1), y_max_b2.unsqueeze(1)), dim=1), dim=1)[0]
    y_hull = torch.max(torch.zeros((1,)), y_hull_max - y_hull_min)

    return y_hull

def get_volume(corners_3d):
    """
        Calculates volume of a cuboid
        The points in the cuboids should be torch of shape (N, 3, 8) or (3, 8)
        or numpy of the shape (3, 8)
    """
    if type(corners_3d) == torch.Tensor:
        if corners_3d.dim() > 2:
            N = corners_3d.shape[0]
        else:
            N = 1
            corners_3d = corners_3d.unsqueeze(0)

        min_val = torch.min(corners_3d, dim= 2)[0] # N x 3
        max_val = torch.max(corners_3d, dim= 2)[0] # N x 3
        diff = max_val - min_val                   # N x 3

        return torch.prod(diff, dim= 1)            # N

    elif type(corners_3d) == np.ndarray:
        min_val = np.min(corners_3d, axis= 1)
        max_val = np.max(corners_3d, axis= 1)
        diff = max_val - min_val

        return np.prod(diff)
    else:
        raise ValueError("Unknown data type for volume")


def remove_rotation_in_boxes(boxes):
    """
    Removes the rotation present in boxes
    :param boxes: N x 4 x 2
    :return: boxes_without_rotation: N x 4 x1 y1 x2 y2
    """
    x1 = torch.min(boxes[:, :, 0], dim=1)[0]
    x2 = torch.max(boxes[:, :, 0], dim=1)[0]

    y1 = torch.min(boxes[:, :, 1], dim=1)[0]
    y2 = torch.max(boxes[:, :, 1], dim=1)[0]

    boxes_without_rotation = torch.cat((x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1)

    return boxes_without_rotation


def iou(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))

        # torch.Tensor
        if data_type == torch.Tensor:
            union = area_a.unsqueeze(0) + area_b.unsqueeze(1) - inter
            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) - inter
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))


    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))


def iou_ign(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of overap of box_b has within box_a, which is handy for dealing with ignore regions.
    Hence, assume that box_b are ignore regions and box_a are anchor boxes, then we may want to know how
    much overlap the anchors have inside of the ignore regions (hence ignore area_b!)

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 5, box_b = N x 4 then the output is M x N
    # box_a = [(H*num_anchors*W) x 5]
    # box_b = 1 x 4
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))
        union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) * 0 - inter * 0

        # torch and numpy have different calls for transpose
        if data_type == torch.Tensor:
            return (inter / union).permute(1, 0)
        elif data_type == np.ndarray:
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

    else:
        raise ValueError('unknown mode {}'.format(mode))


def freeze_bn(network):
   for name, module in network.named_modules():
       if isinstance(module, torch.nn.BatchNorm2d):
           module.eval()

def slow_bn(network, val=0.01):
   for name, module in network.named_modules():
       if isinstance(module, torch.nn.BatchNorm2d):
           module.momentum = val

def freeze_layers(network, blacklist=None, whitelist=None, verbose=False):

    if blacklist is not None:

        for name, param in network.named_parameters():

            if not any([allowed in name for allowed in blacklist]):
                if verbose:
                    logging.info('freezing {}'.format(name))
                param.requires_grad = False

        for name, module in network.named_modules():
            if not any([allowed in name for allowed in blacklist]):
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    #if verbose:
                    #    logging.info('freezing BN {}'.format(name))

    if whitelist is not None:

        for name, param in network.named_parameters():

            if any([banned in name for banned in whitelist]):
                if verbose:
                    logging.info('freezing {}'.format(name))
                param.requires_grad = False
            #else:
            #    logging.info('NOT freezing {}'.format(name))

        for name, module in network.named_modules():
            if any([banned in name for banned in whitelist]):
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    #if verbose:
                        #logging.info('freezing BN {}'.format(name))


def copy_stats(output, pretrained, redo_anchors=False):


    # copy all available stats from pretrained model
    if (not redo_anchors) and os.path.exists(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'anchors.pkl')):
        copyfile(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'anchors.pkl'), os.path.join(output, 'anchors.pkl'))
    if (not redo_anchors) and  os.path.exists(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'bbox_stds.pkl')):
        copyfile(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'bbox_stds.pkl'), os.path.join(output, 'bbox_stds.pkl'))
    if (not redo_anchors) and  os.path.exists(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'bbox_means.pkl')):
        copyfile(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'bbox_means.pkl'), os.path.join(output, 'bbox_means.pkl'))
    if os.path.exists(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'pose_stds.pkl')):
        copyfile(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'pose_stds.pkl'), os.path.join(output, 'pose_stds.pkl'))
    if os.path.exists(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'pose_means.pkl')):
        copyfile(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'pose_means.pkl'), os.path.join(output, 'pose_means.pkl'))
    if os.path.exists(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'imdb.pkl')):
        copyfile(os.path.join(file_parts(file_parts(pretrained)[0])[0], 'imdb.pkl'), os.path.join(output, 'imdb.pkl'))


def load_weights(model, path, replace_module=False):
    """
    Simply loads a pytorch models weights from a given path.
    """
    dst_weights = model.state_dict()
    src_weights = torch.load(path)

    dst_keys = list(dst_weights.keys())
    src_keys = list(src_weights.keys())

    if replace_module:
        for key in src_keys:
            src_weights[key.replace('module.', '')] = src_weights[key]
            del src_weights[key]

        dst_keys = list(dst_weights.keys())
        src_keys = list(src_weights.keys())

    # remove keys not in dst
    for key in src_keys:
        if key not in dst_keys: del src_weights[key]

    # add keys not in src
    for key in dst_keys:
        if key not in src_keys: src_weights[key] = dst_weights[key]

    model.load_state_dict(src_weights)
    logging.info("=> Loaded model weights {}".format(path))

def log_stats(tracker, iteration, start_time, start_iter, max_iter, skip=1, optional='', lr=0.004):
    """
    This function writes the given stats to the log / prints to the screen.
    Also, computes the estimated time arrival (eta) for completion and (dt) delta time per iteration.

    Args:
        tracker (array): dictionary array tracker objects. See below.
        iteration (int): the current iteration
        start_time (float): starting time of whole experiment
        start_iter (int): starting iteration of whole experiment
        max_iter (int): maximum iteration to go to

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    display_str = 'iter: {}'.format((int((iteration + 1)/skip)))
    display_str += ' lr: {:.6f}'.format(lr)

    # compute eta
    time_str, dt = compute_eta(start_time, iteration - start_iter, max_iter - start_iter)

    # cycle through all tracks
    last_group = ''
    for key in sorted(tracker.keys()):

        if type(tracker[key]) == list:

            # compute mean
            meanval = np.mean(tracker[key])

            # get properties
            format = tracker[key + '_obj'].format
            group = tracker[key + '_obj'].group
            name = tracker[key + '_obj'].name

            # logic to have the string formatted nicely
            # basically roughly this format:
            #   iter: {}, group_1 (name: val, name: val), group_2 (name: val), dt: val, eta: val
            if last_group != group and last_group == '':
                display_str += (', {} ({}: ' + format).format(group, name, meanval)

            elif last_group != group:
                display_str += ('), {} ({}: ' + format).format(group, name, meanval)

            else:
                display_str += (', {}: ' + format).format(name, meanval)

            last_group = group

    # append dt and eta
    display_str += '), dt: {:0.2f}, eta: {}'.format(dt, time_str) + optional

    # log
    logging.info(display_str)


def display_stats(vis, tracker, iteration, start_time, start_iter, max_iter, conf_name, conf_pretty, skip=1):
    """
    This function plots the statistics using visdom package, similar to the log_stats function.
    Also, computes the estimated time arrival (eta) for completion and (dt) delta time per iteration.

    Args:
        vis (visdom): the main visdom session object
        tracker (array): dictionary array tracker objects. See below.
        iteration (int): the current iteration
        start_time (float): starting time of whole experiment
        start_iter (int): starting iteration of whole experiment
        max_iter (int): maximum iteration to go to
        conf_name (str): experiment name used for visdom display
        conf_pretty (str): pretty string with ALL configuration params to display

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    # compute eta
    time_str, dt = compute_eta(start_time, iteration - start_iter, max_iter - start_iter)

    # general info
    info = 'Experiment: <b>{}</b>, Eta: <b>{}</b>, Time/it: {:0.2f}s\n'.format(conf_name, time_str, dt)
    info += conf_pretty

    # replace all newlines and spaces with line break <br> and non-breaking spaces &nbsp
    info = info.replace('\n', '<br>')
    info = info.replace(' ', '&nbsp')

    # pre-formatted html tag
    info = '<pre>' + info + '</pre'

    # update the info window
    vis.text(info, win='info', opts={'title': 'info', 'width': 500, 'height': 350})

    # draw graphs for each track
    for key in sorted(tracker.keys()):

        if type(tracker[key]) == list:
            meanval = np.mean(tracker[key])
            group = tracker[key + '_obj'].group
            name = tracker[key + '_obj'].name

            # new data point
            vis.line(X=np.array([(iteration + 1)]), Y=np.array([meanval]), win=group, name=name, update='append',
                     opts={'showlegend': True, 'title': group, 'width': 500, 'height': 350,
                           'xlabel': 'iteration'})


def compute_stats(tracker, stats):
    """
    Copies any arbitary statistics which appear in 'stats' into 'tracker'.
    Also, for each new object to track we will secretly store the objects information
    into 'tracker' with the key as (group + name + '_obj'). This way we can retrieve these properties later.

    Args:
        tracker (array): dictionary array tracker objects. See below.
        stats (array): dictionary array tracker objects. See below.

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    # through all stats
    for stat in stats:

        # get properties
        name = stat['name']
        group = stat['group']
        val = stat['val']

        # convention for identificaiton
        id = group + name

        # init if not exist?
        if not (id in tracker): tracker[id] = []

        # convert tensor to numpy
        if type(val) == torch.Tensor:
            val = val.cpu().detach().numpy()

        # store
        tracker[id].append(val)

        # store object info
        obj_id = id + '_obj'
        if not (obj_id in tracker):
            stat.pop('val', None)
            tracker[id + '_obj'] = stat


def next_iteration(loader, iterator):
    """
    Loads the next iteration of 'iterator' OR makes a new epoch using 'loader'.

    Args:
        loader (object): PyTorch DataLoader object
        iterator (object): python in-built iter(loader) object
    """

    # create if none
    if iterator == None: iterator = iter(loader)

    # next batch
    try:
        images, imobjs = next(iterator)

    # new epoch / shuffle
    except StopIteration:
        iterator = iter(loader)
        images, imobjs = next(iterator)

    return iterator, images, imobjs


def init_training_paths(conf_name, use_tmp_folder=None):
    """
    Simple function to store and create the relevant paths for the project,
    based on the base = current_working_dir (cwd). For this reason, we expect
    that the experiments are run from the root folder.

    data    =  ./data
    output  =  ./output/<conf_name>
    weights =  ./output/<conf_name>/weights
    results =  ./output/<conf_name>/results
    logs    =  ./output/<conf_name>/log

    Args:
        conf_name (str): configuration experiment name (used for storage into ./output/<conf_name>)
    """

    # make paths
    paths = edict()
    # Keep paths relative. That way pickle imdb can be used on different computers
    paths.base = "."
    paths.data = os.path.join(paths.base, 'data')
    paths.output = os.path.join(paths.base, 'output', conf_name)
    paths.weights = os.path.join(paths.output, 'weights')
    paths.logs = os.path.join(paths.output, 'log')

    if use_tmp_folder: paths.results = os.path.join(paths.base, '.tmp_results', conf_name, 'results')
    else: paths.results = os.path.join(paths.output, 'results')

    # make directories
    mkdir_if_missing(paths.output)
    mkdir_if_missing(paths.logs)
    mkdir_if_missing(paths.weights)
    mkdir_if_missing(paths.results)

    return paths


def init_torch(rng_seed, cuda_seed):
    """
    Initializes the seeds for ALL potential randomness, including torch, numpy, and random packages.

    Args:
        rng_seed (int): the shared random seed to use for numpy and random
        cuda_seed (int): the random seed to use for pytorch's torch.cuda.manual_seed_all function
    """

    # default tensor
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # seed everything
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    torch.cuda.manual_seed_all(cuda_seed)

    # make the code deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_tensors():
    """
    Checks on tensors currently loaded within PyTorch
    for debugging purposes only (esp memory leaks).
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device, obj.shape)
        except:
            pass


def resume_checkpoint(optim, model, weights_dir, iteration):
    """
    Loads the optimizer and model pair given the current iteration
    and the weights storage directory.
    """

    optimpath, modelpath, iteration = checkpoint_names(weights_dir, iteration)

    model.load_state_dict(torch.load(modelpath), strict= False)
    logging.info("=> Loaded model weights {}".format(modelpath))

    try:
        optim.load_state_dict(torch.load(optimpath))
        logging.info("=> Loaded optim weights {}".format(optimpath))
    except:
        logging.info("Not loading optim. Some keys missing !!!")

    return iteration


def save_checkpoint(optim, model, weights_dir, iteration):
    """
    Saves the optimizer and model pair given the current iteration
    and the weights storage directory.
    """

    optimpath, modelpath, _ = checkpoint_names(weights_dir, iteration)

    logging.info("=> Saving optim weights {}".format(optimpath))
    torch.save(optim.state_dict(), optimpath)
    logging.info("=> Saving model weights {}".format(modelpath))
    torch.save(model.state_dict(), modelpath)

    return modelpath, optimpath


def checkpoint_names(weights_dir, iteration):
    """
    Single function to determine the saving format for
    resuming and saving models/optim.
    """
    try:
        # If integer is there, use the current directory
        iteration = int(iteration)
    except:
        # Otherwise first get the directory name and then extract the iteration
        # number from the path
        weights_dir = os.path.dirname(iteration)
        iteration   = int(os.path.basename(iteration).split("_")[1])

    optimpath = os.path.join(weights_dir, 'optim_{}_pkl'.format(iteration))
    modelpath = os.path.join(weights_dir, 'model_{}_pkl'.format(iteration))

    return optimpath, modelpath, iteration


def print_weights(model):
    """
    Simply prints the weights for the model using the mean weight.
    This helps keep track of frozen weights, and to make sure
    they initialize as non-zero, although there are edge cases to
    be weary of.
    """

    # find max length
    max_len = 0
    for name, param in model.named_parameters():
        name = str(name).replace('module.', '')
        if (len(name) + 4) > max_len: max_len = (len(name) + 4)

    # print formatted mean weights
    for name, param in model.named_parameters():
        mdata = np.abs(torch.mean(param.data).item())
        name = str(name).replace('module.', '')

        logging.info(('{0:' + str(max_len) + '} {1:6} {2:6}')
                     .format(name, 'mean={:.4f}'.format(mdata), '    grad={}'.format(param.requires_grad)))


def compute_rel_pose(pose_pre, pose):

    pose_rel = np.linalg.inv(pose).dot(pose_pre)
    #pose_rel = pose_pre - pose

    dx = pose_rel[0, 3]
    dy = pose_rel[1, 3]
    dz = pose_rel[2, 3]

    rx, ry, rz = mat2euler(pose_rel)

    return dx, dy, dz, rx, ry, rz


def inverse_rel_pose(pose_pre, pose_rel):

    pose = pose_pre.dot(np.linalg.inv(pose_rel))

    return pose