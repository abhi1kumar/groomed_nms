"""
This file is meant to contain 3D geometry math functions
"""

import os
import sys
from glob import glob
from time import time
import matplotlib.pyplot as plt
import numpy as np
import importlib
import pickle
import logging
import datetime
import pprint
import shutil
import math
import torch
import copy
import cv2
#from scipy.spatial.transform import Rotation as scipy_R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import random
from itertools import combinations


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def project_3d_point(p2, x3d, y3d, z3d):

    coord2d = p2.dot(np.array([x3d, y3d, z3d, 1]))
    coord2d[:2] /= coord2d[2]

    return coord2d[0], coord2d[1], coord2d[2]

def project_3d_points_in_4D_format(p2, points_4d, pad_ones= False):
    """
    Projects 3d points appened with ones to 2d using projection matrix
    :param p2:       np array 4 x 4
    :param points:   np array 4 x N
    :return: coord2d np array 4 x N
    """
    N = points_4d.shape[1]
    z_eps = 1e-2

    if type(points_4d) == np.ndarray:
        if pad_ones:
            points_4d = np.vstack((points_4d, np.ones((1, N))))

        coord2d = np.matmul(p2, points_4d)
        ind = np.where(np.abs(coord2d[2]) > z_eps)
    elif type(points_4d) == torch.Tensor:
        if pad_ones:
            points_4d = torch.cat([points_4d, torch.ones((1, N))], dim= 0)

        coord2d = torch.matmul(p2, points_4d)
        ind = torch.abs(coord2d[2]) > z_eps

    coord2d[:2, ind] /= coord2d[2, ind]

    return coord2d

def backproject_3d_point(p2_inv, x2d, y2d, z2d):

    coord3d = p2_inv.dot(np.array([x2d * z2d, y2d * z2d, z2d, 1]))

    return coord3d[0], coord3d[1], coord3d[2]

def backproject_2d_pixels_in_4D_format(p2_inv, points, pad_ones= False):
    """
        Projects 2d points with x and y in pixels and appened with ones to 3D using inverse of projection matrix
        :param p2_inv:   np array 4 x 4
        :param points:   np array 4 x N or 3 x N
        :param pad_ones: whether to pad_ones or not. 3 X N shaped points need to be padded
        :return: coord2d np array 4 x N
        """
    if pad_ones:
        N = points.shape[1]
        points_4d = np.vstack((points, np.ones((1, N))))
    else:
        points_4d = points

    points_4d[0] = np.multiply(points_4d[0], points_4d[2])
    points_4d[1] = np.multiply(points_4d[1], points_4d[2])

    return np.matmul(p2_inv, points_4d)

def norm(vec):
    return np.linalg.norm(vec)


def get_2D_from_3D(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY):

    if type(cx3d) == torch.Tensor:
        cx3d = cx3d.detach()
        cy3d = cy3d.detach()
        cz3d = cz3d.detach()
        w3d = w3d.detach()
        h3d = h3d.detach()
        l3d = l3d.detach()
        rotY = rotY.detach()

    verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

    if type(cx3d) == np.ndarray:
        ign = np.any(corners_3d[:, 2, :] <= 0, axis=1)
        x = verts3d[:, 0, :].min(axis=1)
        y = verts3d[:, 1, :].min(axis=1)
        x2 = verts3d[:, 0, :].max(axis=1)
        y2 = verts3d[:, 1, :].max(axis=1)

        return np.vstack((x, y, x2, y2)).T, ign

    if type(cx3d) == torch.Tensor:
        ign = torch.any(corners_3d[:, 2, :] <= 0, dim=1)
        x = verts3d[:, 0, :].min(dim=1)[0]
        y = verts3d[:, 1, :].min(dim=1)[0]
        x2 = verts3d[:, 0, :].max(dim=1)[0]
        y2 = verts3d[:, 1, :].max(dim=1)[0]

        return torch.cat((x.unsqueeze(1), y.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1), ign

    else:
        # any boxes behind camera plane?
        ign = np.any(corners_3d[2, :] <= 0)
        x = min(verts3d[:, 0])
        y = min(verts3d[:, 1])
        x2 = max(verts3d[:, 0])
        y2 = max(verts3d[:, 1])

        return np.array([x, y, x2, y2]), ign


def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """
    EPS = 1e-2

    if type(x3d) == np.ndarray:

        p2_batch = np.zeros([x3d.shape[0], 4, 4])
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = np.cos(ry3d)
        ry3d_sin = np.sin(ry3d)

        R = np.zeros([x3d.shape[0], 4, 3])
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = np.zeros([x3d.shape[0], 3, 8])

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = R @ corners_3d

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = p2_batch @ corners_3d

        corners_2d[:, :2, :] /= (corners_2d[:, 2, :][:, np.newaxis, :] + EPS)

        verts3d = corners_2d

    elif type(x3d) == torch.Tensor:

        p2_batch = torch.zeros(x3d.shape[0], 4, 4)
        p2_batch[:, :, :] = p2[np.newaxis, :, :]

        ry3d_cos = torch.cos(ry3d)
        ry3d_sin = torch.sin(ry3d)

        R = torch.zeros(x3d.shape[0], 4, 3)
        R[:, 0, 0] = ry3d_cos
        R[:, 0, 2] = ry3d_sin
        R[:, 1, 1] = 1
        R[:, 2, 0] = -ry3d_sin
        R[:, 2, 2] = ry3d_cos

        corners_3d = torch.zeros(x3d.shape[0], 3, 8)

        # setup X
        corners_3d[:, 0, :] = -l3d[:, np.newaxis] / 2
        corners_3d[:, 0, 1:5] = l3d[:, np.newaxis] / 2

        # setup Y
        corners_3d[:, 1, :] = -h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 2:4] = h3d[:, np.newaxis] / 2
        corners_3d[:, 1, 6:8] = h3d[:, np.newaxis] / 2

        # setup Z
        corners_3d[:, 2, :] = -w3d[:, np.newaxis] / 2
        corners_3d[:, 2, 3:7] = w3d[:, np.newaxis] / 2

        # rotate
        corners_3d = torch.bmm(R, corners_3d)

        corners_3d = corners_3d.to(x3d.device)
        p2_batch = p2_batch.to(x3d.device)

        # translate
        corners_3d[:, 0, :] += x3d[:, np.newaxis]
        corners_3d[:, 1, :] += y3d[:, np.newaxis]
        corners_3d[:, 2, :] += z3d[:, np.newaxis]
        corners_3d[:, 3, :] = 1

        # project to 2D
        corners_2d = torch.bmm(p2_batch, corners_3d)

        corners_2d[:, :2, :] /= (corners_2d[:, 2, :][:, np.newaxis, :] + EPS)

        verts3d = corners_2d

    else:

        # compute rotational matrix around yaw axis
        R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                      [0, 1, 0],
                      [-math.sin(ry3d), 0, +math.cos(ry3d)]])

        # 3D bounding box corners
        x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
        y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
        z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

        x_corners += -l3d / 2
        y_corners += -h3d / 2
        z_corners += -w3d / 2

        # bounding box in object co-ordinate
        corners_3d = np.array([x_corners, y_corners, z_corners])

        # rotate
        corners_3d = R.dot(corners_3d)

        # translate
        corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

        corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
        corners_2D = p2.dot(corners_3D_1)
        corners_2D = corners_2D / (corners_2D[2] + EPS)

        # corners_2D = np.zeros([3, corners_3d.shape[1]])
        # for i in range(corners_3d.shape[1]):
        #    a, b, c, d = argoverse.utils.calibration.proj_cam_to_uv(corners_3d[:, i][np.newaxis, :], p2)
        #    corners_2D[:2, i] = a
        #    corners_2D[2, i] = corners_3d[2, i]

        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

        verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d


def project_3d_corners(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, iou_3d_convention= False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    if iou_3d_convention:
        x_corners = np.array([0, l3d, 0  , l3d, 0  , l3d, l3d, 0])
        y_corners = np.array([0, 0  , h3d, h3d, 0  , 0  , h3d, h3d])
        z_corners = np.array([0, 0  , 0  , 0  , w3d, w3d, w3d, w3d])
    else:
        x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
        y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
        z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

        '''
        order of vertices
        0  upper back right
        1  upper front right
        2  bottom front right
        3  bottom front left
        4  upper front left
        5  upper back left
        6  bottom back left
        7  bottom back right

        bot_inds = np.array([2,3,6,7])
        top_inds = np.array([0,1,4,5])
        '''

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    return corners_2D, corners_3D_1

def get_corners_of_cuboid(x3d, y3d, z3d, w3d, h3d, l3d, ry3d, iou_3d_convention=True):
    if type(x3d) == torch.Tensor:
        N = x3d.shape[0]

        # compute rotational matrix around yaw axis
        R = torch.zeros((N, 3, 3)).float()
        R[:, 0, 0] = torch.cos(ry3d)
        R[:, 0, 2] = torch.sin(ry3d)
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = -torch.sin(ry3d)
        R[:, 2, 2] = torch.cos(ry3d)

        corners = torch.zeros((N, 3, 8)).float()
        # 3D bounding box corners
        if iou_3d_convention:
            '''
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

            '''
            corners[:, 0, [1, 3, 5, 6]] = l3d.unsqueeze(1)
            corners[:, 1, [2, 3, 6, 7]] = h3d.unsqueeze(1)
            corners[:, 2, [4, 5, 6, 7]] = w3d.unsqueeze(1)

        else:
            '''
            order of vertices
            0  upper back right
            1  upper front right
            2  bottom front right
            3  bottom front left
            4  upper front left
            5  upper back left
            6  bottom back left
            7  bottom back right

            bot_inds = np.array([2,3,6,7])
            top_inds = np.array([0,1,4,5])
            '''
            corners[:, 0, [1, 2, 3, 4]] = l3d
            corners[:, 1, [2, 3, 6, 7]] = h3d
            corners[:, 2, [3, 4, 5, 6]] = w3d

        corners[:, 0] -= l3d.unsqueeze(1) / 2
        corners[:, 1] -= h3d.unsqueeze(1) / 2
        corners[:, 2] -= w3d.unsqueeze(1) / 2

        # rotate the batch of boxes using bmm
        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        corners_3d = torch.bmm(R, corners)

        # translate
        corners_3d[:, 0] += x3d.unsqueeze(1)
        corners_3d[:, 1] += y3d.unsqueeze(1)
        corners_3d[:, 2] += z3d.unsqueeze(1)

    elif type(x3d) == np.ndarray:
        N = x3d.shape[0]

        # compute rotational matrix around yaw axis
        R = np.zeros((N, 3, 3)).astype(float)
        R[:, 0, 0] = np.cos(ry3d)
        R[:, 0, 2] = np.sin(ry3d)
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = -np.sin(ry3d)
        R[:, 2, 2] = np.cos(ry3d)

        corners = np.zeros((N, 3, 8)).astype(float)
        # 3D bounding box corners
        if iou_3d_convention:
            '''
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

            '''
            corners[:, 0, [1, 3, 5, 6]] = l3d[:, np.newaxis]
            corners[:, 1, [2, 3, 6, 7]] = h3d[:, np.newaxis]
            corners[:, 2, [4, 5, 6, 7]] = w3d[:, np.newaxis]

        corners[:, 0] -= l3d[:, np.newaxis] / 2
        corners[:, 1] -= h3d[:, np.newaxis] / 2
        corners[:, 2] -= w3d[:, np.newaxis] / 2

        # rotate the batch of boxes using bmm
        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        corners_3d = np.einsum('ijk,ikl->ijl', R, corners)

        # translate
        corners_3d[:, 0] += x3d[:, np.newaxis]
        corners_3d[:, 1] += y3d[:, np.newaxis]
        corners_3d[:, 2] += z3d[:, np.newaxis]

    return corners_3d

def normalize(vec):

    vec /= norm(vec)
    return vec

def snap_to_pi(ry3d):

    if type(ry3d) == torch.Tensor:
        while (ry3d > (math.pi)).any(): ry3d[ry3d > (math.pi)] -= 2 * math.pi
        while (ry3d <= (-math.pi)).any(): ry3d[ry3d <= (-math.pi)] += 2 * math.pi
    elif type(ry3d) == np.ndarray:
        while np.any(ry3d > (math.pi)): ry3d[ry3d > (math.pi)] -= 2 * math.pi
        while np.any(ry3d <= (-math.pi)): ry3d[ry3d <= (-math.pi)] += 2 * math.pi
    else:

        while ry3d > math.pi: ry3d -= math.pi * 2
        while ry3d <= (-math.pi): ry3d += math.pi * 2

    return ry3d

