"""
This file is meant to contain generic utility functions
which can be easily re-used in any project, and are not
specific to any project or framework (except python!).
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
import copy
import cv2
#from scipy.spatial.transform import Rotation as scipy_R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import torch

def copyfile(src, dst):
    shutil.copyfile(src, dst)


def pretty_print(name, input, val_width=40, key_width=0):
    """
    This function creates a formatted string from a given dictionary input.
    It may not support all data types, but can probably be extended.

    Args:
        name (str): name of the variable root
        input (dict): dictionary to print
        val_width (int): the width of the right hand side values
        key_width (int): the minimum key width, (always auto-defaults to the longest key!)

    Example:
        pretty_str = pretty_print('conf', conf.__dict__)
        pretty_str = pretty_print('conf', {'key1': 'example', 'key2': [1,2,3,4,5], 'key3': np.random.rand(4,4)})

        print(pretty_str)
        or
        logging.info(pretty_str)
    """

    # root
    pretty_str = name + ': {\n'

    # determine key width
    for key in input.keys(): key_width = max(key_width, len(str(key)) + 4)

    # cycle keys
    for key in input.keys():

        val = input[key]

        # round values to 3 decimals..
        if type(val) == np.ndarray: val = np.round(val, 3).tolist()

        # difficult formatting
        val_str = str(val)
        if len(val_str) > val_width:
            val_str = pprint.pformat(val, width=val_width, compact=True)
            val_str = val_str.replace('\n', '\n{tab}')
            tab = ('{0:' + str(4 + key_width) + '}').format('')
            val_str = val_str.replace('{tab}', tab)

        # more difficult formatting
        format_str = '{0:' + str(4) + '}{1:' + str(key_width) + '} {2:' + str(val_width) + '}\n'
        pretty_str += format_str.format('', key + ':', val_str)

    # close root object
    pretty_str += '}'

    return pretty_str


def absolute_import(file_path):
    """
    Imports a python module / file given its ABSOLUTE path.

    Args:
         file_path (str): absolute path to a python file to attempt to import
    """

    # module name
    _, name, _ = file_parts(file_path)

    # load the spec and module
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def init_log_file(folder_path, suffix=None, log_level=logging.INFO):
    """
    This function inits a log file given a folder to write the log to.
    it automatically adds a timestamp and optional suffix to the log.
    Anything written to the log will automatically write to console too.

    Example:
        import logging

        init_log_file('output/logs/')
        logging.info('this will show up in both the log AND console!')
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_format = '[%(levelname)s]: %(asctime)s %(message)s'

    if suffix is not None:
        file_name = timestamp + '_' + suffix
    else:
        file_name = timestamp

    file_path = os.path.join(folder_path, file_name)
    logging.basicConfig(filename=file_path, level=log_level, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    return file_path


def denorm_image(im, image_means, image_stds):

    im = copy.deepcopy(im)
    im[:, :, 0] *= image_stds[0]
    im[:, :, 1] *= image_stds[1]
    im[:, :, 2] *= image_stds[2]

    im[:, :, 0] += image_means[0]
    im[:, :, 1] += image_means[1]
    im[:, :, 2] += image_means[2]

    return im

def compute_eta(start_time, idx, total):
    """
    Computes the estimated time as a formatted string as well
    as the change in delta time dt.

    Example:
        from time import time

        start_time = time()

        for i in range(0, total):
            <lengthly computation>
            time_str, dt = compute_eta(start_time, i, total)
    """

    dt = (time() - start_time)/idx
    timeleft = np.max([dt * (total - idx), 0])
    if timeleft > 3600: time_str = '{:.1f}h'.format(timeleft / 3600);
    elif timeleft > 60: time_str = '{:.1f}m'.format(timeleft / 60);
    else: time_str = '{:.1f}s'.format(timeleft);

    return time_str, dt


def interp_color(dist, bounds=[0, 1], color_lo=(0,0, 250), color_hi=(0, 250, 250)):

    percent = (dist - bounds[0]) / (bounds[1] - bounds[0])
    b = color_lo[0] * (1 - percent) + color_hi[0] * percent
    g = color_lo[1] * (1 - percent) + color_hi[1] * percent
    r = color_lo[2] * (1 - percent) + color_hi[2] * percent

    return (b, g, r)

def create_colorbar(height, width, color_lo=(0,0, 250), color_hi=(0, 250, 250)):

    im = np.zeros([height, width, 3])

    for h in range(0, height):

        color = interp_color(h + 0.5, [0, height], color_hi, color_lo)
        im[h, :, 0] = (color[0])
        im[h, :, 1] = (color[1])
        im[h, :, 2] = (color[2])

    return im.astype(np.uint8)


def mkdir_if_missing(directory, delete_if_exist=False):
    """
    Recursively make a directory structure even if missing.

    if delete_if_exist=True then we will delete it first
    which can be useful when better control over initialization is needed.
    """

    if delete_if_exist and os.path.exists(directory): shutil.rmtree(directory)

    # check if not exist, then make
    if not os.path.exists(directory):
        os.makedirs(directory)


def list_files(base_dir, file_pattern):
    """
    Returns a list of files given a directory and pattern
    The results are sorted alphabetically

    Example:
        files = list_files('path/to/images/', '*.jpg')
    """

    return sorted(glob(os.path.join(base_dir) + file_pattern))


def file_parts(file_path):
    """
    Lists a files parts such as base_path, file name and extension

    Example
        base, name, ext = file_parts('path/to/file/dog.jpg')
        print(base, name, ext) --> ('path/to/file/', 'dog', '.jpg')
    """

    base_path, tail = os.path.split(file_path)
    name, ext = os.path.splitext(tail)

    return base_path, name, ext


def pickle_write(file_path, obj):
    """
    Serialize an object to a provided file_path
    """

    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def pickle_read(file_path):
    """
    De-serialize an object from a provided file_path
    """

    with open(file_path, 'rb') as file:
        return pickle.load(file)


def get_color(ind, hex=False):

    colors = [(111, 74, 0),
        (81, 0, 81),
        (128, 64, 128),
        (230, 150, 140),
        (70, 70, 70),
        (102, 102, 156),
        (0, 0, 90),
        (190, 153, 153),
        (244, 35, 232),
        (150, 120, 90),
        (220, 220, 0),
        (107, 142, 35),
        (180, 165, 180),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (250, 170, 160),
        (0, 0, 142),
        (0, 0, 70),
        (150, 100, 100),
        (0, 60, 100),
        (0, 0, 110),
        (153, 153, 153),
        (0, 80, 100),
        (250, 170, 30),
        (0, 0, 230),
        (119, 11, 32),
        (0, 0, 142), ]

    color = colors[ind % len(colors)]

    if hex:
        return '#%02x%02x%02x' % (color[0], color[1], color[2])
    else:
        return color


def draw_3d_box(im, verts, color=(0, 200, 200), thickness=1):

    for lind in range(0, verts.shape[0] - 1):
        v1 = verts[lind]
        v2 = verts[lind + 1]
        cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)

    draw_transparent_polygon(im, verts[5:9, :], blend=0.5, color=color)


def get_polygon_grid(im, poly_verts):

    from matplotlib.path import Path

    nx = im.shape[1]
    ny = im.shape[0]
    #poly_verts = [(1, 1), (5, 1), (5, 9), (3, 2), (1, 1)]

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))

    return grid


def draw_bev(canvas_bev, z3d, l3d, w3d, x3d, ry3d, color=(0, 200, 200), scale=1, thickness=2):

    w = l3d * scale
    l = w3d * scale
    x = x3d * scale
    z = z3d * scale
    r = ry3d*-1

    corners1 = np.array([
        [-w / 2, -l / 2, 1],
        [+w / 2, -l / 2, 1],
        [+w / 2, +l / 2, 1],
        [-w / 2, +l / 2, 1]
    ])

    ry = np.array([
        [+math.cos(r), -math.sin(r), 0],
        [+math.sin(r), math.cos(r), 0],
        [0, 0, 1],
    ])

    corners2 = ry.dot(corners1.T).T

    corners2[:, 0] += w/2 + x + canvas_bev.shape[1] / 2
    corners2[:, 1] += l/2 + z

    draw_line(canvas_bev, corners2[0], corners2[1], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[1], corners2[2], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[2], corners2[3], color=color, thickness=thickness)
    draw_line(canvas_bev, corners2[3], corners2[0], color=color, thickness=thickness)


def draw_line(im, v1, v2, color=(0, 200, 200), thickness=1):

    cv2.line(im, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), color, thickness)


def draw_circle(im, pos, radius=5, thickness=1, color=(250, 100, 100), fill=True):

    if fill: thickness = -1

    cv2.circle(im, (int(pos[0]), int(pos[1])), radius, color=color, thickness=thickness)


def draw_transparent_square(im, pos, alpha=1, radius=5, color=(250, 100, 100)):


    l = pos[1] - radius
    r = pos[1] + radius

    t = pos[0] - radius
    b = pos[0] + radius

    if (np.array([l, r, t, b]) >= 0).any():
        l = np.clip(np.floor(l), 0, im.shape[0]).astype(int)
        r = np.clip(np.floor(r), 0, im.shape[0]).astype(int)

        t = np.clip(np.floor(t), 0, im.shape[1]).astype(int)
        b = np.clip(np.floor(b), 0, im.shape[1]).astype(int)

        # blend
        im[l:r + 1, t:b + 1, 0] = im[l:r + 1, t:b + 1, 0] * alpha + color[0] * (1 - alpha)
        im[l:r + 1, t:b + 1, 1] = im[l:r + 1, t:b + 1, 1] * alpha + color[1] * (1 - alpha)
        im[l:r + 1, t:b + 1, 2] = im[l:r + 1, t:b + 1, 2] * alpha + color[2] * (1 - alpha)




def draw_2d_box(im, box, color=(0, 200, 200), thickness=1):

    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    x2 = (x + w) - 1
    y2 = (y + h) - 1

    cv2.rectangle(im, (int(x), int(y)), (int(x2), int(y2)), color, thickness)


def image_mirror(image):

    image = image[:, ::-1, :]
    image = np.ascontiguousarray(image)

    return image


def ego_mirror(ego):

    pose_dx, pose_dy, pose_dz, pose_rx, pose_ry, pose_rz = ego

    pose_dx *= -1
    pose_ry *= -1
    pose_rz *= -1

    while pose_ry > math.pi: pose_ry -= math.pi * 2
    while pose_ry < (-math.pi): pose_ry += math.pi * 2

    while pose_rz > math.pi: pose_rz -= math.pi * 2
    while pose_rz < (-math.pi): pose_rz += math.pi * 2

    return [pose_dx, pose_dy, pose_dz, pose_rx, pose_ry, pose_rz]


def plot_3D_point(fig, point, color=(255, 100, 100)):

    ax = fig.gca(projection='3d')
    ax.scatter(point[0], point[2], 1.65 -point[1], color=(color[0]/255, color[1]/255, color[2]/255))


def imshow(im, fig_num=None):

    if fig_num is not None: plt.figure(fig_num)

    if len(im.shape) == 2:
        im = np.tile(im, [3, 1, 1]).transpose([1, 2, 0])

    plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR))
    plt.show(block=False)


def imwrite(im, path):

    cv2.imwrite(path, im)


def imread(path):

    return cv2.imread(path)


def draw_tick_marks(im, ticks):

    ticks_loc = list(range(0, im.shape[0] + 1, int((im.shape[0]) / (len(ticks) - 1))))

    for tind, tick in enumerate(ticks):
        y = min(max(ticks_loc[tind], 50), im.shape[0] - 10)
        x = im.shape[1] - 115

        draw_text(im, '-{}m'.format(tick), (x, y), lineType=2, scale=1.1, bg_color=None)



def draw_transparent_box(im, box, blend=0.5, color=(0, 255, 255)):

    x_s = int(np.clip(min(box[0], box[2]), a_min=0, a_max=im.shape[1]))
    x_e = int(np.clip(max(box[0], box[2]), a_min=0, a_max=im.shape[1]))
    y_s = int(np.clip(min(box[1], box[3]), a_min=0, a_max=im.shape[0]))
    y_e = int(np.clip(max(box[1], box[3]), a_min=0, a_max=im.shape[0]))

    im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0] * blend + color[0] * (1 - blend)
    im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1] * blend + color[1] * (1 - blend)
    im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2] * blend + color[2] * (1 - blend)


def draw_transparent_polygon(im, verts, blend=0.5, color=(0, 255, 255)):

    mask = get_polygon_grid(im, verts[:4, :])

    im[mask, 0] = im[mask, 0] * blend + (1 - blend) * color[0]
    im[mask, 1] = im[mask, 1] * blend + (1 - blend) * color[1]
    im[mask, 2] = im[mask, 2] * blend + (1 - blend) * color[2]



def draw_text(im, text, pos, scale=0.4, color=(0, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255),
              blend=0.33, lineType=1):

    pos = [int(pos[0]), int(pos[1])]

    if bg_color is not None:

        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(pos[0] + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2]*blend + bg_color[2] * (1 - blend)

        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
# adopted from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def mat2euler(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])

    else: raise ValueError('singular matrix found in mat2euler')

    return np.array([x, y, z])


def fig_to_im(fig):

    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    w, h, d = buf.shape

    im_pil = Image.frombytes("RGBA", (w, h), buf.tostring())
    im_np = np.array(im_pil)[:,:,:3]

    return im_np


def imzoom(im, zoom=0):

    # single value passed in for both axis?
    # extend same val for w, h
    zoom = np.array(zoom)
    if zoom.size == 1: zoom = np.array([zoom, zoom])

    zoom = np.clip(zoom, a_min=0, a_max=0.99)

    cx = im.shape[1]/2
    cy = im.shape[0]/2

    w = im.shape[1]*(1 - zoom[0])
    h = im.shape[0]*(1 - zoom[-1])

    x1 = int(np.clip(cx - w / 2, a_min=0, a_max=im.shape[1] - 1))
    x2 = int(np.clip(cx + w / 2, a_min=0, a_max=im.shape[1] - 1))
    y1 = int(np.clip(cy - h / 2, a_min=0, a_max=im.shape[0] - 1))
    y2 = int(np.clip(cy + h / 2, a_min=0, a_max=im.shape[0] - 1))

    im = im[y1:y2+1, x1:x2+1, :]

    return im


def imhstack(im1, im2):

    sf = im1.shape[0] / im2.shape[0]

    if sf > 1:
        im2 = cv2.resize(im2, (int(im2.shape[1] / sf), im1.shape[0]))
    elif sf < 1:
        im1 = cv2.resize(im1, (int(im1.shape[1] / sf), im2.shape[0]))


    im_concat = np.hstack((im1, im2))

    return im_concat



def imvstack(im1, im2):

    sf = im1.shape[1] / im2.shape[1]

    if sf > 1:
        im2 = cv2.resize(im2, (int(im2.shape[0] / sf), im1.shape[1]))
    elif sf < 1:
        im1 = cv2.resize(im1, (int(im1.shape[0] / sf), im2.shape[1]))

    im_concat = np.vstack((im1, im2))

    return im_concat

# Calculates Rotation Matrix given euler angles.
# adopted from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def euler2mat(x, y, z):

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]
                    ])

    R_y = np.array([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]
                    ])

    R_z = np.array([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def convertAlpha2Rot(alpha, z3d, x3d):


    if type(z3d) == torch.Tensor:

        ry3d = alpha + torch.atan2(-z3d, x3d) + 0.5 * math.pi

        while torch.any(ry3d > math.pi): ry3d[ry3d > math.pi] -= math.pi * 2
        while torch.any(ry3d <= (-math.pi)): ry3d[ry3d <= (-math.pi)] += math.pi * 2

    elif type(z3d) == np.ndarray:
        ry3d = alpha + np.arctan2(-z3d, x3d) + 0.5 * math.pi

        while np.any(ry3d > math.pi): ry3d[ry3d > math.pi] -= math.pi * 2
        while np.any(ry3d <= (-math.pi)): ry3d[ry3d <= (-math.pi)] += math.pi * 2

    else:

        ry3d = alpha + math.atan2(-z3d, x3d) + 0.5 * math.pi
        ry3d = alpha + math.atan2(x3d, z3d) # + 0.5 * math.pi

        while ry3d > math.pi: ry3d -= math.pi * 2
        while ry3d <= (-math.pi): ry3d += math.pi * 2

    return ry3d


def convertRot2Alpha(ry3d, z3d, x3d):

    if type(z3d) == torch.Tensor:

        alpha = ry3d - torch.atan2(-z3d, x3d) - 0.5 * math.pi

        while torch.any(alpha > math.pi): alpha[alpha > math.pi] -= math.pi * 2
        while torch.any(alpha <= (-math.pi)): alpha[alpha <= (-math.pi)] += math.pi * 2

    elif type(z3d) == np.ndarray:
        alpha = ry3d - np.arctan2(-z3d, x3d) - 0.5 * math.pi

        while np.any(alpha > math.pi): alpha[alpha > math.pi] -= math.pi * 2
        while np.any(alpha <= (-math.pi)): alpha[alpha <= (-math.pi)] += math.pi * 2
    else:

        alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
        #alpha = ry3d - math.atan2(x3d, z3d)# - 0.5 * math.pi

        while alpha > math.pi: alpha -= math.pi * 2
        while alpha <= (-math.pi): alpha += math.pi * 2

    return alpha


def combine(storage, new_entry):
    if storage is None:
        storage = new_entry
    else:
        if storage.ndim > 1:
            storage = np.vstack((storage, new_entry))
        else:
            storage = np.hstack((storage, new_entry))

    return storage


def get_correlation(x,y):

    return np.corrcoef(x, y)[0,1]
