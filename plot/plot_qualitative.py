
import os, sys

import cv2

sys.path.append(os.getcwd())

import glob
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from lib.math_3d import project_3d
from lib.util import create_colorbar, draw_bev, draw_tick_marks, imhstack, draw_3d_box, get_color, mkdir_if_missing
from lib.file_io import *
np.random.seed(0)

split_name         = "validation"
predictions_folder = "output/groumd_nms/results/results_test_with_pred_times_class"
img_save_folder    = "images/qualitative/val1"
p2_folder          = "/home/abhinav/project/mono_object_detection_July_09/output/run_4_on_refact_1/results/results_test/p2"
num_images         = -1
show_ground_truth  = True
increase_frame_index = 0       # increase_frame_index increases the frame number while saving. Useful for ffmpeg merging videos

# split_name         = "video_demo"
# predictions_folder = "output/groumd_nms/results/results_test"
# img_save_folder    = "images/qualitative/" + split_name
# p2_folder          = os.path.join("data/kitti_split1", split_name, "p2")
# num_images         = len(glob.glob(predictions_folder+"/data/*.txt"))
# show_ground_truth  = False
# increase_frame_index = 994      # increase_frame_index increases the frame number while saving. Useful for ffmpeg merging videos

image_folder       = os.path.join("data/kitti_split1", split_name, "image_2")
ground_truth_folder= os.path.join("data/kitti_split1", split_name, "label_2")
if "video_demo" in split_name:
    print("Running with video_demo settings...")
    zfill_number = 10
    compression_ratio = 50
else:
    zfill_number = 6
    compression_ratio = 25

def plot_boxes_on_image_and_in_bev(predictions_img, plot_color, show_3d= True, show_bev= True, thickness= 6):
    plot_color = plot_color[::-1]
    if predictions_img is not None and predictions_img.size > 0:
        # Add dimension if there is a single point
        if predictions_img.ndim == 1:
            predictions_img = predictions_img[np.newaxis, :]

        N = predictions_img.shape[0]
        #   0   1    2     3   4   5  6    7    8    9    10   11   12   13    14      15     16     17
        # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score, width, height, h2d_general  )
        # Add projected 3d center information to predictions_img
        cls = predictions_img[:, 0]
        h3d = predictions_img[:, 8]
        w3d = predictions_img[:, 9]
        l3d = predictions_img[:, 10]
        x3d = predictions_img[:, 11]
        y3d = predictions_img[:, 12] - h3d/2
        z3d = predictions_img[:, 13]
        ry3d = predictions_img[:,14]

        for j in range(N):
            if cls[j] == "Car":
                if show_3d:
                    verts_cur, corners_3d_cur = project_3d(p2, x3d[j], y3d[j], z3d[j], w3d[j], h3d[j], l3d[j], ry3d[j], return_3d=True)
                    draw_3d_box(im_orig, verts_cur, color= plot_color, thickness= 2)
                if show_bev:
                    draw_bev(canvas_bev, z3d[j], l3d[j], w3d[j], x3d[j], ry3d[j], color= plot_color, scale= bev_scale, thickness= thickness)

#================================================================
# Main starts here
#================================================================

# plotting colors and other stuff
bev_w        = 615
bev_scale    = 20
bev_c1       = (51, 51, 51)#(0, 250, 250)
bev_c2       = (255, 255, 255)#(0, 175, 250)
c_gts        = (10, 175, 10)
c            = (255,51,153)#(255,48,51)#(255,0,0)#(114,211,254) #(252,221,152 # RGB

color_gt     = (153,255,51)#(0, 255 , 0)
color_pred_2 = (51,153,255)#(94,45,255)#(255, 128, 0)

if num_images < 0:
    print("No image supplied..")
    #indices_to_plot = [88, 248, 311, 333, 481, 514, 533, 599, 725, 867, 868, 951, 1043, 1140, 1236, 1424, 1768, 1907, 2013, 2131, 2249, 3094, 3513]
    indices_to_plot = [311, 599, 868, 951, 1140, 1236, 1424, 1768, 2131]
    indices_to_plot = [514, 868, 951, 2249]
    # indices_to_plot = np.arange(836)
    num_images = len(indices_to_plot)
    prediction_files = []
    file_index = np.arange(num_images)
    for i in range(num_images):
        prediction_files.append(os.path.join(predictions_folder, "data", str(indices_to_plot[i]).zfill(zfill_number) + ".txt"))
else:
    prediction_files = sorted(glob.glob(os.path.join(predictions_folder, "data", "*.txt")))
    num_prediction_files = len(prediction_files)
    file_index = np.sort(np.random.choice(range(num_prediction_files), num_images, replace=False))
    print("Choosing {} files out of {} prediction files for plotting".format(num_images, num_prediction_files))
mkdir_if_missing(img_save_folder, delete_if_exist=False)

for i in range(num_images):
    predictions_file = prediction_files[file_index[i]]
    basename = os.path.basename(predictions_file)

    image_file_path  = os.path.join(image_folder, basename.replace(".txt", ".png"))
    p2_npy_file_path = os.path.join(p2_folder, basename.replace(".txt", ".npy"))
    im_orig          = imread(image_file_path)
    predictions_img  = read_csv(predictions_file, ignore_warnings= True, use_pandas= True)
    # p2               = np.array([[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01], [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01], [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03], [0, 0, 0, 1]])
    # p2              = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01], [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01], [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03], [0, 0, 0, 1]])
    p2               = np.load(p2_npy_file_path)

    # make color bar
    canvas_bev = create_colorbar(50 * 20, bev_w, color_lo=bev_c1, color_hi=bev_c2)

    if show_ground_truth:
        ground_truth_file_path = os.path.join(ground_truth_folder, basename)
        gt_img = read_csv(ground_truth_file_path,  ignore_warnings= True, use_pandas= True)

        predictions_file_2 = os.path.join("output/kitti_3d_uncertainty/results/results_test_with_class_pred/data", basename)
        predictions_img_2 = read_csv(predictions_file_2, ignore_warnings= True, use_pandas= True)

        plot_boxes_on_image_and_in_bev(predictions_img_2, plot_color= color_pred_2, show_3d= False, thickness= 8)
        plot_boxes_on_image_and_in_bev(gt_img, plot_color= color_gt, show_3d= False)

    plot_boxes_on_image_and_in_bev(predictions_img, plot_color = c)
    canvas_bev = cv2.flip(canvas_bev, 0)

    # draw tick marks
    ticks = [50, 40, 30, 20, 10, 0]
    draw_tick_marks(canvas_bev, ticks)
    im_concat = imhstack(im_orig, canvas_bev)
    save_path = os.path.join(img_save_folder, str(int(basename.split(".")[0]) + increase_frame_index).zfill(zfill_number) + ".png")
    print("Saving to {}".format(save_path))
    imwrite(im_concat, save_path)

    # Save smaller versions of image
    command = "convert -resize " + str(compression_ratio) + "% " + save_path + " " + save_path
    os.system(command)

# making video
# ffmpeg -framerate 24 -i images/qualitative/video_demo/%10d.png