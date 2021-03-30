

"""
    Sample Run:
    python plot/visualize_boxes_on_image.py -i output/kitti_acceptance_prob_overlap_lr_3e-3_freeze_hard_2/results/results_test

    Reads the prediction files and the ground truth files for split 1 and then plots the 2D boxes on the image
    Can be useful to determine if there is some falcy in the prediction which could be leveraged.
    TODO:
    Save the images
    Show the 3D Boxes as well
"""
import os, sys
sys.path.append(os.getcwd())

import argparse

import numpy as np
np.set_printoptions(suppress=True)

import plot.plotting_params as params
from plot.common_operations import *
import matplotlib.pyplot as plt
from lib.file_io import *
from lib.math_3d import project_3d_corners
from lib.util import draw_3d_box

import glob

image_folder               = "data/kitti_split1/validation/image_2/"
ground_truth_folder        = "data/kitti_split1/validation/label_2/"
p2_folder                  = "output/run_4_on_refact_1/results/results_test/p2"

np.random.seed(0)
num_images = 200

thresh_height = 0.3
thresh_area   = 0.002

def plot_boxes(ax, boxes, img_width, img_height, edgecolor= 'r', is_predictions= True, p2= None):
    if boxes is not None and boxes.size > 0:
        if boxes.ndim == 1:
            boxes = boxes[np.newaxis, :]

        # print("")
        for j in range(boxes.shape[0]):
            #   0    1    2   3     4   5  6    7    8   9    10   11   12   13   14    15     16     17
            # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score, width, height, h2d_general  )
            if boxes[j,0] == 'Car' or boxes[j,0] == 'Cyclist' or boxes[j,0] == 'Pedestrian':
                x1 = boxes[j, 4]
                y1 = boxes[j, 5]
                x2 = boxes[j, 6]
                y2 = boxes[j, 7]
                rect_width  = x2 - x1
                rect_height = y2 - y1

                z3d = boxes[j, 13]
                h3d = boxes[j, 8]
                flag     = True
                boxcolor = edgecolor
                if is_predictions:
                    # print("{:d}, {:.2f} {:.2f} {:.2f} {:.2f} {:.5f}".format(j, x1, y1, rect_width / img_width, rect_height / img_height, (rect_width * rect_height) / (img_width * img_height)))
                    plt.text(x1, y1, str(j) + ", {:.2f}".format(z3d), fontsize= 10, color= 'r')
                    if y1 < thresh_height*img_height or y1 > (1 - thresh_height)*img_height:
                        if rect_width*rect_height < thresh_area * img_width*img_height:
                            boxcolor = 'r'
                else:
                    plt.text(x1, y2, ", {:.2f}".format(z3d), fontsize= 10, color= 'r')
                    if z3d < 8:
                        boxcolor = 'r'

                if flag:
                    draw_rectangle(ax, img_width= img_width, img_height= img_height, rect_x_left= x1, rect_y_left= y1,
                               rect_width= rect_width, rect_height= rect_height, edgecolor= boxcolor)

                    if is_predictions and p2 is not None:
                        corners_2D, _ = project_3d_corners(p2= p2, x3d=boxes[j, 11], y3d= boxes[j, 12] - h3d/2,
                                                           z3d= z3d, w3d = boxes[j, 9], h3d= h3d,
                                                           l3d= boxes[j, 10], ry3d= boxes[j, 14], iou_3d_convention=True,
                                                           return_in_4D=False)
                        corners_2D = corners_2D.T #8 x 4
                        draw_3d_box(im= ax, verts= corners_2D, color= "orange", thickness= 1, iou_3d_convention= True, multi_color= True)
    return ax

#===============================================================================
# Argument Parsing
#===============================================================================
ap      = argparse.ArgumentParser()
ap.add_argument('-i', '--input_folder'   ,                 default= 'output/kitti_acceptance_prob_overlap_lr_3e-3_freeze_hard_2/results/results_test', help= 'path of the input folder')
args    = ap.parse_args()

prediction_folders_relative = ["data"] #["data_oracle_nms_816", "data_before_nms"]
input_folder = args.input_folder

for prediction_folder_relative in prediction_folders_relative:
    prediction_files     = sorted(glob.glob(os.path.join(input_folder, prediction_folder_relative + "/*.txt")))

    num_prediction_files = len(prediction_files)
    print("Choosing {} files out of {} prediction files for plotting".format(num_images, num_prediction_files))
    file_index = np.arange(num_images) #np.sort(np.random.choice(range(num_prediction_files), num_images, replace=False))

    for i in range(num_images):
        filename = prediction_files[file_index[i]]
        basename = os.path.basename(filename)
        ground_truth_file_path = os.path.join(ground_truth_folder, basename)
        img_path               = os.path.join(image_folder, basename.replace(".txt", ".png"))

        predictions_img = read_csv(filename, ignore_warnings=True, use_pandas= True)
        gt_img          = read_csv(ground_truth_file_path, ignore_warnings=True, use_pandas= True)
        rgb_img         = imread(img_path)[:, :, ::-1] # H x W x 3

        try:
            p2_path     = os.path.join(p2_folder   , basename.replace(".txt", ".npy"))
            p2          = read_numpy(p2_path)
        except:
            p2          = None

        # Create figure and axes
        fig, ax = plt.subplots(1)
        ax.imshow(rgb_img)

        plt.title(basename)
        ax = plot_boxes(ax= ax, boxes= gt_img         , img_width= rgb_img.shape[1], img_height= rgb_img.shape[0], edgecolor= "limegreen"      , is_predictions= False, p2= p2)
        ax = plot_boxes(ax= ax, boxes= predictions_img, img_width= rgb_img.shape[1], img_height= rgb_img.shape[0], edgecolor= params.dodge_blue, is_predictions= True , p2= p2)

        plt.show()
        plt.close()