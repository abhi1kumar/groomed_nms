

"""
    Sample Run:
    python test/get_oracle_nms.py

    Gives the results when we use different oracle scores. eg 3D IOU and 2D IOU
"""

import os, sys
sys.path.append(os.getcwd())

import argparse

import numpy as np
np.set_printoptions(suppress=True)

from lib.file_io import read_csv
from lib.rpn_util import *

import glob

np.random.seed(0)
num_images = 3769
num_predictions_boxes = -1

ground_truth_folder        = "data/kitti_split1/validation/label_2/"
conf_path                  = "output/kitti_3d_uncertainty/conf.pkl"
p2_folder_relative         = "p2"

def get_oracle_nms(input_folder, prediction_folder_relative, save_folder, scoring_method= "iou_3d"):
    print("Scoring method = {}".format(scoring_method))
    predictions_all = None
    gts_all = None
    scores_new_all = None

    p2_folder = os.path.join(input_folder, p2_folder_relative)

    pattern = os.path.join(input_folder, prediction_folder_relative, "*.txt")
    print("Searching {}".format(pattern))
    prediction_files = sorted(glob.glob(pattern))
    num_prediction_files = len(prediction_files)
    print("Choosing {} files out of {} prediction files for plotting".format(num_images, num_prediction_files))
    if num_predictions_boxes > 0:
        print("Taking {} boxes per image...".format(num_predictions_boxes))
    file_index = np.sort(np.random.choice(range(num_prediction_files), num_images, replace=False))

    for i in range(num_images):
        filename = prediction_files[file_index[i]]
        basename = os.path.basename(filename)
        ground_truth_file_path = os.path.join(ground_truth_folder, basename)

        p2_npy_file = os.path.join(p2_folder, basename.replace(".txt", ".npy"))
        p2 = np.load(p2_npy_file)

        predictions_img = read_csv(filename, ignore_warnings=True, use_pandas= True)
        gt_img = read_csv(ground_truth_file_path, ignore_warnings=True, use_pandas= True)
        num_gt = 0

        if predictions_img is not None and predictions_img.shape[0] > 0 and gt_img is not None:

            # Add dimension if there is a single point
            if gt_img.ndim == 1:
                gt_img = gt_img[np.newaxis, :]

            if predictions_img.ndim == 1:
                predictions_img = predictions_img[np.newaxis, :]

            if num_predictions_boxes > 0:
                predictions_img = predictions_img[:num_predictions_boxes]

            # Remove labels
            labels_img = predictions_img[:, 0]
            labels_index_img = np.zeros(labels_img.shape)
            labels_index_img[labels_img == "Car"]        = 1
            labels_index_img[labels_img == "Pedestrian"] = 2
            labels_index_img[labels_img == "Cyclist"]    = 3

            predictions_img = predictions_img[:, 1:].astype(float)
            gt_img = gt_img[:, 1:].astype(float)

            #        0   1    2     3   4   5  6    7    8    9    10   11   12   13    14      15     16     17
            # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score, width, height, h2d_general  )
            # Add projected 3d center information to predictions_img
            predictions_centers_3d = predictions_img[:, 10:13].T
            predictions_centers_3d_2d = project_3d_points_in_4D_format(p2, predictions_centers_3d, pad_ones=True)
            predictions_centers_3d_2d = predictions_centers_3d_2d[:3].T
            predictions_img = np.hstack((predictions_img, predictions_centers_3d_2d))

            # Add projected 3d center information to gt_img
            gt_centers_3d = gt_img[:, 10:13].T
            gt_centers_3d_2d = project_3d_points_in_4D_format(p2, gt_centers_3d, pad_ones=True)
            gt_centers_3d_2d = gt_centers_3d_2d[:3].T
            gt_img = np.hstack((gt_img, gt_centers_3d_2d))

            row_index = np.where(gt_img[:, 12].astype(int) == -1000)[0]
            gt_img = np.delete(gt_img, row_index, axis=0)

            num_gt = gt_img.shape[0]
            if num_gt > 0:
                # Compute the best overlaps between images
                overlaps = iou(predictions_img[:, 3:7], gt_img[:, 3:7], mode='combinations')
                max_overlaps = np.max(overlaps, axis=1)

                gt_max_overlaps_index = np.argmax(overlaps, axis=1).astype(int)
                gt_matched_img = gt_img[gt_max_overlaps_index].copy()

                suff_overlap_ind = np.where(max_overlaps > 0.01)[0]
                predictions_img  = predictions_img [suff_overlap_ind]
                gt_matched_img   = gt_matched_img  [suff_overlap_ind]
                labels_index_img = labels_index_img[suff_overlap_ind]
                max_overlaps_img = max_overlaps    [suff_overlap_ind]

                predictions_all  = combine(predictions_all, predictions_img)
                gts_all          = combine(gts_all, gt_matched_img)

                N = suff_overlap_ind.shape[0]

                if N > 0 :
                    if scoring_method == "iou_3d":
                        scores_new_img = np.zeros((N,))
                        for j in range(N):
                            _, corners_3d_b1 = project_3d_corners(p2, predictions_img[j, 10], predictions_img[j, 11],
                                                                  predictions_img[j, 12], w3d=predictions_img[j, 8],
                                                                  h3d=predictions_img[j, 7], l3d=predictions_img[j, 9],
                                                                  ry3d=predictions_img[j, 13], iou_3d_convention=True)
                            _, corners_3d_b2 = project_3d_corners(p2, gt_matched_img[j, 10], gt_matched_img[j, 11],
                                                                  gt_matched_img[j, 12], w3d=gt_matched_img[j, 8],
                                                                  h3d=gt_matched_img[j, 7], l3d=gt_matched_img[j, 9],
                                                                  ry3d=gt_matched_img[j, 13], iou_3d_convention=True)
                            _, scores_new_img[j] = iou3d(corners_3d_b1[:3], corners_3d_b2[:3])
                        score_threshold = 0.6

                    elif scoring_method == "iou_3d_approximate":
                        x3d_pred = torch.from_numpy(predictions_img[:, 10]).float()
                        y3d_pred = torch.from_numpy(predictions_img[:, 11]).float()
                        z3d_pred = torch.from_numpy(predictions_img[:, 12]).float()
                        l3d_pred = torch.from_numpy(predictions_img[:,  9]).float()
                        w3d_pred = torch.from_numpy(predictions_img[:,  8]).float()
                        h3d_pred = torch.from_numpy(predictions_img[:,  7]).float()
                        ry3d_pred= torch.from_numpy(predictions_img[:, 13]).float()

                        x3d_gt = torch.from_numpy(gt_matched_img[:, 10]).float()
                        y3d_gt = torch.from_numpy(gt_matched_img[:, 11]).float()
                        z3d_gt = torch.from_numpy(gt_matched_img[:, 12]).float()
                        l3d_gt = torch.from_numpy(gt_matched_img[:,  9]).float()
                        w3d_gt = torch.from_numpy(gt_matched_img[:,  8]).float()
                        h3d_gt = torch.from_numpy(gt_matched_img[:,  7]).float()
                        ry3d_gt= torch.from_numpy(gt_matched_img[:, 13]).float()

                        corners_3d_b1 = get_corners_of_cuboid(x3d= x3d_pred, y3d= y3d_pred, z3d= z3d_pred, w3d= w3d_pred, h3d= h3d_pred, l3d= l3d_pred, ry3d= ry3d_pred)
                        corners_3d_b2 = get_corners_of_cuboid(x3d= x3d_gt  , y3d= y3d_gt  , z3d= z3d_gt  , w3d= w3d_gt  , h3d= h3d_gt  , l3d= l3d_gt  , ry3d= ry3d_gt)
                        _, scores_new_img = iou3d_approximate(corners_3d_b1, corners_3d_b2)
                        scores_new_img    = scores_new_img.numpy()
                        score_threshold   = 0.6

                    elif scoring_method == "iou_2d":
                        scores_new_img = max_overlaps_img
                        score_threshold = conf.score_thres

                    scores_new_all = combine(scores_new_all, scores_new_img)

                    # Do oracle NMS
                    """
                    cls_ind = boxes_img[:, 5].astype(int) - 1
                    x1 = boxes_img[:, 0]
                    y1 = boxes_img[:, 1]
                    x2 = boxes_img[:, 2]
                    y2 = boxes_img[:, 3]
                    score = boxes_img[:, 4]
                    x3d = boxes_img[:, 6]
                    y3d = boxes_img[:, 7]
                    z3d = boxes_img[:, 8]
                    w3d = boxes_img[:, 9]
                    h3d = boxes_img[:, 10]
                    l3d = boxes_img[:, 11]
                    ry3d = boxes_img[:, 12]
                    #        0   1    2     3   4   5  6    7    8    9    10   11   12   13    14
                    # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score)
                    """
                    aboxes = np.zeros((N, 16))
                    aboxes[:, 0:4] = predictions_img[:, 3:7]
                    aboxes[:, 4]   = scores_new_img              # score
                    aboxes[:, 5]   = labels_index_img
                    aboxes[:, 6:9] = predictions_img[:, 10:13]   # x3d, y3d, z3d
                    aboxes[:, 9]   = predictions_img[:, 8]       # w3d
                    aboxes[:, 10]  = predictions_img[:, 7]       # h3d
                    aboxes[:, 11]  = predictions_img[:, 9]       # l3d
                    aboxes[:, 12]  = predictions_img[:, 13]

                    aboxes_2 = get_nms(aboxes, conf, score_threshold= score_threshold)
                    write_image_boxes_to_txt_file(aboxes_2, conf, save_folder= save_folder, id= basename.split(".")[0])

                    num_pred_boxes =  aboxes_2.shape[0]

        else:
            empty_filename = os.path.join(save_folder, basename.split(".")[0] + ".txt")
            os.system("touch " + empty_filename)
            num_pred_boxes = 0

        # print("{} gt_boxes= {} pred_boxes= {}".format(basename, num_gt, num_pred_boxes))

        if (i + 1) % 250 == 0:
            print("{} images done".format(i + 1))

    print("Running evaluation on {}".format(save_folder))
    evaluate_kitti_results_verbose(data_folder= "./data", test_dataset_name= "kitti_split1", results_folder= "./" + save_folder, split_name= "validation", test_iter= "test", conf= conf, use_logging= False)

#===============================================================================
# Argument Parsing
#===============================================================================
ap      = argparse.ArgumentParser()
ap.add_argument('-i', '--input_folder'   ,                 default= 'output/kitti_3d_uncertainty/results/results_before_nms', help= 'path of the input folder')
args    = ap.parse_args()

conf = pickle_read(conf_path)

oracle_nms_prediction_folder_relative = "data"
prediction_folder_relative            = "data"
save_folder= os.path.join('output/kitti_3d_uncertainty/results/results_test', oracle_nms_prediction_folder_relative)
mkdir_if_missing(save_folder)

get_oracle_nms(args.input_folder, prediction_folder_relative, save_folder, scoring_method = "iou_3d")
# Faster version of IOU_3D.
# get_oracle_nms(args.input_folder, prediction_folder_relative, save_folder, scoring_method = "iou_3d_approximate")

get_oracle_nms(args.input_folder, prediction_folder_relative, save_folder, scoring_method = "iou_2d")