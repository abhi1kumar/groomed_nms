

import os,sys
sys.path.append(os.getcwd())

import argparse
from plot.common_operations import *

#===============================================================================
# Argument Parsing
#===============================================================================
ap      = argparse.ArgumentParser()
ap.add_argument('-i', '--input_folder'   , nargs='+', default= ["output/groumd_nms/results/results_test"], help= 'path of the input folder')
ap.add_argument('-l', '--labels'         , nargs='+', default= ["GrooMeD NMS"], help= 'labels of the input folder')
ap.add_argument('-s', '--save_prefix'                    , default=None  , help= 'save file prefix')
ap.add_argument('-t', '--threshold_score', type=   float , default= 0.0  , help= 'threshold on score. Greater than this value are kept')
ap.add_argument('-d', '--threshold_depth', type=   float , default= 100.0, help= 'threshold on z_depth. Less than this value are kept')
args    = ap.parse_args()

input_folders     = args.input_folder
labels            = args.labels
threshold_score   = args.threshold_score
threshold_depth   = args.threshold_depth
num_input_folders = len(input_folders)

input_folders     = args.input_folder
labels            = args.labels
threshold_score   = args.threshold_score
threshold_depth   = args.threshold_depth
num_input_folders = len(input_folders)

if args.save_prefix is None:
    ymin3d = 0.9
    ymin2d = 0.8
    ystep = 0.02
    ydelta = 0.002
elif "before_after_nms" in args.save_prefix:
    ymin3d = 0.6
    ymin2d = 0
    ystep = 0.08
    ydelta = 0.008
elif "with_without" in args.save_prefix:
    ymin3d = 0.96
    ymin2d = 0.96
    ystep = 0.008
    ydelta = 0.0008

prediction_folders_relative = ["data"]#, "data_before_nms"]
num_images            = -1
iou_on_x              = True
num_predictions_boxes = 500
display_frequency     = 250
num_bins              = 25

throw_samples_flag    = False
throw_samples_in_bin  = False
frac_to_keep          = 0.7
show_message          = False
colors_list           = ["purple", "orange", "dodgerblue", "red", "cyan"]
colors_list[num_input_folders-1] = "red"

print("Number of models to plot = {}".format(num_input_folders))
print("Number of bins = {}".format(num_bins))
bins = np.arange(num_bins+1)/float(num_bins)

plt.figure(figsize= params.size, dpi= params.DPI)
for i in range(num_input_folders):
    for prediction_folder_relative in prediction_folders_relative:
        curr_folder = input_folders[i]
        full_folder_path = os.path.join(curr_folder, prediction_folder_relative)
        if os.path.exists(full_folder_path):
            big_list = read_folder_and_get_all_errors(curr_folder, prediction_folder_relative, num_images= num_images, num_predictions_boxes= num_predictions_boxes, threshold_score= threshold_score, threshold_depth= threshold_depth)
            iou_3d_all = big_list[0]
            score      = big_list[2]
            occlusion  = big_list[5]
            # 1    occluded     Integer (0,1,2,3) indicating occlusion state:
            # 0 = fully visible, 1 = partly occluded
            # 2 = largely occluded, 3 = unknown
            index = np.where(occlusion <= 2)[0]
            # index = np.where(np.logical_and(occlusion > 0, occlusion <= 2))[0]
            iou_3d_all = iou_3d_all[index]
            score      = score[index]
            plot_one_error_with_box_error(score, iou_3d_all, name=r"Box Confidence", plt= plt, show_message= show_message, throw_samples_flag= throw_samples_flag, throw_samples_in_bin= throw_samples_in_bin, iou_on_x= iou_on_x, do_decoration= False, label= labels[i], color= colors_list[i], bins= bins)
        else:
            print("{} DNE".format(full_folder_path))

plt.grid(True)
xticks = np.arange(0.0, 1.001, step=0.1)
xticklabels = [str(round(float(x),1)) if i%2 == 0 else "" for i,x in enumerate(xticks.tolist()) ]
plt.xticks(xticks, xticklabels)
plt.yticks(np.arange(ymin3d, 1 + ydelta, step= ystep))
plt.xlim((0, 1.0))
plt.ylim(bottom= ymin3d-ydelta, top= 1 + ydelta)
plt.xlabel(r"Box IoU$_{3D}$")
plt.ylabel("Box Confidence")
plt.legend(loc= "lower right", fontsize= params.legend_fs+4, borderaxespad= params.legend_border_axes_plot_border_pad, labelspacing= params.legend_vertical_label_spacing)#, borderpad= params.legend_border_pad, handletextpad= params.legend_marker_text_spacing+1)
save_file = "scores_vs_iou3d_occ_gt_1.png"
if args.save_prefix is not None:
    save_file = save_file.split(".")[0] + "_"  + args.save_prefix + ".png"
save_path = os.path.join(params.IMAGE_DIR, save_file)
print("")
savefig(plt, save_path)
plt.close()