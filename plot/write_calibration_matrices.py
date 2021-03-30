import glob
import os, sys
sys.path.append(os.getcwd())
import numpy as np

input_folder  = "data/kitti_split1/video_demo"
image_folder  = os.path.join(input_folder, "image_2")
calib_folder  = os.path.join(input_folder, "calib")
p2_npy_folder = os.path.join(input_folder, "p2")

R_rect_00 = [9.999454e-01, 7.259129e-03, -7.519551e-03, -7.292213e-03, 9.999638e-01, -4.381729e-03, 7.487471e-03,  4.436324e-03, 9.999621e-01]
P_rect_02 = [7.188560e+02, 0.000000e+00, 6.071928e+02,   4.538225e+01, 0.000000e+00,  7.188560e+02, 1.852157e+02, -1.130887e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03]

R_rect_temp   = np.array(R_rect_00).astype(float).reshape((3,3))
R_rect        = np.zeros((4, 4))
R_rect[3,3]   = 1
R_rect[:3,:3] = R_rect_temp

P_rect_temp   = np.array(P_rect_02).astype(float).reshape((3,4))
P_rect        = np.zeros((4, 4))
P_rect[3,3]   = 1
P_rect[:3]    = P_rect_temp

P    = np.matmul(P_rect, R_rect)
P3_4 = P[:3].flatten()

# P2: x.12decimale+02
write_string = np.array2string(P3_4, precision= 12, separator=' ', threshold=np.inf, max_line_width=np.inf)
write_string = write_string[1:-1]
write_string = " ".join(write_string.split( ))
write_string = "P2: " + write_string
print(write_string)

prediction_files = sorted(glob.glob(image_folder + "/*.png"))
for i in range(len(prediction_files)):
    filename         = prediction_files[i]
    basename         = os.path.basename(filename)

    output_file_path = os.path.join(calib_folder , basename.replace(".png", ".txt"))
    with open(output_file_path, "w") as text_file:
        text_file.write(write_string)

    npy_file_path    = os.path.join(p2_npy_folder, basename.replace(".png", ".npy"))
    np.save(npy_file_path, P)

    if i%250 == 0 or i== len(prediction_files)-1:
        print("{} images done".format(i))