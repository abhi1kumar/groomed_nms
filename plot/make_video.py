

import os,sys
sys.path.append(os.getcwd())

import cv2
import glob

VIDEO_DIR       = "images"
video_name      = "demo.avi"
frame_rate      = 30
img_save_folder = "images/qualitative/video_demo"

#===============================================================================
# Now make the video by loading the saved images
#===============================================================================
img_array       = []
saved_files     = sorted(glob.glob(img_save_folder + "/*.png"))
video_save_path = os.path.join(VIDEO_DIR, video_name)

# Reference
# https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
for i, filename in enumerate(saved_files):
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

    if i > 10:
        break

out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("\nSaved output video at {} fps to {}".format(frame_rate, video_save_path))