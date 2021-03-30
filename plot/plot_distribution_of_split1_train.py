
import os,sys
sys.path.append(os.getcwd())

from lib.imdb_util import *
from plot.common_operations import *
import plot.plotting_params as params

from scripts.config.kitti_3d_warmup import *
conf = Config()

lbls       = conf.lbls
ilbls      = conf.ilbls
min_gt_vis = conf.min_gt_vis
min_gt_h   = conf.min_gt_h
max_gt_h   = conf.max_gt_h

split_names  = ["training", "validation"]
pickle_files = ["imdb.pkl", "imdb_val.pkl"]

for i in range(len(split_names)):
    dataset    = Dataset(conf, "data", "output/kitti_3d_warmup", data_type= split_names[i])
    pkl_path   = os.path.join("output/kitti_3d_warmup", pickle_files[i])
    print("Reading pickle file {}...".format(pkl_path))
    imobjs     = pickle_read(pkl_path)
    num_images = len(imobjs)

    distance = []
    for img_index in range(num_images):
        imobj  = imobjs[img_index]
        gts    = imobj ['gts']
        igns, rmvs = determine_ignores(gts, lbls, ilbls, min_gt_vis, min_gt_h)

        # In imobj, everything in 2D is multiplied by scale and therfore 2D coordinates are ground truths in 512
        gts_all = bbXYWH2Coords(np.array([gt.bbox_full for gt in gts]))
        gts_3d  = np.array([gt.bbox_3d for gt in gts])

        # filter out irrelevant cls, and ignore cls
        gts_val = gts_all[(rmvs == False) & (igns == False), :]
        gts_3d  = gts_3d[(rmvs == False) & (igns == False), :]

        #  0    1     2      3    4    5     6     7     8     9     10      11          12        13         14        15
        # [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY, elevation, alpha_sin, alpha_cos, axis_lbl, head_lbl]
        cz3d = gts_3d[:, 9]

        distance += cz3d.tolist()

    distance = np.array(distance).flatten()

    plt.figure(figsize= (8,6), dpi= params.DPI)
    z_max  = 60
    n_bins = 60
    bins   = np.arange(0, z_max+1, z_max/(n_bins))
    n, _, _ = plt.hist(distance, bins, facecolor= params.color2, alpha=0.75)

    plt.xlabel('Distance (in m)')
    plt.ylabel('Histogram')
    plt.xlim(0, z_max)
    plt.ylim(0, np.ceil(np.max(n)/100)*100)
    plt.grid(True)
    savefig(plt, "images/z_distribution_of_split1_" + split_names[i] + ".png")
    plt.show()
