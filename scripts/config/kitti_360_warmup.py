from easydict import EasyDict as edict
import numpy as np

def Config():

    conf = edict()
        
    # ----------------------------------------
    #  general
    # ----------------------------------------

    conf.model = 'densenet121_3d_dilate_decomp_alpha'
    
    # solver settings
    conf.solver_type = 'adam'
    conf.lr = 0.0004
    conf.momentum = 0.9
    conf.weight_decay = 0.0005
    conf.max_iter = 80000
    conf.warmup = 3000
    conf.snapshot_iter = 20000
    conf.display = 50
    conf.do_test = True
    
    conf.fast_eval = True
    
    # sgd parameters
    conf.lr_policy = 'poly'
    conf.lr_steps = None
    conf.lr_target = conf.lr * 0.00001

    conf.pretrained = 'output/kitti_360_warmup_100/weights/model_20000_pkl'
    # conf.copy_stats = True

    # random
    conf.rng_seed = 5
    conf.cuda_seed = 8
    
    # misc network
    conf.image_means = [0.485, 0.456, 0.406]
    conf.image_stds = [0.229, 0.224, 0.225]
    conf.feat_stride = 16
    
    conf.has_3d = True
    conf.has_un = False
    
    conf.decomp_alpha = True

    # ----------------------------------------
    #  image sampling and datasets
    # ----------------------------------------

    # scale sampling  
    conf.test_scale = 384
    conf.crop_size = [384, 1438]
    conf.mirror_prob = 0.5
    conf.distort_prob = -1
    
    # datasets
    conf.dataset_test = 'kitti_split1'
    conf.datasets_train = [{'name': 'kitti_split1', 'anno_fmt': 'kitti_det', 'im_ext': '.png', 'scale': 1}]
    conf.use_3d_for_2d = True
    
    # percent expected height ranges based on test_scale
    # used for anchor selection 
    conf.percent_anc_h = [0.0625, 0.75]
    
    # labels settings
    conf.min_gt_h = conf.test_scale*conf.percent_anc_h[0]
    conf.max_gt_h = conf.test_scale*conf.percent_anc_h[1]
    conf.min_gt_vis = 0.65
    conf.ilbls = ['Van', 'ignore']
    conf.lbls = ['Car', 'Building']
    
    # ----------------------------------------
    #  detection sampling
    # ----------------------------------------
    
    # detection sampling
    conf.batch_size = 16
    conf.fg_image_ratio = 1.0
    conf.box_samples = 0.20
    conf.fg_fraction = 0.20
    conf.bg_thresh_lo = 0
    conf.bg_thresh_hi = 0.5
    conf.fg_thresh = 0.5
    conf.ign_thresh = 0.5
    conf.best_thresh = 0.35

    # ----------------------------------------
    #  inference and testing
    # ----------------------------------------

    # nms
    conf.nms_topN_pre = 3000
    conf.nms_topN_post = 40
    conf.nms_thres = 0.4
    conf.clip_boxes = False

    conf.test_protocol = 'kitti'
    conf.test_db = 'kitti'
    conf.test_min_h = 0
    conf.min_det_scales = [0, 0]

    # ----------------------------------------
    #  anchor settings
    # ----------------------------------------
    
    # clustering settings
    conf.cluster_anchors = 0
    conf.even_anchors = 0
    conf.expand_anchors = 0
                             
    conf.anchors = None

    conf.bbox_means = None
    conf.bbox_stds = None
    
    # initialize anchors
    base = (conf.max_gt_h / conf.min_gt_h) ** (1 / (12 - 1))
    conf.anchor_scales = np.array([conf.min_gt_h * (base ** i) for i in range(0, 12)])
    conf.anchor_ratios = np.array([0.5, 1.0, 1.5])
    
    # loss logic
    conf.hard_negatives = True
    conf.focal_loss = 0
    conf.cls_2d_lambda = 1
    conf.iou_2d_lambda = 1
    conf.bbox_2d_lambda = 0
    conf.bbox_3d_lambda = 1
    conf.bbox_axis_head_lambda = 0.35
    
    conf.bbox_un_lambda = 0
    conf.infer_2d_from_3d = False
    conf.score_thres = 0.75

    return conf

