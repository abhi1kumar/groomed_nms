import torch.nn as nn
import torch.nn.functional as F
import sys
import math

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.rpn_util import *
from lib.loss.aploss import APLoss
from lib.groomed_nms import differentiable_nms
from plot.common_operations import *

class RPN_3D_loss(nn.Module):

    def __init__(self, conf, verbose=True, save_height= False):

        super(RPN_3D_loss, self).__init__()

        self.num_classes = len(conf.lbls) + 1
        self.num_anchors = conf.anchors.shape[0]
        self.anchors = conf.anchors
        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds
        self.feat_stride = conf.feat_stride
        self.fg_fraction = conf.fg_fraction
        self.box_samples = conf.box_samples
        self.ign_thresh = conf.ign_thresh
        self.nms_thres = conf.nms_thres
        self.fg_thresh = conf.fg_thresh
        self.bg_thresh_lo = conf.bg_thresh_lo
        self.bg_thresh_hi = conf.bg_thresh_hi
        self.best_thresh = conf.best_thresh
        self.hard_negatives = conf.hard_negatives
        self.focal_loss = conf.focal_loss

        self.crop_size = conf.crop_size

        self.cls_2d_lambda = conf.cls_2d_lambda
        self.iou_2d_lambda = conf.iou_2d_lambda
        self.bbox_2d_lambda = conf.bbox_2d_lambda
        self.bbox_3d_lambda = conf.bbox_3d_lambda

        self.bbox_axis_head_lambda = 0 if not ('bbox_axis_head_lambda' in conf) else conf.bbox_axis_head_lambda
        self.bbox_3d_iou_lambda = 0 if not ('bbox_3d_iou_lambda' in conf) else conf.bbox_3d_iou_lambda

        self.has_un = 0 if not ('has_un' in conf) else conf.has_un
        self.bbox_un_lambda = 0 if not ('bbox_un_lambda' in conf) else conf.bbox_un_lambda

        self.decomp_alpha = False if not ('decomp_alpha' in conf) else conf.decomp_alpha
        self.bbox_un_dynamic = False if not ('bbox_un_dynamic' in conf) else conf.bbox_un_dynamic
        self.infer_2d_from_3d = False if not ('infer_2d_from_3d' in conf) else conf.infer_2d_from_3d

        self.n_frames = 0

        self.lbls = conf.lbls
        self.ilbls = conf.ilbls

        self.min_gt_vis = conf.min_gt_vis
        self.min_gt_h = conf.min_gt_h
        self.max_gt_h = conf.max_gt_h

        self.torch_bool = hasattr(torch, 'bool')
        self.torch_bool_type = torch.cuda.ByteTensor if not self.torch_bool else torch.cuda.BoolTensor
        self.verbose = verbose
        self.orientation_bins = False if not ('orientation_bins' in conf) else conf.orientation_bins

        self.use_nms_in_loss                        = False         if not('use_nms_in_loss' in conf)                     else conf.use_nms_in_loss
        self.diff_nms_pruning_method                = "linear"      if not('diff_nms_pruning_method' in conf)             else conf.diff_nms_pruning_method
        self.diff_nms_temperature                   = 1             if not('diff_nms_temperature' in conf)                else conf.diff_nms_temperature
        self.diff_nms_valid_box_prob_threshold      = 0.3           if not('diff_nms_valid_box_prob_threshold' in conf)   else conf.diff_nms_valid_box_prob_threshold
        self.diff_nms_boxes_2d                      = "normal"      if not('diff_nms_boxes_2d' in conf)                   else conf.diff_nms_boxes_2d
        self.diff_nms_group_boxes                   = True          if not('diff_nms_group_boxes' in conf)                else conf.diff_nms_group_boxes
        self.diff_nms_mask_group_boxes              = True          if not('diff_nms_mask_group_boxes' in conf)           else conf.diff_nms_mask_group_boxes
        self.diff_nms_group_size                    = 100           if not('diff_nms_group_size' in conf)                 else conf.diff_nms_group_size
        self.after_nms_lambda                       = 1         if not ('after_nms_lambda' in conf)                       else conf.after_nms_lambda
        self.after_nms_loss_mode                    = "rank"    if not ('after_nms_loss_mode' in conf)                    else conf.after_nms_loss_mode
        self.overlap_in_nms                         = "2d"      if not ('overlap_in_nms' in conf)                         else conf.overlap_in_nms

        self.predict_acceptance_prob                = False     if not ('predict_acceptance_prob' in conf)                else conf.predict_acceptance_prob
        self.acceptance_prob_lambda                 = 0         if not ('acceptance_prob_lambda' in conf)                 else conf.acceptance_prob_lambda
        self.acceptance_prob_mode                   = "regress" if not ('acceptance_prob_mode' in conf)                   else conf.acceptance_prob_mode
        self.acceptance_prob_classify_bins          = 2         if not ('acceptance_prob_classify_bins' in conf)          else conf.acceptance_prob_classify_bins
        self.acceptance_prob_classify_num_class     = self.acceptance_prob_classify_bins - 1
        self.acceptance_prob_classify_sort_K        = 0         if not ('acceptance_prob_classify_sort_K' in conf)        else conf.acceptance_prob_classify_sort_K
        self.boxes_for_acceptance_prob              = "all"     if not ('boxes_for_acceptance_prob' in conf)              else conf.boxes_for_acceptance_prob
        self.acceptance_prob_overlap_thres          = 0.01      if not ('acceptance_prob_overlap_thres' in conf)          else conf.acceptance_prob_overlap_thres
        self.use_acceptance_prob_in_regression_loss = False     if not ('use_acceptance_prob_in_regression_loss' in conf) else conf.use_acceptance_prob_in_regression_loss
        self.weigh_acceptance_prob_regularization   = False     if not ('weigh_acceptance_prob_regularization' in conf)   else conf.weigh_acceptance_prob_regularization
        self.rank_with_class_confidence             = False     if not ('rank_with_class_confidence' in conf)             else conf.rank_with_class_confidence
        self.rank_boxes_of_all_images_at_once       = False     if not ('rank_boxes_of_all_images_at_once' in conf)       else conf.rank_boxes_of_all_images_at_once
        self.weigh_3D_regression_loss_by_gt_iou3d   = False     if not ('weigh_3D_regression_loss_by_gt_iou3d' in conf)   else conf.weigh_3D_regression_loss_by_gt_iou3d
        self.best_target_box_beta                   = 0.3       if not ('best_target_box_beta' in conf)                   else conf.best_target_box_beta

        if self.predict_acceptance_prob or self.use_nms_in_loss:
            if self.boxes_for_acceptance_prob == "overlaps":
                logging.info("Using 2D overlaps >= {} for selecting the boxes".format(self.acceptance_prob_overlap_thres))
            else:
                logging.info("Using {} boxes".format(self.boxes_for_acceptance_prob))

        if self.predict_acceptance_prob:
            if self.acceptance_prob_lambda:
                if self.acceptance_prob_mode == "classify":
                    self.bins_boundary = torch.from_numpy(conf.bins_boundary)
                    if self.acceptance_prob_classify_sort_K > 0 and self.acceptance_prob_classify_num_class == 1:
                        logging.info("Classifying top {} boxes to to {} IOU3D bins and {} class...".format(self.acceptance_prob_classify_sort_K, self.acceptance_prob_classify_bins, self.acceptance_prob_classify_num_class))
                    else:
                        logging.info("Classifying ordinally acceptance probabilities to {} IOU3D bins and {} class...".format(self.acceptance_prob_classify_bins, self.acceptance_prob_classify_num_class))
                elif self.acceptance_prob_mode == "rank":
                    logging.info("Ranking acceptance probabilities on IOU3D overlap...")
                    if self.rank_with_class_confidence:
                        logging.info("Ranking acceptance probabilities after multiplying with class confidence...")
                elif self.acceptance_prob_mode == "regress":
                    logging.info("Regressing acceptance probabilities to IOU3D...")
                    if self.weigh_acceptance_prob_regularization:
                        logging.info("Weighing acceptance prob regularization by ground truth IOU3D")
                elif self.acceptance_prob_mode == "likelihood":
                    logging.info("Minimizing negative log likelihood to get acceptance probabilities...")

            if self.use_acceptance_prob_in_regression_loss:
                logging.info("Weighing 3D bbox regression loss by acceptance probability")

        if self.weigh_3D_regression_loss_by_gt_iou3d:
            logging.info("Weighing 3D bbox regression loss by ground truth IOU3D")

        if self.use_nms_in_loss:
            output_string = "Using GrooMeD-NMS with {} pruning, valid_box_prob_threshold= {}".format(self.diff_nms_pruning_method, self.diff_nms_valid_box_prob_threshold)
            if self.diff_nms_pruning_method != "linear":
                output_string += ", temperature {:.2f}".format(self.diff_nms_temperature)
            output_string += " on {} 2D boxes, {} overlaps".format(self.diff_nms_boxes_2d, self.overlap_in_nms)
            output_string += " in the loss..."
            logging.info(output_string)

            if self.diff_nms_group_boxes:
                logging.info("Grouping boxes with max group_size= {} in GrooMeD-NMS".format(self.diff_nms_group_size))
                if self.diff_nms_mask_group_boxes:
                    logging.info("Masking  the prune matrix (P) in groups of GrooMeD-NMS")
            else:
                logging.info("No grouping done")
            if self.after_nms_loss_mode == "classify":
                logging.info("Classifying acceptance probabilities after NMS into 1/0...")
            elif self.after_nms_loss_mode == "rank":
                logging.info("Ranking acceptance probabilities after NMS...")
                if self.rank_with_class_confidence:
                    logging.info("Ranking acceptance probabilities after multiplying with class confidence...")
                if self.rank_boxes_of_all_images_at_once:
                    logging.info("Ranking boxes of all images at once ")
                else:
                    logging.info("Ranking boxes imagewise")
            elif self.after_nms_loss_mode == "regress":
                logging.info("Regressing acceptance probabilities after NMS to 1/0...")

            logging.info("Best Target box beta = {}".format(self.best_target_box_beta))

        self.debug= False
        self.debug_nms_assignment= False
        self.batch_cnt = 0

    def forward(self, cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, rois=None, rois_3d=None, rois_3d_cen=None, bbox_acceptance_prob= None, bbox_acceptance_prob_cls= None, key='gts'):
        """
        Forward pass over the loss

        :param cls              = predicted torch.size([batch x (H*num_anchors*W) x num_classes])
        :param prob             = predicted torch.size([batch x (H*num_anchors*W) x num_classes])
        :param bbox_2d          = predicted torch.size([batch x (H*num_anchors*W) x           4])
        :param bbox_3d          = predicted torch.size([batch x (H*num_anchors*W) x          11])
        :param imobj            = list containing ground truths
        :param feat_size        = [32, 110]
        :param rois             = predicted torch.Size([batch x (H*num_anchors*W) x          5])
        :param rois_3d_new      = predicted torch.Size([batch x (H*num_anchors*W) x         11])
        :param rois_2d_cen_new  = predicted torch.Size([batch x (H*num_anchors*W) x          2])

        :return:
        """
        self.batch_cnt += 1
        stats = []
        loss = torch.tensor(0).type(cls.dtype)

        FG_ENC = 1000
        BG_ENC = 2000
        IGN_FLAG = 3000

        has_vel = False

        batch_size, h_times_w_times_anchors, _ = cls.shape
        apLoss = APLoss()
        #======================================================================
        # Get all rois
        #======================================================================
        if rois is None:
            rois = locate_anchors(self.anchors, feat_size, self.feat_stride, convert_tensor=True)
            rois = rois.type(cls.dtype)

        if rois.shape[0] > batch_size: rois = rois[:batch_size]
        if len(rois.shape) == 2: rois = rois.unsqueeze(0)

        #======================================================================
        # All Predictions
        #======================================================================
        prob_detach = prob.cpu().detach().numpy() #[batch x (H*num_anchors*W) x num_classes]

        if not self.infer_2d_from_3d:
            bbox_x = bbox_2d[:, :, 0]
            bbox_y = bbox_2d[:, :, 1]
            bbox_w = bbox_2d[:, :, 2]
            bbox_h = bbox_2d[:, :, 3]

        bbox_x3d = bbox_3d[:, :, 0]
        bbox_y3d = bbox_3d[:, :, 1]
        bbox_z3d = bbox_3d[:, :, 2]
        bbox_w3d = bbox_3d[:, :, 3]
        bbox_h3d = bbox_3d[:, :, 4]
        bbox_l3d = bbox_3d[:, :, 5]
        bbox_rot3d = torch.zeros(bbox_l3d.shape).type(cls.dtype)

        if self.orientation_bins > 0:
            bbox_alpha = bbox_3d[:, :, 6]
            bbox_alpha_bins = bbox_3d[:, :, 7:]

            bbox_rsin = torch.zeros(bbox_l3d.shape).type(torch.cuda.FloatTensor)
            bbox_rcos = torch.zeros(bbox_l3d.shape).type(torch.cuda.FloatTensor)
            bbox_axis = torch.zeros(bbox_l3d.shape).type(torch.cuda.FloatTensor)
            bbox_head = torch.zeros(bbox_l3d.shape).type(torch.cuda.FloatTensor)

        elif self.decomp_alpha:

            bbox_rsin = bbox_3d[:, :, 6]
            bbox_rcos = bbox_3d[:, :, 7]
            bbox_axis = bbox_3d[:, :, 8]
            bbox_head = bbox_3d[:, :, 9]

            if self.has_un:
                if bbox_acceptance_prob is not None:
                    bbox_acceptance_prob = bbox_acceptance_prob.clamp(min=0.0005)
                else:
                    bbox_acceptance_prob = bbox_3d[:, :, 10:].clamp(min=0.0005)
        else:
            bbox_rot3d = bbox_3d[:, :, 6]
            bbox_un = torch.ones(bbox_z3d.shape, requires_grad=False)

        bbox_x3d_raw  = torch.zeros((batch_size, h_times_w_times_anchors)).type(cls.dtype).cuda()
        bbox_y3d_raw  = torch.zeros((batch_size, h_times_w_times_anchors)).type(cls.dtype).cuda()
        bbox_z3d_raw  = torch.zeros((batch_size, h_times_w_times_anchors)).type(cls.dtype).cuda()
        bbox_ry3d_raw = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()

        #======================================================================
        # All Targets
        #======================================================================
        bbox_x_tar       = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_y_tar       = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_w_tar       = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_h_tar       = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()

        bbox_x3d_tar     = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_y3d_tar     = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_z3d_tar     = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_w3d_tar     = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_h3d_tar     = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_l3d_tar     = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        # We name this as rot3d and not ry3d or alpha since rot3d can be chosen to be either of that
        bbox_rot3d_tar   = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()

        if self.decomp_alpha:
            bbox_axis_tar = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
            bbox_head_tar = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
            bbox_rsin_tar = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
            bbox_rcos_tar = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()

        bbox_x3d_raw_tar  = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_y3d_raw_tar  = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_z3d_raw_tar  = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_l3d_raw_tar  = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_w3d_raw_tar  = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_h3d_raw_tar  = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_ry3d_raw_tar = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_alph_raw_tar = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()
        bbox_alph_raw_tar_2  = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False).type(cls.dtype).cuda()

        coords_2d_512_tar = torch.zeros((batch_size, h_times_w_times_anchors, 4)).type(cls.dtype).cuda()
        ious_2d           = torch.zeros((batch_size, h_times_w_times_anchors))
        abs_err_z         = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False)
        abs_err_rot3d     = torch.zeros((batch_size, h_times_w_times_anchors), requires_grad=False)

        bbox_acceptance_prob_tar = torch.zeros((batch_size, h_times_w_times_anchors)).type(cls.dtype).cuda()
        if self.acceptance_prob_lambda and self.acceptance_prob_mode  ==  "classify":
                bbox_acceptance_prob_logits_tar = torch.zeros((batch_size, h_times_w_times_anchors, self.acceptance_prob_classify_num_class)).type(cls.dtype).cuda()
                self.bins_boundary = self.bins_boundary.type(cls.dtype).cuda()
                bbox_acceptance_prob_logits_wts = torch.zeros(bbox_acceptance_prob_logits_tar.shape).type(cls.dtype).cuda()
        else:
            bbox_acceptance_prob_logits_tar = None
            bbox_acceptance_prob_logits_wts = None

        if self.use_nms_in_loss:
            scores_after_nms  = torch.zeros((batch_size, h_times_w_times_anchors)).type(cls.dtype).cuda()
            targets_after_nms = torch.zeros((batch_size, h_times_w_times_anchors)).type(cls.dtype).cuda()

        labels        = np.zeros(cls.shape[0:2])
        labels_weight = np.zeros(cls.shape[0:2])
        labels_scores = np.zeros(cls.shape[0:2])
        bbox_weights  = np.zeros(cls.shape[0:2])

        accept_prob_box_weight = np.zeros((batch_size, h_times_w_times_anchors))

        #======================================================================
        # Compile 2D and 3D predictions deltas
        # bbox_x3d is transformed version and normalized version of the bounding box.
        # raw_x   --transformation--> denormalized_x  --mu_sigma_normalization--> x
        # Get the original denormalized version by first multiplying by std and
        # adding mean.
        # Then carry out the transformation using equation (3) of the paper
        # https://arxiv.org/pdf/1907.06038.pdf
        #======================================================================
        coords_2d_512 = bbox_transform_inv(rois[0], bbox_2d, means=self.bbox_means[0, :], stds= self.bbox_stds[0, :])  # torch.Size([ (H*num_anchors*W) x          5])

        bbox_x3d_dn = bbox_x3d * self.bbox_stds[:, 4][0] + self.bbox_means[:, 4][0]
        bbox_y3d_dn = bbox_y3d * self.bbox_stds[:, 5][0] + self.bbox_means[:, 5][0]
        bbox_z3d_dn = bbox_z3d * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
        bbox_w3d_dn = bbox_w3d * self.bbox_stds[:, 7][0] + self.bbox_means[:, 7][0]
        bbox_h3d_dn = bbox_h3d * self.bbox_stds[:, 8][0] + self.bbox_means[:, 8][0]
        bbox_l3d_dn = bbox_l3d * self.bbox_stds[:, 9][0] + self.bbox_means[:, 9][0]
        bbox_rot3d_dn = bbox_rot3d * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]

        if self.decomp_alpha:
            bbox_rsin_dn = bbox_rsin * self.bbox_stds[:, 11][0] + self.bbox_means[:, 11][0]
            bbox_rcos_dn = bbox_rcos * self.bbox_stds[:, 12][0] + self.bbox_means[:, 12][0]

        if rois_3d is None:
            rois_3d = self.anchors[rois[:, 4].type(torch.cuda.LongTensor), :]
            rois_3d = torch.tensor(rois_3d, requires_grad=False).type(cls.dtype)

        if rois_3d.shape[0] > batch_size: rois_3d = rois_3d[:batch_size]
        if len(rois_3d.shape) == 2: rois_3d = rois_3d.unsqueeze(0)

        # compute 3d transform
        widths  = rois[:, :, 2] - rois[:, :, 0] + 1.0
        heights = rois[:, :, 3] - rois[:, :, 1] + 1.0
        ctr_x = rois[:, :, 0] + 0.5 * widths
        ctr_y = rois[:, :, 1] + 0.5 * heights

        if rois_3d_cen is None:
            bbox_x3d_dn = bbox_x3d_dn * widths + ctr_x
            bbox_y3d_dn = bbox_y3d_dn * heights + ctr_y
        else:

            if rois_3d_cen.shape[0] > batch_size: rois_3d_cen = rois_3d_cen[:batch_size]
            if len(rois_3d_cen.shape) == 2: rois_3d_cen = rois_3d_cen.unsqueeze(0)

            bbox_x3d_dn = bbox_x3d_dn * widths + rois_3d_cen[:, :, 0]
            bbox_y3d_dn = bbox_y3d_dn * heights + rois_3d_cen[:, :, 1]

        bbox_z3d_dn = rois_3d[:, :, 4] + bbox_z3d_dn
        bbox_w3d_raw = torch.exp(bbox_w3d_dn) * rois_3d[:, :, 5]
        bbox_h3d_raw = torch.exp(bbox_h3d_dn) * rois_3d[:, :, 6]
        bbox_l3d_raw = torch.exp(bbox_l3d_dn) * rois_3d[:, :, 7]

        bbox_rot3d_raw = rois_3d[:, :, 8] + bbox_rot3d_dn

        if self.decomp_alpha:
            bbox_rsin_dn = rois_3d[:, :, 9] + bbox_rsin_dn #torch.asin(bbox_rsin_dn.clamp(min=-0.999, max=0.999))
            bbox_rcos_dn = rois_3d[:, :, 10] + bbox_rcos_dn #torch.acos(bbox_rcos_dn.clamp(min=-0.999, max=0.999)) - math.pi/2

        num_boxes_per_batch = 0
        rois_numpy        = rois   .detach().cpu().numpy()
        rois_3d_numpy     = rois_3d.detach().cpu().numpy()
        rois_2d_cen_numpy = rois_3d_cen.detach().cpu().numpy()

        num_gt            = 0
        fg_index_gt       = []
        #======================================================================
        # Target assignment
        #======================================================================
        for img_index in range(0, batch_size):

            imobj  = imobjs[img_index]
            gts    = imobj [key]
            p2     = torch.from_numpy(imobj.p2)    .type(cls.dtype).cuda()
            fg_num = 0

            #       --           --
            #      | a   0   b  c |
            # p2 = | 0   d   e  f |
            #      | 0   0   1  h |
            #       __          __
            p2_a = imobj.p2[0, 0].item()
            p2_b = imobj.p2[0, 2].item()
            p2_c = imobj.p2[0, 3].item()
            p2_d = imobj.p2[1, 1].item()
            p2_e = imobj.p2[1, 2].item()
            p2_f = imobj.p2[1, 3].item()
            p2_h = imobj.p2[2, 3].item()

            # filter gts
            # Removed imobj.imH constraint on height. This constraint was not there in original Garrick code and
            # keeping this constraint gives way lower validation 3DIOUs
            # igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h, imobj.imH)
            igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

            # accumulate boxes
            # In imobj, everything in 2D is multiplied by scale and therfore 2D coordinates are ground truths in 512
            gts_all = bbXYWH2Coords(np.array([gt.bbox_full for gt in gts]))
            gts_3d = np.array([gt.bbox_3d for gt in gts])

            if not ((rmvs == False) & (igns == False)).any():
                continue

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]
            gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            # accumulate labels
            box_lbls = np.array([gt.cls for gt in gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]
            box_lbls = np.array([clsName2Ind(self.lbls, cls) for cls in box_lbls])

            if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

                num_boxes_per_batch += gts_val.shape[0]

                #===============================================================
                # Get Targets (transformations of ground truths)
                #===============================================================
                # bbox regression
                # transforms   = np array [(H*num_anchors*W) x 23]  transformations wrt anchors
                # ( 4 2D ( 2 delta_scaled + 2 scaled_log) + 1 class_label + 16 3D (2 delta_scaled + 1 delta + 3 scaled_log + 1 delta)
                # + 2 axis label)
                # and not wrt predictions. later normalized
                # ols          = np array [(H*num_anchors*W) x M] M= number of gts_val
                # raw_gt       = np array [(H*num_anchors*W) x 21]  ( 4 X1Y1X2Y2 + 1 class_label + 16)
                #                               [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY, elevation, alpha_sin, alpha_cos,
                #                                axis_lbl, head_lbl]
                transforms, ols, raw_gt = compute_targets(gts_val, gts_ign, box_lbls, rois_numpy[img_index], self.fg_thresh,
                                                  self.ign_thresh, self.bg_thresh_lo, self.bg_thresh_hi,
                                                  self.best_thresh, anchors=self.anchors,  gts_3d=gts_3d,
                                                  tracker=rois_numpy[img_index, :, 4], rois_3d=rois_3d_numpy[img_index],
                                                          rois_3d_cen=rois_2d_cen_numpy[img_index])

                # Normalize transforms by mean and standard deviations
                # normalize 2d
                transforms[:, 0:4] -= self.bbox_means[:, 0:4]
                transforms[:, 0:4] /= self.bbox_stds[:, 0:4]

                if self.decomp_alpha:
                    # normalize 3d
                    transforms[:, 5:14] -= self.bbox_means[:, 4:13]
                    transforms[:, 5:14] /= self.bbox_stds[:, 4:13]
                else:
                    # normalize 3d
                    transforms[:, 5:12] -= self.bbox_means[:, 4:11]
                    transforms[:, 5:12] /= self.bbox_stds[:, 4:11]

                # ==============================================================
                # Classification of boxes into foreground, background and ignore
                # ==============================================================
                transforms_labels = transforms[:, 4]
                labels_fg  = transforms_labels > 0
                labels_bg  = transforms_labels < 0
                labels_ign = transforms_labels == 0

                fg_inds    = np.flatnonzero(labels_fg)
                bg_inds    = np.flatnonzero(labels_bg)
                ign_inds   = np.flatnonzero(labels_ign)

                transforms = torch.from_numpy(transforms).type(cls.dtype).cuda()
                raw_gt     = torch.from_numpy(raw_gt)    .type(cls.dtype).cuda()

                labels[img_index, fg_inds]  = transforms[fg_inds, 4].cpu().numpy()
                labels[img_index, ign_inds] = IGN_FLAG
                labels[img_index, bg_inds]  = 0

                #===============================================================
                # Assign Targets - transformations of ground truths) as well as raw ones
                #===============================================================
                bbox_x_tar[img_index, :] = torch.from_numpy(transforms[:, 0]).type(cls.dtype).cuda()
                bbox_y_tar[img_index, :] = torch.from_numpy(transforms[:, 1]).type(cls.dtype).cuda()
                bbox_w_tar[img_index, :] = torch.from_numpy(transforms[:, 2]).type(cls.dtype).cuda()
                bbox_h_tar[img_index, :] = torch.from_numpy(transforms[:, 3]).type(cls.dtype).cuda()

                bbox_x3d_tar[img_index, :]  = torch.from_numpy(transforms[:, 5]).type(cls.dtype).cuda()
                bbox_y3d_tar[img_index, :]  = torch.from_numpy(transforms[:, 6]).type(cls.dtype).cuda()
                bbox_z3d_tar[img_index, :]  = torch.from_numpy(transforms[:, 7]).type(cls.dtype).cuda()
                bbox_w3d_tar[img_index, :]  = torch.from_numpy(transforms[:, 8]).type(cls.dtype).cuda()
                bbox_h3d_tar[img_index, :]  = torch.from_numpy(transforms[:, 9]).type(cls.dtype).cuda()
                bbox_l3d_tar[img_index, :]  = torch.from_numpy(transforms[:, 10]).type(cls.dtype).cuda()
                bbox_rot3d_tar[img_index, :] = torch.from_numpy(transforms[:, 11]).type(cls.dtype).cuda()

                if self.decomp_alpha:
                    bbox_axis_tar[img_index, :] = torch.from_numpy(raw_gt[:, 19]).type(cls.dtype).cuda()
                    bbox_head_tar[img_index, :] = torch.from_numpy(raw_gt[:, 20]).type(cls.dtype).cuda()
                    bbox_rsin_tar[img_index, :] = torch.from_numpy(transforms[:, 12]).type(cls.dtype).cuda()
                    bbox_rcos_tar[img_index, :] = torch.from_numpy(transforms[:, 13]).type(cls.dtype).cuda()

                # raw_gt       = np array [(H*num_anchors*W) x 21]  ( 4 X1Y1X2Y2 + 1 class_label + 16)
                #                                 5   6    7       8    9    10   11     12    13    14    15
                #                               [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY, elevation, alpha_sin, alpha_cos,
                #                                axis_lbl, head_lbl]
                bbox_x3d_raw_tar [img_index, :] = raw_gt[:, 12]
                bbox_y3d_raw_tar [img_index, :] = raw_gt[:, 13]
                bbox_z3d_raw_tar [img_index, :] = raw_gt[:, 14]
                bbox_l3d_raw_tar [img_index, :] = raw_gt[:, 10]
                bbox_w3d_raw_tar [img_index, :] = raw_gt[:,  8]
                bbox_h3d_raw_tar [img_index, :] = raw_gt[:,  9]
                bbox_ry3d_raw_tar[img_index, :] = raw_gt[:, 15]

                if self.decomp_alpha:
                    bbox_alph_raw_tar[img_index, :] = raw_gt[:, 11]
                    bbox_alph_raw_tar_2[img_index, :] = bbox_alph_raw_tar[img_index, :].clone()
                else:
                    bbox_alph_raw_tar = bbox_rot3d_tar[img_index, :] * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]
                    bbox_alph_raw_tar = rois_3d[:, 8] + bbox_alph_raw_tar

                # ==============================================================
                # Compile 2D target deltas
                # ==============================================================
                deltas_2d_tar_img = torch.cat(
                    (bbox_x_tar[img_index].unsqueeze(1), bbox_y_tar[img_index].unsqueeze(1),
                     bbox_w_tar[img_index].unsqueeze(1), bbox_h_tar[img_index].unsqueeze(1)),
                    dim= 1)

                coords_2d_512_img_tar = bbox_transform_inv(rois[img_index], deltas_2d_tar_img, means= self.bbox_means[0, :],
                                                           stds= self.bbox_stds[0, :])  # torch.Size([ (H*num_anchors*W) x          5])

                # ==============================================================
                # Convert projected 3D_2D predictions to 3D
                # ==============================================================
                # dn = denormalized
                # re-scale all 2D in (512, 1760) resolution back to original size
                # We will be doing for all the boxes and then choose only those which are fgs
                bbox_x3d_2d_raw_img = bbox_x3d_dn[img_index]/imobj['scale_factor']
                bbox_y3d_2d_raw_img = bbox_y3d_dn[img_index]/imobj['scale_factor']
                bbox_z3d_2d_raw_img = bbox_z3d_dn[img_index]

                #       --           --
                #      | a   0   b  c |
                # p2 = | 0   d   e  f |
                #      | 0   0   1  h |
                #       __          __
                #
                # x3d_2d_raw * z3d_2d_raw = a * x3d_raw + b * z3d_raw + c
                # y3d_2d_raw * z3d_2d_raw = d * y3d_raw + e * z3d_raw + f
                #              z3d_2d_raw =                   z3d_raw + h
                # Substituting z2d in z3d_dn, we have
                # x3d_2d_raw * (z3d_raw + h) = a * x3d_raw + b * z3d_raw + c
                # y3d_2d_raw * (z3d_raw + h) = d * y3d_raw + e * z3d_raw + f
                #
                # Rearranging
                # x3d_raw = (x3d_2d_raw * (z3d_raw + h) - b * z3d_raw - c)/a
                # y3d_raw = (y3d_2d_raw * (z3d_raw + h) - e * z3d_raw - f)/b

                z3d_raw_img = bbox_z3d_2d_raw_img - p2_h
                x3d_raw_img = ((z3d_raw_img + p2_h) * bbox_x3d_2d_raw_img - p2_b * (z3d_raw_img) - p2_c) / p2_a
                y3d_raw_img = ((z3d_raw_img + p2_h) * bbox_y3d_2d_raw_img - p2_e * (z3d_raw_img) - p2_f) / p2_d

                bbox_x3d_raw[img_index, :] = x3d_raw_img
                bbox_y3d_raw[img_index, :] = y3d_raw_img
                bbox_z3d_raw[img_index, :] = z3d_raw_img

                if self.decomp_alpha:
                    axis_sin_mask = bbox_axis_tar[img_index, :] == 1  # axis label = 0 cosine, axis label =1  sin
                    head_pos_mask = bbox_head_tar[img_index, :] == 1  # add pi or not? --> 0 nothing, 1 add pi

                    if not self.torch_bool:
                        axis_sin_mask = torch.from_numpy(np.array(axis_sin_mask, dtype=np.uint8))
                        head_pos_mask = torch.from_numpy(np.array(head_pos_mask, dtype=np.uint8))

                    bbox_rot3d_raw[img_index] = bbox_rcos_dn[img_index, :]
                    bbox_rot3d_raw[img_index, axis_sin_mask] = bbox_rsin_dn[img_index, :][axis_sin_mask]
                    bbox_rot3d_raw[img_index, head_pos_mask] = bbox_rot3d_raw[img_index, head_pos_mask] + math.pi

                bbox_rot3d_raw_snapped_img = snap_to_pi(bbox_rot3d_raw[img_index].detach())
                bbox_ry3d_raw[img_index]   = convertAlpha2Rot(bbox_rot3d_raw_snapped_img.detach(), z3d= bbox_z3d_raw[img_index].detach(), x3d= bbox_x3d_raw[img_index].detach())

                abs_err_z    [img_index]   = torch.abs(bbox_z3d_raw[img_index].detach()    - bbox_z3d_raw_tar [img_index].detach())
                abs_err_rot3d[img_index]   = torch.abs(bbox_rot3d_raw_snapped_img.detach() - bbox_alph_raw_tar[img_index])

                coords_2d_512_img = coords_2d_512[img_index]
                # ----------------------------------------
                # box sampling
                # ----------------------------------------
                if self.box_samples == np.inf:
                    fg_num = len(fg_inds)
                    bg_num = len(bg_inds)
                else:
                    fg_num = min(round(rois[img_index].shape[0]*self.box_samples * self.fg_fraction), len(fg_inds))
                    bg_num = min(round(rois[img_index].shape[0]*self.box_samples - fg_num), len(bg_inds))

                if self.hard_negatives:
                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        scores = prob_detach[img_index, fg_inds, labels[img_index, fg_inds].astype(int)]
                        fg_score_ascend = (scores).argsort()
                        fg_inds = fg_inds[fg_score_ascend]
                        fg_inds = fg_inds[0:fg_num]

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[img_index, bg_inds, labels[img_index, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:
                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        fg_inds = np.random.choice(fg_inds, fg_num, replace=False)

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)

                labels_weight[img_index, bg_inds] = BG_ENC
                labels_weight[img_index, fg_inds] = FG_ENC
                bbox_weights[img_index, fg_inds] = 1

                # ----------------------------------------
                # compute IoU stats
                # ----------------------------------------
                if fg_num > 0:
                    fg_inds_tensor          = torch.from_numpy(fg_inds).long()
                    if not self.infer_2d_from_3d:
                        ious_2d[img_index, fg_inds_tensor]           = iou(coords_2d_512_img[fg_inds_tensor, :], coords_2d_512_img_tar[fg_inds_tensor, :], mode='list')
                        coords_2d_512_tar[img_index, fg_inds_tensor] = coords_2d_512_img_tar[fg_inds_tensor, :]

            else:
                #======================================================================
                # Handle background image
                #======================================================================
                bg_inds = np.arange(0, rois[img_index].shape[0])

                if self.box_samples == np.inf: bg_num = len(bg_inds)
                else: bg_num = min(round(self.box_samples * (1 - self.fg_fraction)), len(bg_inds))

                if self.hard_negatives:
                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[img_index, bg_inds, labels[img_index, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]
                else:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)

                labels[img_index, :] = 0
                labels_weight[img_index, bg_inds] = BG_ENC

            if self.predict_acceptance_prob:
                if self.boxes_for_acceptance_prob == "all":
                    acceptance_prob_valid_inds = torch.arange(h_times_w_times_anchors)
                elif self.boxes_for_acceptance_prob == "overlaps":
                    max_overlaps         = torch.max(torch.from_numpy(ols), dim=1)[0]
                    acceptance_prob_valid_inds = (max_overlaps > self.acceptance_prob_overlap_thres).nonzero().squeeze()
                else:
                    if fg_num > 0:
                        acceptance_prob_valid_inds = fg_inds_tensor
                    else:
                        acceptance_prob_valid_inds = torch.Tensor([])

                if acceptance_prob_valid_inds.shape[0] > 0 :
                    accept_prob_box_weight[img_index, acceptance_prob_valid_inds] = FG_ENC

                    # We need to detach the ry3d variable since convertAlphatoRot uses a while loop based on condition
                    # This manual assignment causes the failure of Pytorch autograd
                    corners_3d_b1 = get_corners_of_cuboid(x3d=bbox_x3d_raw[img_index, acceptance_prob_valid_inds],
                                                          y3d=bbox_y3d_raw[img_index, acceptance_prob_valid_inds],
                                                          z3d=bbox_z3d_raw[img_index, acceptance_prob_valid_inds],
                                                          w3d=bbox_w3d_raw[img_index, acceptance_prob_valid_inds],
                                                          h3d=bbox_h3d_raw[img_index, acceptance_prob_valid_inds],
                                                          l3d=bbox_l3d_raw[img_index, acceptance_prob_valid_inds],
                                                          ry3d=bbox_ry3d_raw[img_index, acceptance_prob_valid_inds].clone().detach())
                    corners_3d_b2 = get_corners_of_cuboid(x3d=bbox_x3d_raw_tar[img_index, acceptance_prob_valid_inds],
                                                          y3d=bbox_y3d_raw_tar[img_index, acceptance_prob_valid_inds],
                                                          z3d=bbox_z3d_raw_tar[img_index, acceptance_prob_valid_inds],
                                                          w3d=bbox_w3d_raw_tar[img_index, acceptance_prob_valid_inds],
                                                          h3d=bbox_h3d_raw_tar[img_index, acceptance_prob_valid_inds],
                                                          l3d=bbox_l3d_raw_tar[img_index, acceptance_prob_valid_inds],
                                                          ry3d=bbox_ry3d_raw_tar[img_index, acceptance_prob_valid_inds])

                    _, iou_3d_tar = iou3d_approximate(corners_3d_b1, corners_3d_b2)
                    bbox_acceptance_prob_tar[img_index, acceptance_prob_valid_inds] = iou_3d_tar
                    if self.acceptance_prob_mode  ==  "classify":
                        for classifier_ind in range(self.acceptance_prob_classify_num_class):
                            if self.acceptance_prob_classify_sort_K > 0 and self.acceptance_prob_classify_num_class == 1:
                                # Sort the iou_3d_tar in decreasing order
                                # Consider the highest K iou_3d_tar as one class and remaining as another class
                                sorted_index = torch.sort(-iou_3d_tar)[1]
                                bound_ind_1  = acceptance_prob_valid_inds[sorted_index[self.acceptance_prob_classify_sort_K:]]
                                bound_ind_2  = acceptance_prob_valid_inds[sorted_index[0:self.acceptance_prob_classify_sort_K]]
                            else:
                                # Assign indices to the ordinal classification
                                # A Simple Approach to Ordinal Classification, Kim et al
                                # https://link.springer.com/content/pdf/10.1007/3-540-44795-4_13.pdf
                                bound_ind_1 = acceptance_prob_valid_inds[iou_3d_tar <= self.bins_boundary[classifier_ind]]
                                bound_ind_2 = acceptance_prob_valid_inds[iou_3d_tar > self.bins_boundary[classifier_ind]]

                            if bound_ind_1.shape[0] > 0:
                                bbox_acceptance_prob_logits_tar[img_index, bound_ind_1, classifier_ind] = 0
                                bbox_acceptance_prob_logits_wts[img_index, bound_ind_1, classifier_ind] = 1

                            if bound_ind_2.shape[0] > 0:
                                bbox_acceptance_prob_logits_tar[img_index, bound_ind_2, classifier_ind] = 1
                                if bound_ind_1.shape[0] > 0:
                                    weight_scale = (bound_ind_1.shape[0]/bound_ind_2.shape[0])
                                else:
                                    weight_scale = 1.0
                                bbox_acceptance_prob_logits_wts[img_index, bound_ind_2, classifier_ind] = weight_scale

                            if self.debug:
                                print("No of boxes positives= {} negatives= {} total= {}".format(bound_ind_2.shape[0], bound_ind_1.shape[0], bound_ind_2.shape[0] + bound_ind_1.shape[0]))
                                iou_temp1 = iou_3d_tar[sorted_index[ self.acceptance_prob_classify_sort_K:]].clone().detach().cpu().numpy()
                                iou_temp2 = iou_3d_tar[sorted_index[0:self.acceptance_prob_classify_sort_K]].clone().detach().cpu().numpy()
                                # print(iou_temp1)
                                num_bins  = 50
                                my_bins = (np.arange(num_bins+1))/num_bins
                                plt.hist(iou_temp1, bins= my_bins, color= "limegreen", label= "negatives")
                                plt.hist(iou_temp2, bins= my_bins, color= "orange", label= "positives")
                                plt.grid(True)
                                plt.legend()
                                plt.show()
                                plt.close()

            if self.use_nms_in_loss:
                if bbox_acceptance_prob is not None:
                    scores_to_nms_img = bbox_acceptance_prob[img_index, :, 0]
                    if self.rank_with_class_confidence:
                        scores_to_nms_img = scores_to_nms_img * torch.max(prob[img_index, :, 1:], dim= 1)[0]
                else:
                    scores_to_nms_img = torch.max(prob[img_index, :, 1:], dim= 1)[0].clone()

                if fg_num > 0:
                    # Sort them in decreasing order
                    _, sorted_index = torch.sort(scores_to_nms_img[fg_inds_tensor], descending= True)
                    num_boxes_for_nms       = min(500, sorted_index.shape[0])

                    # Get the foreground and background for NMS. This will be different from usual foreground since
                    # because of the computational issues, we only have at max 500 boxes in the NMS
                    # print(fg_inds_tensor)
                    fg_index_for_nms        = fg_inds_tensor[sorted_index[:num_boxes_for_nms]]

                    if scores_to_nms_img.is_cuda:
                        fg_index_np = fg_index_for_nms.cpu()
                    else:
                        fg_index_np = fg_index_for_nms
                    fg_index_np     = fg_index_np.clone().numpy()
                    bg_index_for_nms= torch.from_numpy(np.setdiff1d(np.arange(scores_to_nms_img.shape[0]), fg_index_np))

                    corners_3d_b1 = get_corners_of_cuboid(x3d=bbox_x3d_raw[img_index, fg_index_for_nms],
                                                      y3d=bbox_y3d_raw[img_index, fg_index_for_nms],
                                                      z3d=bbox_z3d_raw[img_index, fg_index_for_nms],
                                                      w3d=bbox_w3d_raw[img_index, fg_index_for_nms],
                                                      h3d=bbox_h3d_raw[img_index, fg_index_for_nms],
                                                      l3d=bbox_l3d_raw[img_index, fg_index_for_nms],
                                                      ry3d=bbox_ry3d_raw[img_index, fg_index_for_nms].clone().detach())            # N x 3 x 8

                    # corners are in the shape N x 3 x 8
                    # bring them to  the shape 3 x N*8
                    corners_3d_b1_reshaped     = corners_3d_b1.transpose(1, 2).reshape((-1, 3)).transpose(0, 1)                    # 3 x (N*8)

                    # Project them to 2D
                    corners_3d_2d_b1_reshaped  = project_3d_points_in_4D_format(p2, corners_3d_b1_reshaped, pad_ones= True)        # 4 x (N*8)

                    corners_3d_2d_b1           = corners_3d_2d_b1_reshaped.transpose(0, 1).reshape((-1, 8, 4)).transpose(1, 2)     # N x 4 x 8

                    coords_2d_from_3d_b1_x1    = torch.min(corners_3d_2d_b1[:, 0], dim= 1)[0]
                    coords_2d_from_3d_b1_y1    = torch.min(corners_3d_2d_b1[:, 1], dim= 1)[0]
                    coords_2d_from_3d_b1_x2    = torch.max(corners_3d_2d_b1[:, 0], dim= 1)[0]
                    coords_2d_from_3d_b1_y2    = torch.max(corners_3d_2d_b1[:, 1], dim= 1)[0]
                    coords_2d_from_3d_b1       = torch.cat([coords_2d_from_3d_b1_x1.unsqueeze(1), coords_2d_from_3d_b1_y1.unsqueeze(1), coords_2d_from_3d_b1_x2.unsqueeze(1), coords_2d_from_3d_b1_y2.unsqueeze(1)], dim= 1)
                    coords_2d_from_3d_512_b1   = coords_2d_from_3d_b1 * imobj['scale_factor']

                    # Calculate iou2d overlaps
                    if self.diff_nms_boxes_2d == "normal":
                        ious_2d_for_nms_img    = iou(coords_2d_512_img[fg_index_for_nms, :], coords_2d_512_img[fg_index_for_nms, :], mode='combinations')
                    elif self.diff_nms_boxes_2d == "projected":
                        ious_2d_for_nms_img    = iou(coords_2d_from_3d_512_b1              , coords_2d_from_3d_512_b1              , mode='combinations')

                    if self.overlap_in_nms == "2d":
                        ious_for_nms_img    = ious_2d_for_nms_img
                    elif self.overlap_in_nms == "3d" or self.overlap_in_nms == "product":
                        # Calculate iou3d overlaps
                        _, ious_3d_for_nms_img  = iou3d_approximate(corners_3d_b1      , corners_3d_b1    , mode= "combinations", method= "generalized")
                        ious_3d_for_nms_img     = 0.5*(1+ ious_3d_for_nms_img)

                        if self.overlap_in_nms == "3d":
                            ious_for_nms_img    = ious_3d_for_nms_img
                        elif self.overlap_in_nms == "product":
                            ious_for_nms_img    = ious_2d_for_nms_img * ious_3d_for_nms_img

                    # ==============================================================
                    # Pass the boxes through our differentiable GrooMeD-NMS
                    # ==============================================================
                    _, _, scores_after_nms_img = differentiable_nms(scores_unsorted= scores_to_nms_img[fg_index_for_nms], iou_unsorted= ious_for_nms_img.clone().detach(), nms_threshold= self.nms_thres, pruning_method= self.diff_nms_pruning_method, temperature= self.diff_nms_temperature, valid_box_prob_threshold = self.diff_nms_valid_box_prob_threshold, return_sorted_prob= False, group_boxes= self.diff_nms_group_boxes, mask_group_boxes= self.diff_nms_mask_group_boxes, group_size= self.diff_nms_group_size)

                    scores_after_nms[img_index, fg_index_for_nms] = scores_after_nms_img
                    # scores_after_nms[img_index, bg_index_for_nms] = scores_to_nms_img[bg_index_for_nms]

                    # ==============================================================
                    # Get targets (best box) after NMS for each object by computing IOU3D
                    # ==============================================================
                    # No of boxes with +1 <= No of valid ground truth boxes
                    # All other boxes should have 0
                    gts_2d_img_tensor = torch.from_numpy(gts_val).type(cls.dtype).cuda()
                    gts_3d_img_tensor = torch.from_numpy(gts_3d).type(cls.dtype).cuda()
                    #          [0   1     2      3    4    5     6     7     8     9     10
                    # gts_3d = [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY, elevation, alpha_sin, alpha_cos, axis_lbl, head_lbl]
                    corners_3d_b2 = get_corners_of_cuboid(x3d= gts_3d_img_tensor[:, 7],
                                                      y3d= gts_3d_img_tensor[:, 8],
                                                      z3d= gts_3d_img_tensor[:, 9],
                                                      w3d= gts_3d_img_tensor[:, 3],
                                                      h3d= gts_3d_img_tensor[:, 4],
                                                      l3d= gts_3d_img_tensor[:, 5],
                                                      ry3d= gts_3d_img_tensor[:, 10])

                    _, iou_3d_with_gt_nms  = iou3d_approximate(corners_3d_b1           , corners_3d_b2    , mode= "combinations", method= "generalized")
                    ious_2d_with_gt_nms    = iou(coords_2d_512_img[fg_index_for_nms, :], gts_2d_img_tensor, mode= "combinations")
                    # Scores with ground truth is obtained by multiplying with iou3d and iou2d
                    # We multiply by iou2d since for some ground predictions there could be no predictions
                    # In such a case an erroneous box could be marked as 1
                    scores_with_gt_nms     = 0.5*(1 + iou_3d_with_gt_nms)*ious_2d_with_gt_nms
                    _, max_indices_nms     = torch.max(scores_with_gt_nms , dim= 0)
                    # Filter out the box with scores_with_gt < 0.1
                    max_indices_nms        = max_indices_nms[scores_with_gt_nms.gather(0, max_indices_nms.unsqueeze(0)).squeeze() > self.best_target_box_beta]
                    # Sometimes we might get 2D array, flatten it to 1D always
                    max_indices_nms        = max_indices_nms.flatten()
                    max_indices_nms_img    = fg_index_for_nms[max_indices_nms]
                    targets_after_nms[img_index, max_indices_nms_img] = 1

                    num_gt += gts_3d_img_tensor.shape[0]
                    fg_index_gt.append(max_indices_nms_img.clone().cpu().numpy())
                    # if max_indices_nms.shape[0] != torch.unique(max_indices_nms).shape[0]:
                    #     print(gts_3d_img_tensor.shape, iou_3d_with_gt_nms.shape, max_indices_nms_img.shape)
                    #     print(max_indices_nms_img, iou_3d_with_gt_nms.gather(0, max_indices_nms.unsqueeze(0)) )
                    # sys.exit(0)

                    if self.debug_nms_assignment:
                        img_path = imobjs[img_index].path
                        print(img_path)
                        img_temp = cv2.imread(img_path)[:,:,::-1]
                        scale_factor = 512 / img_temp.shape[0]

                        h = np.round(img_temp.shape[0] * scale_factor).astype(int)
                        w = np.round(img_temp.shape[1] * scale_factor).astype(int)

                        # resize
                        img_temp = cv2.resize(img_temp, (w, h))
                        plt.subplot(2, 1, 1)
                        plt.imshow(img_temp)
                        ax = plt.gca()
                        for f in range(fg_index_for_nms.shape[0]):
                            rect_x_left, rect_y_left, rect_width, rect_height = get_left_point_width_height(coords_2d_512_img[fg_index_for_nms[f]])
                            draw_rectangle(ax, rect_x_left= rect_x_left, rect_y_left= rect_y_left, rect_width= rect_width, rect_height= rect_height, edgecolor= 'orange')

                        for g in range(gts_val.shape[0]):
                            rect_x_left, rect_y_left, rect_width, rect_height = get_left_point_width_height(gts_val[g])
                            draw_rectangle(ax, rect_x_left= rect_x_left, rect_y_left= rect_y_left, rect_width= rect_width, rect_height= rect_height, edgecolor= 'limegreen')
                            plt.text(rect_x_left + rect_width/2.0, rect_y_left + rect_height/2.0, self.lbls[box_lbls[g]-1][:3], color= 'limegreen')

                        for f in range(max_indices_nms_img.shape[0]):
                            rect_x_left, rect_y_left, rect_width, rect_height = get_left_point_width_height(coords_2d_512_img[max_indices_nms_img[f]])
                            draw_rectangle(ax, rect_x_left= rect_x_left, rect_y_left= rect_y_left, rect_width= rect_width, rect_height= rect_height, edgecolor= 'dodgerblue')
                            text = "{:.2f}, {:.2f}".format(scores_with_gt_nms[max_indices_nms[f], f].item(), iou_3d_with_gt_nms[max_indices_nms[f], f].item())
                            plt.text(rect_x_left + rect_width, rect_y_left, text, color= 'dodgerblue', size= 8)

                        plt.subplot(2, 1, 2)
                        plt.imshow(img_temp)
                        ax = plt.gca()
                        for f in range(fg_index_for_nms.shape[0]):
                            rect_x_left, rect_y_left, rect_width, rect_height = get_left_point_width_height(coords_2d_from_3d_512_b1[f])
                            draw_rectangle(ax, rect_x_left= rect_x_left, rect_y_left= rect_y_left, rect_width= rect_width, rect_height= rect_height, edgecolor= 'purple')

                        for g in range(gts_val.shape[0]):
                            rect_x_left, rect_y_left, rect_width, rect_height = get_left_point_width_height(gts_val[g])
                            draw_rectangle(ax, rect_x_left= rect_x_left, rect_y_left= rect_y_left, rect_width= rect_width, rect_height= rect_height, edgecolor= 'limegreen')
                            plt.text(rect_x_left + rect_width/2.0, rect_y_left + rect_height/2.0, self.lbls[box_lbls[g]-1][:3], color= 'limegreen')

                        for f in range(max_indices_nms_img.shape[0]):
                            rect_x_left, rect_y_left, rect_width, rect_height = get_left_point_width_height(coords_2d_512_img[max_indices_nms_img[f]])
                            draw_rectangle(ax, rect_x_left= rect_x_left, rect_y_left= rect_y_left, rect_width= rect_width, rect_height= rect_height, edgecolor= 'dodgerblue')
                            text = "{:.2f}, {:.2f}".format(scores_with_gt_nms[max_indices_nms[f], f].item(), iou_3d_with_gt_nms[max_indices_nms[f], f].item())
                            plt.text(rect_x_left + rect_width, rect_y_left, text, color= 'dodgerblue', size= 8)

                        if gts_val.shape[0] != max_indices_nms_img.shape[0]:
                            plt.show()
                        plt.close()

            # grab label predictions (for weighing purposes)
            active = labels[img_index, :] != IGN_FLAG
            labels_scores[img_index, active] = prob_detach[img_index, active, labels[img_index, active].astype(int)]

        # ----------------------------------------
        # useful statistics
        # ----------------------------------------

        fg_inds_all = np.flatnonzero((labels > 0) & (labels != IGN_FLAG))
        bg_inds_all = np.flatnonzero((labels == 0) & (labels != IGN_FLAG))

        fg_inds_unravel = np.unravel_index(fg_inds_all, prob_detach.shape[0:2])
        bg_inds_unravel = np.unravel_index(bg_inds_all, prob_detach.shape[0:2])

        cls_pred = cls.argmax(dim=2).cpu().detach().numpy()

        if self.cls_2d_lambda and len(fg_inds_all) > 0:
            acc_fg = np.mean(cls_pred[fg_inds_unravel] == labels[fg_inds_unravel])
            stats.append({'name': 'fg', 'val': acc_fg, 'format': '{:0.2f}', 'group': 'acc'})

        if self.cls_2d_lambda and len(bg_inds_all) > 0:
            acc_bg = np.mean(cls_pred[bg_inds_unravel] == labels[bg_inds_unravel])
            if self.verbose: stats.append({'name': 'bg', 'val': acc_bg, 'format': '{:0.2f}', 'group': 'acc'})

        # ----------------------------------------
        # box weighting
        # ----------------------------------------

        fg_inds = np.flatnonzero(labels_weight == FG_ENC)
        bg_inds = np.flatnonzero(labels_weight == BG_ENC)
        active_inds = np.concatenate((fg_inds, bg_inds), axis=0)

        fg_num = len(fg_inds)
        bg_num = len(bg_inds)

        labels_weight[...] = 0.0
        box_samples = fg_num + bg_num

        fg_inds_unravel = np.unravel_index(fg_inds, labels_weight.shape)
        bg_inds_unravel = np.unravel_index(bg_inds, labels_weight.shape)
        active_inds_unravel = np.unravel_index(active_inds, labels_weight.shape)

        labels_weight[active_inds_unravel] = 1.0

        if self.fg_fraction is not None:

            if fg_num > 0:

                fg_weight = (self.fg_fraction /(1 - self.fg_fraction)) * (bg_num / fg_num)
                labels_weight[fg_inds_unravel] = fg_weight
                labels_weight[bg_inds_unravel] = 1.0

            else:
                labels_weight[bg_inds_unravel] = 1.0

        # different method of doing hard negative mining
        # use the scores to normalize the importance of each sample
        # hence, encourages the network to get all "correct" rather than
        # becoming more correct at a decision it is already good at
        # this method is equivalent to the focal loss with additional mean scaling
        if self.focal_loss:

            weights_sum = 0

            # re-weight bg
            if bg_num > 0:
                bg_scores = labels_scores[bg_inds_unravel]
                bg_weights = (1 - bg_scores) ** self.focal_loss
                weights_sum += np.sum(bg_weights)
                labels_weight[bg_inds_unravel] *= bg_weights

            # re-weight fg
            if fg_num > 0:
                fg_scores = labels_scores[fg_inds_unravel]
                fg_weights = (1 - fg_scores) ** self.focal_loss
                weights_sum += np.sum(fg_weights)
                labels_weight[fg_inds_unravel] *= fg_weights


        # ----------------------------------------
        # classification loss
        # ----------------------------------------
        labels = torch.tensor(labels, requires_grad=False)
        labels = labels.view(-1).type(torch.cuda.LongTensor)

        labels_weight = torch.tensor(labels_weight, requires_grad=False)
        labels_weight = labels_weight.view(-1).type(cls.dtype)

        cls_reshaped  = cls.view(-1, cls.shape[2])
        prob_reshaped = prob.view(-1, prob.shape[2])

        if self.cls_2d_lambda:

            # cls loss
            active = labels_weight > 0

            if np.any(active.cpu().numpy()):

                loss_cls = F.cross_entropy(cls_reshaped[active, :], labels[active], reduction='none', ignore_index=IGN_FLAG)
                # Remember that nll_loss takes log probabilities
                # https://pytorch.org/docs/0.4.0/nn.html?highlight=nll_loss#torch.nn.NLLLoss
                # We comment out the line below because of the following reasons
                # (a) there is clipping of the loss which might be different
                # (b) F.cross_entropy is more stable than F.nll_loss
                # loss_cls = F.nll_loss(torch.log(prob_reshaped[active, :]), labels[active], reduction='none', ignore_index=IGN_FLAG)
                loss_cls = (loss_cls * labels_weight[active])

                # simple gradient clipping
                loss_cls = loss_cls.clamp(min=0, max=2000)

                # take mean and scale lambda
                loss_cls = loss_cls.mean()
                loss_cls *= self.cls_2d_lambda

                loss += loss_cls

                stats.append({'name': 'cls', 'val': loss_cls.detach(), 'format': '{:0.4f}', 'group': 'loss'})

        if self.predict_acceptance_prob or self.use_nms_in_loss:
            # Get the active boxes
            accept_prob_active = None
            if self.boxes_for_acceptance_prob == "all":
                accept_prob_active = torch.arange(h_times_w_times_anchors).long().cuda()
            elif self.boxes_for_acceptance_prob == "overlaps":
                accept_prob_valid   = np.flatnonzero(accept_prob_box_weight == FG_ENC)
                accept_prob_invalid = np.flatnonzero(accept_prob_box_weight == BG_ENC)
                accept_prob_active  = np.concatenate((accept_prob_valid, accept_prob_invalid), axis=0)
                accept_prob_active  = torch.from_numpy(accept_prob_active).long().cuda()
            elif self.boxes_for_acceptance_prob == "foregrounds" or self.use_nms_in_loss:
                if np.sum(bbox_weights) > 0:
                    bbox_weights_fg = torch.tensor(bbox_weights, requires_grad=False).type(cls.dtype).view(-1)
                    accept_prob_active  = bbox_weights_fg > 0

            if bbox_acceptance_prob is not None:
                # Flatten the tensors
                bbox_acceptance_prob     = bbox_acceptance_prob[:, :, 0].view(-1)
                bbox_acceptance_prob_tar = bbox_acceptance_prob_tar.view(-1)

            if bbox_acceptance_prob_cls is not None:
                bbox_acceptance_prob_cls        = bbox_acceptance_prob_cls.view(-1)
            if bbox_acceptance_prob_logits_tar is not None:
                bbox_acceptance_prob_logits_tar = bbox_acceptance_prob_logits_tar.view(-1)
            if bbox_acceptance_prob_logits_wts is not None:
                bbox_acceptance_prob_logits_wts = bbox_acceptance_prob_logits_wts.view(-1)

        # ---------------------------------------
        # bbox acceptance prob regression loss
        # ----------------------------------------
        if self.predict_acceptance_prob and self.acceptance_prob_lambda:
            # print("Going inside this loop...")

            # Get which mode to follow- classify/rank/regress
            if accept_prob_active is None:
                bbox_prob_loss_unweighted = torch.zeros((1,)).float().cuda()

            elif self.acceptance_prob_mode  ==  "classify":
                loss_bbox_prob = F.binary_cross_entropy(bbox_acceptance_prob_cls[accept_prob_active], bbox_acceptance_prob_logits_tar[accept_prob_active], reduction='none')
                loss_bbox_prob = bbox_acceptance_prob_logits_wts[accept_prob_active]*loss_bbox_prob
                bbox_prob_loss_unweighted = loss_bbox_prob[torch.isfinite(loss_bbox_prob)].mean()

            elif self.acceptance_prob_mode  ==  "rank":
                if self.rank_with_class_confidence:
                    bbox_acceptance_prob_new = bbox_acceptance_prob * torch.max(prob_reshaped[:, 1:], dim= 1)[0]
                else:
                    bbox_acceptance_prob_new = bbox_acceptance_prob

                # APloss considers 0 as background, 1 as foreground and -1 as invalid
                bbox_acceptance_prob_rank_tar = -torch.ones(bbox_acceptance_prob_tar[accept_prob_active].shape)
                bbox_acceptance_prob_rank_tar[bbox_acceptance_prob_tar[accept_prob_active] >= 0.6] = 1
                bbox_acceptance_prob_rank_tar[bbox_acceptance_prob_tar[accept_prob_active] <  0.6] = 0

                bbox_prob_loss_unweighted = apLoss(bbox_acceptance_prob_new[accept_prob_active], bbox_acceptance_prob_rank_tar.detach().clone()).squeeze()

            elif self.acceptance_prob_mode  ==  "regress" or self.acceptance_prob_mode  ==  "likelihood":
                if self.boxes_for_acceptance_prob == "all":
                    loss_bbox_prob_regress  = F.l1_loss(bbox_acceptance_prob[accept_prob_active]    , bbox_acceptance_prob_tar[accept_prob_active], reduction='none')
                    loss_bbox_prob_regress  = loss_bbox_prob_regress * labels_weight[accept_prob_active]
                elif self.boxes_for_acceptance_prob == "overlaps":
                    loss_bbox_prob_regress  = F.l1_loss(bbox_acceptance_prob[accept_prob_active],  bbox_acceptance_prob_tar[accept_prob_active].detach().clone(), reduction='none')
                elif self.boxes_for_acceptance_prob == "foregrounds":
                    if self.acceptance_prob_mode == "likelihood":
                        loss_bbox_prob_regress = -torch.log(bbox_acceptance_prob[accept_prob_active])
                    else:
                        loss_bbox_prob_regress = F.l1_loss(bbox_acceptance_prob[accept_prob_active],  bbox_acceptance_prob_tar[accept_prob_active].detach().clone(), reduction='none')

                if self.weigh_acceptance_prob_regularization:
                    loss_bbox_prob_regress = loss_bbox_prob_regress * bbox_acceptance_prob_tar[accept_prob_active].detach().clone()
                loss_bbox_prob = loss_bbox_prob_regress
                bbox_prob_loss_unweighted = loss_bbox_prob[torch.isfinite(loss_bbox_prob)].mean()

            bbox_prob_loss = self.acceptance_prob_lambda * bbox_prob_loss_unweighted
            loss += bbox_prob_loss

            if self.acceptance_prob_mode  ==  "classify":
                loss_display_name = "bbox_prob_class"
            elif self.acceptance_prob_mode  ==  "rank":
                loss_display_name = "bbox_prob_rank"
            elif self.acceptance_prob_mode  ==  "regress":
                loss_display_name = "bbox_prob_reg"
            elif self.acceptance_prob_mode  == "likelihood":
                loss_display_name = "bbox_prob_nll"
            stats.append({'name': loss_display_name, 'val': bbox_prob_loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})

        # ---------------------------------------
        # After NMS loss
        # ----------------------------------------
        if self.use_nms_in_loss and self.after_nms_lambda:
            if self.after_nms_loss_mode  ==  "rank" and not self.rank_boxes_of_all_images_at_once:
                # rank boxes of one image at a time and therefore, do not flatten
                pass
            else:
                scores_after_nms          = scores_after_nms.view(-1)
                targets_after_nms         = targets_after_nms.view(-1)

            # Get which mode to follow- classify/rank/regress
            if accept_prob_active is None:
                bbox_nms_loss_unweighted = torch.zeros((1,)).float().cuda()

            elif self.after_nms_loss_mode  ==  "classify":
                weights_nms = torch.ones(targets_after_nms[accept_prob_active].shape).type(cls.dtype).cuda()
                ind_positives_nms = targets_after_nms[accept_prob_active] == 1
                ind_negatives_nms = targets_after_nms[accept_prob_active] == 0
                num_positives = torch.sum(ind_positives_nms).item()
                num_negatives = torch.sum(ind_negatives_nms).item()
                if num_positives > 0 and num_negatives > 0:
                    # weights_nms[ind_positives_nms] = np.power(1/num_positives, 0.25)
                    weights_nms[ind_negatives_nms] = np.power(num_positives/num_negatives, 0.25)
                loss_bbox_nms = F.binary_cross_entropy(scores_after_nms[accept_prob_active], targets_after_nms[accept_prob_active].detach(), reduction='none')
                loss_bbox_nms = weights_nms.detach() * loss_bbox_nms

                bbox_nms_loss_unweighted = loss_bbox_nms[torch.isfinite(loss_bbox_nms)].mean()

            elif self.after_nms_loss_mode  ==  "rank":
                if self.rank_boxes_of_all_images_at_once:
                    bbox_nms_loss_unweighted = apLoss(scores_after_nms[accept_prob_active], targets_after_nms[accept_prob_active].detach()).squeeze()
                else:
                    bbox_nms_loss_unweighted = torch.zeros((1,)).float().cuda()
                    img_cnt = 0
                    for img_index in range(batch_size):
                        if np.sum(bbox_weights[img_index]) > 0:
                            img_cnt += 1
                            bbox_weights_fg_img    = torch.tensor(bbox_weights[img_index], requires_grad=False).type(cls.dtype)
                            accept_prob_active_img = bbox_weights_fg_img > 0
                            bbox_nms_loss_unweighted += apLoss(scores_after_nms[img_index, accept_prob_active_img], targets_after_nms[img_index, accept_prob_active_img].detach()).squeeze()
                    if img_cnt > 0:
                        bbox_nms_loss_unweighted /= img_cnt
                    bbox_nms_loss_unweighted = bbox_nms_loss_unweighted.squeeze()

            elif self.after_nms_loss_mode  == "regress":
                loss_bbox_nms_regress = F.l1_loss(scores_after_nms[accept_prob_active],  targets_after_nms[accept_prob_active].detach().clone(), reduction='none')
                bbox_nms_loss_unweighted = loss_bbox_nms_regress[torch.isfinite(loss_bbox_nms_regress)].mean()

            bbox_after_nms_loss = self.after_nms_lambda * bbox_nms_loss_unweighted
            loss += bbox_after_nms_loss

            if self.after_nms_loss_mode  ==  "classify":
                loss_display_name = "bbox_after_nms_class"
            elif self.after_nms_loss_mode  ==  "rank":
                loss_display_name = "bbox_after_nms_rank"
            elif self.after_nms_loss_mode  ==  "regress":
                loss_display_name = "bbox_after_nms_reg"
            elif self.after_nms_loss_mode  == "likelihood":
                loss_display_name = "bbox_after_nms_nll"
            stats.append({'name': loss_display_name, 'val': bbox_after_nms_loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})

        # ----------------------------------------
        # bbox regression loss
        # ----------------------------------------

        if np.sum(bbox_weights) > 0:

            bbox_weights = torch.tensor(bbox_weights, requires_grad=False).type(cls.dtype).view(-1)

            active = bbox_weights > 0

            if self.has_un:
                bbox_acceptance_prob = bbox_acceptance_prob[:, :, 0].view(-1)

            if self.bbox_2d_lambda:
                # bbox loss 2d
                # Bring everything in terms of no of boxes
                bbox_x_tar = bbox_x_tar.view(-1)
                bbox_y_tar = bbox_y_tar.view(-1)
                bbox_w_tar = bbox_w_tar.view(-1)
                bbox_h_tar = bbox_h_tar.view(-1)

                bbox_x = bbox_x[:, :].unsqueeze(2).view(-1)
                bbox_y = bbox_y[:, :].unsqueeze(2).view(-1)
                bbox_w = bbox_w[:, :].unsqueeze(2).view(-1)
                bbox_h = bbox_h[:, :].unsqueeze(2).view(-1)

                # Pass through the loss
                loss_bbox_x = F.smooth_l1_loss(bbox_x[active], bbox_x_tar[active], reduction='none')
                loss_bbox_y = F.smooth_l1_loss(bbox_y[active], bbox_y_tar[active], reduction='none')
                loss_bbox_w = F.smooth_l1_loss(bbox_w[active], bbox_w_tar[active], reduction='none')
                loss_bbox_h = F.smooth_l1_loss(bbox_h[active], bbox_h_tar[active], reduction='none')

                loss_bbox_x = (loss_bbox_x * bbox_weights[active]).mean()
                loss_bbox_y = (loss_bbox_y * bbox_weights[active]).mean()
                loss_bbox_w = (loss_bbox_w * bbox_weights[active]).mean()
                loss_bbox_h = (loss_bbox_h * bbox_weights[active]).mean()

                bbox_2d_loss = (loss_bbox_x + loss_bbox_y + loss_bbox_w + loss_bbox_h)
                bbox_2d_loss *= self.bbox_2d_lambda

                loss += bbox_2d_loss
                if self.verbose: stats.append({'name': 'bbox_2d', 'val': bbox_2d_loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})

            # bbox center distance
            bbox_x3d_raw_tar = bbox_x3d_raw_tar.view(-1)
            bbox_y3d_raw_tar = bbox_y3d_raw_tar.view(-1)
            bbox_z3d_raw_tar = bbox_z3d_raw_tar.view(-1)

            bbox_x3d_raw = bbox_x3d_raw[:, :].view(-1)
            bbox_y3d_raw = bbox_y3d_raw[:, :].view(-1)
            bbox_z3d_raw = bbox_z3d_raw[:, :].view(-1)

            cen_dist = torch.sqrt(((bbox_x3d_raw[active] - bbox_x3d_raw_tar[active])**2)
                                  + ((bbox_y3d_raw[active] - bbox_y3d_raw_tar[active])**2)
                                  + ((bbox_z3d_raw[active] - bbox_z3d_raw_tar[active])**2))

            cen_match = (cen_dist <= 0.20).type(cls.dtype)

            if self.verbose: stats.append({'name': 'cen', 'val': cen_dist.mean().detach(), 'format': '{:0.2f}', 'group': 'misc'})
            stats.append({'name': 'cen_dist_lt_0.2', 'val': cen_match.mean().detach(), 'format': '{:0.3f}', 'group': 'misc'})

            if self.bbox_3d_lambda:

                # bbox loss 3d
                # Bring everything in terms of no of boxes
                bbox_x3d_tar = bbox_x3d_tar.view(-1)
                bbox_y3d_tar = bbox_y3d_tar.view(-1)
                bbox_z3d_tar = bbox_z3d_tar.view(-1)
                bbox_w3d_tar = bbox_w3d_tar.view(-1)
                bbox_h3d_tar = bbox_h3d_tar.view(-1)
                bbox_l3d_tar = bbox_l3d_tar.view(-1)
                bbox_rot3d_tar = bbox_rot3d_tar.view(-1)

                bbox_x3d = bbox_x3d[:, :].view(-1)
                bbox_y3d = bbox_y3d[:, :].view(-1)
                bbox_z3d = bbox_z3d[:, :].view(-1)
                bbox_w3d = bbox_w3d[:, :].view(-1)
                bbox_h3d = bbox_h3d[:, :].view(-1)
                bbox_l3d = bbox_l3d[:, :].view(-1)
                bbox_rot3d = bbox_rot3d[:, :].view(-1)

                # Pass through the loss
                loss_bbox_x3d = F.smooth_l1_loss(bbox_x3d[active], bbox_x3d_tar[active], reduction='none')
                loss_bbox_y3d = F.smooth_l1_loss(bbox_y3d[active], bbox_y3d_tar[active], reduction='none')
                loss_bbox_z3d = F.smooth_l1_loss(bbox_z3d[active], bbox_z3d_tar[active], reduction='none')
                loss_bbox_w3d = F.smooth_l1_loss(bbox_w3d[active], bbox_w3d_tar[active], reduction='none')
                loss_bbox_h3d = F.smooth_l1_loss(bbox_h3d[active], bbox_h3d_tar[active], reduction='none')
                loss_bbox_l3d = F.smooth_l1_loss(bbox_l3d[active], bbox_l3d_tar[active], reduction='none')
                loss_bbox_ry3d = F.smooth_l1_loss(bbox_rot3d[active], bbox_rot3d_tar[active], reduction='none')

                if self.orientation_bins > 0:

                    bbox_alph_raw_tar_2 = bbox_alph_raw_tar_2[:, :].view(-1)
                    bbox_alpha = bbox_alpha[:, :].view(-1)
                    bbox_alpha_bins = bbox_alpha_bins.view(-1, self.orientation_bins)

                    bins = torch.arange(-math.pi, math.pi, 2 * math.pi / self.orientation_bins)
                    difs = (bbox_alph_raw_tar_2[active].unsqueeze(0) - bins.unsqueeze(1))
                    bin_label = difs.abs().argmin(dim=0)
                    bin_dist = bbox_alph_raw_tar_2[active] - bins[bin_label]

                    bin_loss = F.cross_entropy(bbox_alpha_bins[active, :], bin_label, reduction='none') * self.bbox_axis_head_lambda
                    loss_bbox_ry3d = F.smooth_l1_loss(bbox_alpha[active], bin_dist, reduction='none')

                    alpha_pred = bins[bbox_alpha_bins[active].clone().detach().argmax(dim=1)] + bbox_alpha[
                        active].clone().detach()

                    abs_err_rot3d[active] = (bbox_alph_raw_tar_2[active] - alpha_pred).abs()

                    loss_bbox_ry3d = (loss_bbox_ry3d + bin_loss)

                    if self.verbose: stats.append({'name': 'a_bin', 'val': (bbox_alpha_bins[active].detach().argmax(dim=1) == bin_label).type(
                        torch.cuda.FloatTensor).mean(), 'format': '{:0.2f}', 'group': 'acc'})

                    a = 1

                elif self.decomp_alpha:

                    bbox_rsin_tar = bbox_rsin_tar.view(-1)
                    bbox_rcos_tar = bbox_rcos_tar.view(-1)
                    bbox_axis_tar = bbox_axis_tar.view(-1)
                    bbox_head_tar = bbox_head_tar.view(-1)

                    bbox_axis_sin_mask = bbox_axis_tar[active] == 1
                    bbox_head_pos_mask = bbox_head_tar[active] == 1

                    bbox_rsin = bbox_rsin[:, :].view(-1)
                    bbox_rcos = bbox_rcos[:, :].view(-1)
                    bbox_axis = bbox_axis[:, :].view(-1)
                    bbox_head = bbox_head[:, :].view(-1)

                    loss_bbox_rsin = F.smooth_l1_loss(bbox_rsin[active], bbox_rsin_tar[active], reduction='none')
                    loss_bbox_rcos = F.smooth_l1_loss(bbox_rcos[active], bbox_rcos_tar[active], reduction='none')
                    loss_axis = F.binary_cross_entropy(bbox_axis[active], bbox_axis_tar[active], reduction='none')
                    loss_head = F.binary_cross_entropy(bbox_head[active], bbox_head_tar[active], reduction='none')

                    loss_bbox_ry3d = loss_bbox_rcos
                    loss_bbox_ry3d[bbox_axis_sin_mask] = loss_bbox_rsin[bbox_axis_sin_mask]

                    # compute axis accuracy
                    fg_points = bbox_axis[active].detach().cpu().numpy()
                    fg_labels = bbox_axis_tar[active].detach().cpu().numpy()
                    if self.verbose: stats.append({'name': 'axis', 'val': ((fg_points >= 0.5) == fg_labels).mean(), 'format': '{:0.2f}', 'group': 'acc'})

                    # compute axis accuracy
                    fg_points = bbox_head[active].detach().cpu().numpy()
                    fg_labels = bbox_head_tar[active].detach().cpu().numpy()
                    if self.verbose: stats.append({'name': 'head', 'val': ((fg_points >= 0.5) == fg_labels).mean(), 'format': '{:0.2f}', 'group': 'acc'})

                loss_bbox_x3d  = (loss_bbox_x3d * bbox_weights[active])
                loss_bbox_y3d  = (loss_bbox_y3d * bbox_weights[active])
                loss_bbox_z3d  = (loss_bbox_z3d * bbox_weights[active])
                loss_bbox_w3d  = (loss_bbox_w3d * bbox_weights[active])
                loss_bbox_h3d  = (loss_bbox_h3d * bbox_weights[active])
                loss_bbox_l3d  = (loss_bbox_l3d * bbox_weights[active])
                loss_bbox_ry3d = (loss_bbox_ry3d * bbox_weights[active])

                if self.decomp_alpha and self.orientation_bins <= 0:
                    loss_axis = (loss_axis * bbox_weights[active])
                    loss_head = (loss_head * bbox_weights[active])

                if self.weigh_3D_regression_loss_by_gt_iou3d:
                    loss_bbox_x3d  = loss_bbox_x3d  * bbox_acceptance_prob_tar[active].detach().clone()
                    loss_bbox_y3d  = loss_bbox_y3d  * bbox_acceptance_prob_tar[active].detach().clone()
                    loss_bbox_z3d  = loss_bbox_z3d  * bbox_acceptance_prob_tar[active].detach().clone()
                    loss_bbox_w3d  = loss_bbox_w3d  * bbox_acceptance_prob_tar[active].detach().clone()
                    loss_bbox_h3d  = loss_bbox_h3d  * bbox_acceptance_prob_tar[active].detach().clone()
                    loss_bbox_l3d  = loss_bbox_l3d  * bbox_acceptance_prob_tar[active].detach().clone()
                    loss_bbox_ry3d = loss_bbox_ry3d * bbox_acceptance_prob_tar[active].detach().clone()
                    if self.decomp_alpha:
                        loss_axis = (loss_axis * bbox_acceptance_prob_tar[active].detach().clone())
                        loss_head = (loss_head * bbox_acceptance_prob_tar[active].detach().clone())

                if self.bbox_un_dynamic:
                    loss_bbox_3d_init = float((loss_bbox_w3d[torch.isfinite(loss_bbox_w3d)].mean()
                                        + loss_bbox_h3d[torch.isfinite(loss_bbox_h3d)].mean()
                                        + loss_bbox_l3d[torch.isfinite(loss_bbox_l3d)].mean()
                                        + loss_bbox_ry3d[torch.isfinite(loss_bbox_ry3d)].mean()
                                        + loss_bbox_x3d[torch.isfinite(loss_bbox_x3d)].mean()
                                        + loss_bbox_y3d[torch.isfinite(loss_bbox_y3d)].mean()
                                        + loss_bbox_z3d[torch.isfinite(loss_bbox_z3d)].mean()).item())*self.bbox_3d_lambda

                    if self.decomp_alpha:
                        loss_bbox_3d_init += float((loss_axis[torch.isfinite(loss_axis)].mean()
                                                    + loss_head[torch.isfinite(loss_head)].mean()).item())*self.bbox_axis_head_lambda

                    if self.n_frames == 0:
                        self.bbox_un_lambda = loss_bbox_3d_init
                        self.n_frames += 1
                    else:
                        self.n_frames = min(100, self.n_frames + 1)
                        self.bbox_un_lambda = loss_bbox_3d_init/self.n_frames + self.bbox_un_lambda*(self.n_frames - 1)/self.n_frames
                        #self.bbox_un_lambda = loss_bbox_3d_init*(1 - 0.90) + self.bbox_un_lambda*0.90

                if self.use_acceptance_prob_in_regression_loss or (self.bbox_un_dynamic and self.bbox_un_lambda > 0):
                    loss_bbox_x3d  = loss_bbox_x3d  * bbox_acceptance_prob[active]
                    loss_bbox_y3d  = loss_bbox_y3d  * bbox_acceptance_prob[active]
                    loss_bbox_z3d  = loss_bbox_z3d  * bbox_acceptance_prob[active]
                    loss_bbox_w3d  = loss_bbox_w3d  * bbox_acceptance_prob[active]
                    loss_bbox_h3d  = loss_bbox_h3d  * bbox_acceptance_prob[active]
                    loss_bbox_l3d  = loss_bbox_l3d  * bbox_acceptance_prob[active]
                    loss_bbox_ry3d = loss_bbox_ry3d * bbox_acceptance_prob[active]
                    if self.decomp_alpha:
                        loss_axis = (loss_axis * bbox_acceptance_prob[active])
                        loss_head = (loss_head * bbox_acceptance_prob[active])
                if self.predict_acceptance_prob or self.use_acceptance_prob_in_regression_loss or (self.bbox_un_dynamic and self.bbox_un_lambda > 0):
                    stats.append({'name': 'conf', 'val': bbox_acceptance_prob[active].mean().detach(), 'format': '{:0.2f}', 'group': 'misc'})

                bbox_3d_loss = (loss_bbox_x3d[torch.isfinite(loss_bbox_x3d)].mean()
                                + loss_bbox_y3d[torch.isfinite(loss_bbox_y3d)].mean()
                                + loss_bbox_z3d[torch.isfinite(loss_bbox_z3d)].mean()
                                + loss_bbox_w3d[torch.isfinite(loss_bbox_w3d)].mean()
                                + loss_bbox_h3d[torch.isfinite(loss_bbox_h3d)].mean()
                                + loss_bbox_l3d[torch.isfinite(loss_bbox_l3d)].mean()
                                + loss_bbox_ry3d[torch.isfinite(loss_bbox_ry3d)].mean())

                if self.decomp_alpha and self.orientation_bins <= 0:
                    bbox_3d_loss += (loss_axis.mean() + loss_head.mean())*self.bbox_axis_head_lambda

                bbox_3d_loss *= self.bbox_3d_lambda

                loss += bbox_3d_loss

                stats.append({'name': 'bbox_3d', 'val': bbox_3d_loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})

            if self.bbox_un_lambda > 0:

                loss_bbox_un = (1 - bbox_acceptance_prob[active]).mean()
                loss_bbox_un *= self.bbox_un_lambda

                loss += loss_bbox_un

                stats.append({'name': 'un', 'val': loss_bbox_un.detach(), 'format': '{:0.4f}', 'group': 'loss'})

            abs_err_z = abs_err_z.view(-1)
            stats.append({'name': 'z', 'val': abs_err_z[active].detach().mean(), 'format': '{:0.2f}', 'group': 'misc'})

            abs_err_rot3d = abs_err_rot3d.view(-1)
            stats.append({'name': 'rot', 'val': abs_err_rot3d[active].detach().mean(), 'format': '{:0.2f}', 'group': 'misc'})

            if not self.infer_2d_from_3d:
                ious_2d = ious_2d.view(-1)
                stats.append({'name': 'iou_2d', 'val': ious_2d[active & torch.isfinite(ious_2d)].detach().mean(), 'format': '{:0.2f}', 'group': 'acc'})

            # use a 2d IoU based log loss
            if self.iou_2d_lambda and (ious_2d[active] != 0).any() and not self.infer_2d_from_3d:
                iou_2d_loss = -torch.log(ious_2d[active])
                iou_2d_loss = (iou_2d_loss * bbox_weights[active])
                # if self.use_acceptance_prob_in_regression_loss:
                #     iou_2d_loss = (iou_2d_loss * bbox_acceptance_prob[active])
                iou_2d_loss = iou_2d_loss[torch.isfinite(iou_2d_loss)].mean()

                iou_2d_loss *= self.iou_2d_lambda
                loss += iou_2d_loss

                if self.verbose: stats.append({'name': 'iou_2d_los', 'val': iou_2d_loss.detach(), 'format': '{:0.4f}', 'group': 'loss'})

            stats.append({'name': 'total', 'val': loss.clone().detach(), 'format': '{:0.4f}', 'group': 'loss'})

        return loss, stats
