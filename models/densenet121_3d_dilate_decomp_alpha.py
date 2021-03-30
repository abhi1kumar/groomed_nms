import torch.nn as nn
from torchvision import models
from lib.rpn_util import *
import torch


def dilate_layer(layer, val):

    layer.dilation = val
    layer.padding = val


class RPN(nn.Module):


    def __init__(self, phase, base, conf):
        super(RPN, self).__init__()

        self.base = base

        del self.base.transition3.pool

        # dilate
        dilate_layer(self.base.denseblock4.denselayer1.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer2.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer3.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer4.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer5.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer6.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer7.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer8.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer9.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer10.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer11.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer12.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer13.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer14.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer15.conv2, 2)
        dilate_layer(self.base.denseblock4.denselayer16.conv2, 2)

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]

        self.prop_feats = nn.Sequential(
            nn.Conv2d(self.base[-1].num_features, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # outputs
        # classification
        self.cls = nn.Conv2d(self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1)

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.bbox_alpha = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_axis = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_head = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # The input to softmax will be
        # batch x num_classes x (H*num_anchors) x W
        # Take softmax along dimension 1
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.feat_stride = conf.feat_stride
        self.feat_size = [0, 0]  # Initialize this variable with zeros
        self.anchors = conf.anchors

        self.predict_acceptance_prob        = False if not('predict_acceptance_prob' in conf) else conf.predict_acceptance_prob
        self.acceptance_prob_mode           = "regress" if not ('acceptance_prob_mode' in conf) else conf.acceptance_prob_mode
        self.acceptance_prob_classify_type  = "classify" if not ('acceptance_prob_classify_type' in conf) else conf.acceptance_prob_classify_type
        self.acceptance_prob_classify_bins  = 2 if not('acceptance_prob_classify_bins' in conf) else conf.acceptance_prob_classify_bins
        self.acceptance_prob_classify_num_class = self.acceptance_prob_classify_bins - 1
        self.acceptance_prob_num_layers     = 1 if not ('acceptance_prob_num_layers' in conf) else conf.acceptance_prob_num_layers
        self.acceptance_prob_num_channels   = 128

        self.use_nms_in_loss                = False if not('use_nms_in_loss' in conf) else conf.use_nms_in_loss
        if self.use_nms_in_loss:
            self.acceptance_prob_mode = "regress"
        if self.predict_acceptance_prob:
            output_channels = self.num_anchors

            if self.acceptance_prob_mode == "classify":
                if self.acceptance_prob_classify_type == "classify":
                    logging.info("Classifying acceptance prob...")
                    output_channels = output_channels * self.acceptance_prob_classify_num_class
                elif self.acceptance_prob_classify_type == "regress_then_classify":
                    logging.info("Regressing then classifying acceptance prob...")
                    self.acceptance_prob_classifier = nn.Linear(1, self.acceptance_prob_classify_num_class)
            else:
                logging.info("Predicting acceptance prob by Regressing/Ranking/Likelihood...")

            # Make the acceptance prob branch
            self.acceptance_prob = nn.Sequential()
            for i in range(self.acceptance_prob_num_layers):
                if i == 0:
                    input_channel_layer = self.prop_feats[0].out_channels
                else:
                    input_channel_layer = self.acceptance_prob_num_channels

                if i== self.acceptance_prob_num_layers-1:
                    output_channel_layer = output_channels
                else:
                    output_channel_layer = self.acceptance_prob_num_channels

                self.acceptance_prob.add_module('layer_' + str(i), nn.Conv2d(input_channel_layer, output_channel_layer, 1))
                if i< self.acceptance_prob_num_layers-1:
                    self.acceptance_prob.add_module('relu_' + str(i), nn.ReLU())

            logging.info(self.acceptance_prob)

    def forward(self, x):
        """
        :param x: torch.size([batch, 3   , 512, 1760]
        :return:
        :param cls               = probabilities before smax torch.size([batch x (H*num_anchors*W) x num_classes])
        :param prob              = probabilities after  smax torch.size([batch x (H*num_anchors*W) x num_classes])
        :param bbox_2d           = torch.size([batch x (H*num_anchors*W) x           4])
        :param bbox_3d           = torch.size([batch x (H*num_anchors*W) x          11])
        :param feat_size         = [32, 110]
        :param rois_cloned       = torch.Size([batch x (H*num_anchors*W) x          5])
        :param rois_3d_cloned    = torch.Size([batch x (H*num_anchors*W) x         11])
        :param rois_2d_cen_cloned= torch.Size([batch x (H*num_anchors*W) x          2])
        """
        #print("Init shape", x.shape)          # [batch, 3   , 512, 1760]
        x          = self.base(x)                                # torch.Size([batch x 1024 x feat_H x feat_W)])
        prop_feats = self.prop_feats(x)                          # torch.Size([batch x  512 x feat_H x feat_W)])
        cls        = self.cls(prop_feats)                        # torch.Size([batch x  144 x feat_H x feat_W)])

        batch_size = x.size(0)
        feat_h     = cls.size(2)
        feat_w     = cls.size(3)

        # bbox 2d
        bbox_x     = self.bbox_x(prop_feats)                     # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_y     = self.bbox_y(prop_feats)                     # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_w     = self.bbox_w(prop_feats)                     # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_h     = self.bbox_h(prop_feats)                     # torch.Size([batch x num_anchors x feat_H x feat_W)])

        # bbox 3d
        bbox_x3d   = self.bbox_x3d(prop_feats)                    # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_y3d   = self.bbox_y3d(prop_feats)                    # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_z3d   = self.bbox_z3d(prop_feats)                    # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_w3d   = self.bbox_w3d(prop_feats)                    # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_h3d   = self.bbox_h3d(prop_feats)                    # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_l3d   = self.bbox_l3d(prop_feats)                    # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_alpha = self.bbox_alpha(prop_feats)                  # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_axis  = self.sigmoid(self.bbox_axis(prop_feats))     # torch.Size([batch x num_anchors x feat_H x feat_W)])
        bbox_head  = self.sigmoid(self.bbox_head(prop_feats))     # torch.Size([batch x num_anchors x feat_H x feat_W)])

        # reshape for cross entropy along dimension 1
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)  #[batch,  4 , 32*36          , 110]

        # score probabilities
        prob = self.softmax(cls)

        # reshape for consistency
        bbox_x = flatten_tensor(bbox_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_alpha = flatten_tensor(bbox_alpha.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_axis = flatten_tensor(bbox_axis.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_head = flatten_tensor(bbox_head.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_alpha, bbox_alpha.clone(), bbox_axis, bbox_head), dim=2)

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)   #[batch x num_classes x (H*num_anchors) x W ] --> [batch x (H*num_anchors*W) x num_classes]
        prob = flatten_tensor(prob) #[batch x num_classes x (H*num_anchors) x W ] --> [batch x (H*num_anchors*W) x num_classes]

        if self.predict_acceptance_prob:
            acceptance_prob_raw = self.acceptance_prob(prop_feats)

            if self.acceptance_prob_mode == "classify":
                if self.acceptance_prob_classify_type == "classify":
                    acceptance_prob     = None

                    acceptance_prob_cls = acceptance_prob_raw.view(batch_size, self.acceptance_prob_classify_num_class, feat_h * self.num_anchors, feat_w)
                    acceptance_prob_cls = flatten_tensor(acceptance_prob_cls)
                elif self.acceptance_prob_classify_type == "regress_then_classify":
                    acceptance_prob     = self.sigmoid(flatten_tensor(acceptance_prob_raw.view(batch_size, 1, feat_h * self.num_anchors, feat_w)))

                    acceptance_prob_2   = acceptance_prob_raw.view(batch_size*feat_h * self.num_anchors*feat_w, 1)
                    acceptance_prob_2   = self.acceptance_prob_classifier(acceptance_prob_2) #batch*H*num_anchors*W x nClass
                    acceptance_prob_cls = acceptance_prob_2.view(batch_size, feat_h * self.num_anchors * feat_w, self.acceptance_prob_classify_num_class)

                acceptance_prob_cls = self.sigmoid(acceptance_prob_cls)
            else:
                acceptance_prob     = self.sigmoid(flatten_tensor(acceptance_prob_raw.view(batch_size, 1, feat_h * self.num_anchors, feat_w)))
                acceptance_prob_cls = None
        else:
            acceptance_prob = None
            acceptance_prob_cls = None

        # =====================================================================
        # The first value of feat_size is zero and therefore it will run once
        # Update rois when different feature size is encountered
        # =====================================================================
        if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
            self.feat_size = [feat_h, feat_w]
            self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
            self.rois = self.rois.type(torch.cuda.FloatTensor)  # torch.Size([(H*num_anchors*W) x 11])

            # more computations
            self.rois_3d = self.anchors[self.rois[:, 4].type(torch.LongTensor), :]
            self.rois_3d = torch.tensor(self.rois_3d, requires_grad=False).type(torch.cuda.FloatTensor) # torch.Size([(H*num_anchors*W) x 11])

            # compute 3d transform
            rois_widths  = self.rois[:, 2] - self.rois[:, 0] + 1.0
            rois_heights = self.rois[:, 3] - self.rois[:, 1] + 1.0
            rois_ctr_x   = self.rois[:, 0] + 0.5 * (rois_widths)   # torch.Size([(H*num_anchors*W) ])
            rois_ctr_y   = self.rois[:, 1] + 0.5 * (rois_heights)  # torch.Size([(H*num_anchors*W) ])
            self.rois_2d_cen = torch.cat((rois_ctr_x.unsqueeze(1), rois_ctr_y.unsqueeze(1)), dim=1) # torch.Size([(H*num_anchors*W) x 2])

        # =====================================================================
        # Clone rois based on batch_size
        # =====================================================================
        rois_cloned        = self.rois.clone()       .unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([batch x (H*num_anchors*W) x 5 ])
        rois_3d_cloned     = self.rois_3d.clone()    .unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([batch x (H*num_anchors*W) x 11])
        rois_2d_cen_cloned = self.rois_2d_cen.clone().unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([batch x (H*num_anchors*W) x 2 ])

        if self.training:
            return cls, prob, bbox_2d, bbox_3d, feat_size, rois_cloned, rois_3d_cloned, rois_2d_cen_cloned, acceptance_prob, acceptance_prob_cls
        else:
            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois, acceptance_prob, acceptance_prob_cls


def build(conf, phase):

    train = phase.lower() == 'train'
    if train:
        logging.info("Using densenet121 pre-trained on image-net as the base network..")
    else:
        logging.info("Using densenet121 initialized from scratch as the base network..")
    densenet121 = models.densenet121(pretrained=train)

    rpn_net = RPN(phase, densenet121.features, conf)

    if train: rpn_net.train()
    else: rpn_net.eval()

    return rpn_net
