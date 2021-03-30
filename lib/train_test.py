

from lib.rpn_util import *
from lib.loss.rpn_3d import *

def training_update(conf, net, criterion_det, optimizer, train_loader, images, imobjs, tracker, cnt_batch, idx):
    # Train mode
    net.train()

    # Remember to freeze the batch norm or layers if we train
    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist
    freeze_layers(net, freeze_blacklist, freeze_whitelist)
    if 'freeze_bn' in conf and conf.freeze_bn: freeze_bn(net)

    # Adjust learning rate
    adjust_lr(conf, optimizer, cnt_batch, len(train_loader))

    # forward
    # cls                  = torch.size([batch x (H * num_anchors * W) x num_classes])
    # prob                 = torch.size([batch x (H * num_anchors * W) x num_classes])
    # bbox_2d              = torch.size([batch x (H * num_anchors * W) x           4])
    # bbox_3d              = torch.size([batch x (H * num_anchors * W) x          11])
    # feat_size            = [32, 110]
    # rois_new             = torch.Size([batch x (H * num_anchors * W) x          5])
    # rois_3d_new          = torch.Size([batch x (H * num_anchors * W) x         11])
    # rois_2d_cen_cloned   = torch.Size([batch x (H * num_anchors * W) x          2])
    # bbox_acceptance_prob = torch.Size([batch x (H * num_anchors * W) x          2]) or None
    cls, prob, bbox_2d, bbox_3d, feat_size, rois_cloned, rois_3d_cloned, rois_2d_cen_cloned, bbox_acceptance_prob, bbox_acceptance_prob_cls = net(images)

    # Calculate loss
    det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, rois_cloned, rois_3d_cloned, rois_2d_cen_cloned, bbox_acceptance_prob, bbox_acceptance_prob_cls)

    # Backprop
    loss_backprop(det_loss, net, optimizer, conf= conf, iteration= idx)

    # Update stats
    compute_stats(tracker, det_stats)

def test_kitti_3d(test_dataset_name, net, conf, paths, test_iter= None, split_name= "validation", test_loader= None, criterion= None, print_loss= True):
    """
    Test the KITTI framework for object detection in 3D
    """
    save_before_nms = False
    save_p2         = False
    save_constraint = False

    # necessary paths
    root_data_folder = paths.data
    if test_iter is None:
        numpy_save_folder = os.path.join(paths.results,    'results_test')
    else:
        numpy_save_folder = os.path.join(paths.results,     'results_' + str(test_iter))
    predictions_folder    = os.path.join(numpy_save_folder, 'data')
    mkdir_if_missing(predictions_folder, delete_if_exist=False)

    if save_before_nms:
        predictions_before_nms_folder = os.path.join(numpy_save_folder, 'data_before_nms')
        mkdir_if_missing(predictions_before_nms_folder, delete_if_exist=True)

    if save_p2:
        p2_folder = os.path.join(numpy_save_folder, 'p2')
        mkdir_if_missing(p2_folder, delete_if_exist=True)

    if test_iter is None:
        # fix paths slightly
        _, test_iter, _ = file_parts(predictions_folder.replace('/data', ''))
        test_iter = test_iter.replace('results_', '')

    # Eval mode
    net.eval()
    # Deactivate autograd engine
    with torch.no_grad():

        if not test_loader:
            logging.info("\nTesting the model on {} images without dataloader...".format(split_name))

            from lib.imdb_util import read_kitti_cal

            imlist = list_files(os.path.join(root_data_folder, test_dataset_name, split_name, 'image_2', ''),
                                '*' + conf.datasets_train[0]['im_ext'])

            # Pre-processing object for each image
            preprocess = Preprocess([conf.test_scale], conf.image_means, conf.image_stds)

            # Init
            start_time = time()

            for img_index, impath in enumerate(imlist):
                # Read image on your own
                im = imread(impath)
                _, id, _ = file_parts(impath)

                # Read in calib
                p2 = read_kitti_cal(
                    os.path.join(root_data_folder, test_dataset_name, 'validation', 'calib', id + '.txt'))

                # forward test image and nms
                aboxes = im_detect_3d(im, net, conf, preprocess, p2)

                # post-nms and score thresholding
                aboxes = aboxes[:min(conf.nms_topN_post, aboxes.shape[0])]

                scores_img = aboxes[:, 4]
                score_threshold = conf.score_thres
                gt_thresh_indices = np.where(scores_img > score_threshold)[0]
                aboxes = aboxes[gt_thresh_indices]

                boxes_image = convert_image_predictions_to_correct_entries(aboxes, conf, p2)
                _ = write_image_boxes_to_txt_file(boxes_image, conf, save_folder= predictions_folder, id= id, save_constraint= save_constraint)

                # display stats
                if (img_index + 1) % 500 == 0 or img_index == len(imlist) - 1:
                    time_str, dt = compute_eta(start_time, img_index + 1, len(imlist))
                    logging.info('images {:4d}/{:4d}, dt: {:0.3f}, time_left: {}'.format(img_index + 1, len(imlist), dt, time_str))

        else:
            logging.info("\nTesting the model on {} images with dataloader...".format(split_name))

            display_string = "Saving boxes after NMS"
            if save_before_nms or save_p2:
                if save_before_nms:
                    display_string += ", boxes before nms"
                if save_p2:
                    display_string += ", p2 of each image"
            logging.info(display_string)

            if save_constraint:
                logging.info("Writing consistency constraint in the prediction files...")

            if criterion is None:
                criterion = RPN_3D_loss(conf)

            start_time = time()
            tracker    = edict()

            for batch_index, (im, imobjs) in enumerate(test_loader):
                # Forward pass through the network
                cls, prob, bbox_2d, bbox_3d, feat_size, rois_cloned, rois_3d_cloned, rois_2d_cen_cloned, bbox_acceptance_prob, bbox_acceptance_prob_cls = net(im)

                # Calculate loss
                if print_loss:
                    # Send the cloned copies of the variables otherwise they get changed because of the manipulation in
                    # the loss. This results in wrong MAP values
                    if bbox_acceptance_prob is not None:
                        bbox_acceptance_prob_2 = bbox_acceptance_prob.clone()
                    else:
                        bbox_acceptance_prob_2 = None

                    if bbox_acceptance_prob_cls is not None:
                        bbox_acceptance_prob_cls_2 = bbox_acceptance_prob_cls.clone()
                    else:
                        bbox_acceptance_prob_cls_2 = None

                    _, det_stats = criterion(cls.clone(), prob.clone(), bbox_2d.clone(), bbox_3d.clone(), imobjs, feat_size, rois_cloned.clone(), rois_3d_cloned, rois_2d_cen_cloned, bbox_acceptance_prob_2, bbox_acceptance_prob_cls_2)
                    compute_stats(tracker, det_stats)

                # Display stats
                if (batch_index + 1) % 100 == 0 or batch_index == len(test_loader) - 1:
                    if print_loss:
                        log_stats(tracker, iteration= batch_index, start_time= start_time, start_iter= 0, max_iter= len(test_loader), show_max= True)
                    else:
                        time_str, dt = compute_eta(start_time, batch_index + 1, len(test_loader))
                        logging.info('testing {:4d}/{:4d}, dt: {:0.3f}, eta: {}'.format(batch_index + 1, len(test_loader), dt, time_str))

                # Convert predictions to true variables after NMS
                # Shape = 2 x num_boxes x 20ish
                boxes_all, boxes_before_nms_all = convert_network_predictions_to_true_variables_and_nms(prob, bbox_2d, bbox_3d, rois_cloned[0], imobjs, conf, bbox_acceptance_prob= bbox_acceptance_prob, bbox_acceptance_prob_cls= bbox_acceptance_prob_cls, save_constraint= save_constraint, save_before_nms= save_before_nms)

                for img_index in range(im.shape[0]):
                    p2       = imobjs[img_index]['p2']
                    impath   = imobjs[img_index]['path']
                    _, id, _ = file_parts(impath)

                    _ = write_image_boxes_to_txt_file(boxes_all[img_index], conf, save_folder= predictions_folder, id= id, save_constraint= save_constraint)
                    if save_before_nms:
                        _ = write_image_boxes_to_txt_file(boxes_before_nms_all[img_index], conf, save_folder= predictions_before_nms_folder, id= id, save_constraint= save_constraint)
                    if save_p2:
                        save_numpy(id + '.npy', p2, save_folder= p2_folder, show_message=False)

    logging.info("Running evaluation on {}".format(predictions_folder))
    evaluate_kitti_results_verbose(root_data_folder, test_dataset_name= test_dataset_name, results_folder= predictions_folder, split_name= split_name, test_iter= test_iter, conf= conf)
