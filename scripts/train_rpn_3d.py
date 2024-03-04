

"""
    Sample Run:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_rpn_3d.py --config=kitti_3d_multi_warmup_final_v1
    CUDA_VISIBLE_DEVICES=0 python scripts/train_rpn_3d.py --config=kitti_3d_multi_warmup_final_v1         --restore=10000
    CUDA_VISIBLE_DEVICES=0 python scripts/train_rpn_3d.py --config=kitti_3d_multi_warmup_final_v1_batch_3 --restore=40000 --test
    CUDA_VISIBLE_DEVICES=0 python scripts/train_rpn_3d.py --config=kitti_v1_general_consistency           --restore output/kitti_3d_v2_step/weights/model_70000_pkl
"""
# ============================================
# python modules
# ============================================
import os, sys
sys.path.append(os.getcwd())

import numpy as np
from getopt import getopt

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
np.set_printoptions(suppress=True)

# ============================================
# custom modules
# ============================================
from lib.core import *
from lib.imdb_util import *
from lib.loss.rpn_3d import *
from lib.train_test import *

def main(argv):

    # ============================================
    # parse arguments
    # ============================================
    opts, args = getopt(argv, '', ['config=', 'restore=', 'test'])

    # defaults
    conf_name = None
    restore = None

    # read opts
    for opt, arg in opts:

        if opt in ('--config'): conf_name = arg
        if opt in ('--restore'): restore = arg
        if opt in ('--test'):
            test = True
        else:
            test = False

    # required opt
    if conf_name is None:
        raise ValueError('Please provide a configuration file name, e.g., --config=<config_name>')

    # ============================================
    # basic setup
    # ============================================

    conf = init_config(conf_name)
    paths = init_training_paths(conf_name)

    init_torch(conf.rng_seed, conf.cuda_seed)
    init_log_file(paths.logs)

    if 'copy_stats' in conf and conf.copy_stats and 'pretrained' in conf:
        copy_stats(paths.output, conf.pretrained)

    # defaults
    start_iter = 0
    tracker = edict()
    iterator = None

    train_split   = "training"
    test_split    = "validation"
    dataset = Dataset(conf, paths.data, paths.output)

    generate_anchors(conf, dataset.imdb, paths.output)
    compute_bbox_stats(conf, dataset.imdb, paths.output)

    # ============================================
    # defaults mostly to False
    # ============================================
    conf.infer_2d_from_3d = False if not ('infer_2d_from_3d' in conf) else conf.infer_2d_from_3d
    conf.bbox_un_dynamic = False if not ('bbox_un_dynamic' in conf) else conf.bbox_un_dynamic

    # ============================================
    # store and show config
    # ============================================
    pickle_write(os.path.join(paths.output, 'conf.pkl'), conf)
    pretty = pretty_print('conf', conf)
    logging.info(pretty)

    # ============================================
    # network and loss
    # ============================================

    # training network
    rpn_net, optimizer = init_training_model(conf, paths.output)

    # setup loss
    criterion_det = RPN_3D_loss(conf)

    # custom pretrained network
    if 'pretrained' in conf and not test:
        load_weights(rpn_net, conf.pretrained)

    # resume training
    if restore:
        start_iter = resume_checkpoint(optimizer, rpn_net, paths.weights, restore) - 1

    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist
    freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist, verbose=True)
    if 'slow_bn' in conf and conf.slow_bn: slow_bn(rpn_net, conf.slow_bn)
    if 'freeze_bn' in conf and conf.freeze_bn: freeze_bn(rpn_net)

    optimizer.zero_grad()
    iteration = start_iter

    start_time = time()

    if not test:
        # ============================================
        # train
        # ============================================
        logging.info("\nTraining the model...")
        while iteration < conf.max_iter:

            # next iteration
            iterator, images, imobjs = next_iteration(dataset.loader, iterator)

            #  learning rate
            adjust_lr(conf, optimizer, iteration)

            # forward
            cls, prob, bbox_2d, bbox_3d, feat_size, rois, rois_3d, rois_3d_cen, bbox_acceptance_prob, bbox_acceptance_prob_cls  = rpn_net(images)

            # loss
            det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, rois, rois_3d, rois_3d_cen, bbox_acceptance_prob, bbox_acceptance_prob_cls )

            loss_backprop(det_loss, rpn_net, optimizer, conf=conf, iteration=iteration)

            # keep track of stats
            compute_stats(tracker, det_stats)

            # -----------------------------------------
            # display
            # -----------------------------------------
            if (iteration + 1) % conf.display == 0 and iteration > start_iter:
                lr = get_lr(optimizer)

                # log results
                log_stats(tracker, iteration, start_time, start_iter, conf.max_iter, lr= lr)

                # reset tracker
                tracker = edict()

            # ============================================
            # test network
            # ============================================
            if ((iteration + 1) % conf.snapshot_iter == 0 or iteration == conf.max_iter-1) and iteration > start_iter:
                # store checkpoint
                save_checkpoint(optimizer, rpn_net, paths.weights, (iteration + 1))

                if conf.do_test:

                    # eval mode
                    rpn_net.eval()

                    # necessary paths
                    results_path = os.path.join(paths.results, 'results_{}'.format((iteration + 1)))

                    # -----------------------------------------
                    # test kitti
                    # -----------------------------------------
                    if conf.test_protocol.lower() == 'kitti':
                        results_path = os.path.join(results_path, 'data')
                        mkdir_if_missing(results_path, delete_if_exist=True)

                        test_kitti_3d_old(conf.dataset_test, rpn_net, conf, results_path, paths.data)

                        #test_kitti_3d("kitti_split1", rpn_net, conf, paths, test_iter= None,
                        #split_name= test_split, test_loader= None, criterion= criterion_det, print_loss= False)
                    else:
                        logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))

                    # train mode
                    rpn_net.train()

                    freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)
                    if 'slow_bn' in conf and conf.slow_bn: slow_bn(rpn_net, conf.slow_bn)
                    if 'freeze_bn' in conf and conf.freeze_bn: freeze_bn(rpn_net)

            iteration += 1
    else:
        # ============================================
        # test network
        # ============================================
        if conf.test_protocol.lower() == 'kitti':
            logging.info("\nTesting the model on {} images without dataloader...".format(test_split))
            rpn_net.eval()
            results_path = os.path.join(paths.results, 'results_test')
            results_path = os.path.join(results_path, 'data')
            mkdir_if_missing(results_path)
            test_kitti_3d_old(conf.dataset_test, rpn_net, conf, results_path, paths.data)
            #for dataset in test_dataset.list_of_datasets:
            #    test_kitti_3d(dataset['name'], rpn_net, conf, paths, test_iter= None,
            #             split_name= test_split, test_loader= test_dataset.loader, criterion= criterion_det, print_loss= False)
            #test_kitti_3d("kitti_split1", rpn_net, conf, paths, test_iter= None,
            #            split_name= test_split, test_loader= None, criterion= criterion_det, print_loss= False)
        else:
            logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))


# run from command line
if __name__ == "__main__":
    main(sys.argv[1:])
