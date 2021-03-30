#===============================================================================
# Evaluation Scripts
#===============================================================================

source dependencies/cuda_8.0_env

# === KITTI Val 1 Split ====
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_rpn_3d.py --config=groumd_nms        --restore=50000 --test

# === KITTI Val 2 Split ====
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_rpn_3d.py --config=groumd_nms_split2 --restore=50000 --test
