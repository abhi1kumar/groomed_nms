#===============================================================================
# Training Scripts
#===============================================================================

# === KITTI Val 1 Split ====
source dependencies/cuda_8.0_env
# First train the warmup
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_rpn_3d.py --config=kitti_3d_warmup

# Then train the model with uncertainty and GrooMeD-NMS
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_rpn_3d.py --config=groumd_nms


# === KITTI Val 2 Split ====
source dependencies/cuda_8.0_env
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_rpn_3d.py --config=kitti_3d_warmup_split2
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_rpn_3d.py --config=groumd_nms_split2


# === KITTI Test (Full) Split ====
source dependencies/cuda_8.0_env
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_rpn_3d.py --config=kitti_3d_warmup_full_train_2
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_rpn_3d.py --config=groumd_nms_full_train_2


#===============================================================================
# Ablation Studies
#===============================================================================
