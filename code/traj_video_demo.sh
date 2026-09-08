#!/bin/bash
set -u
cd /path/to/rlhnd
source .venv/bin/activate
NANO=logs/realhand_v2_cosmos3_nano_hot3dfix_h100_260901/checkpoints/stage1_20k.ckpt
PB=/path/to/hand_tracking_ablation/results/pred_cache_baselines
HFV=/path/to/HandFlow/output/viz
HFD=/path/to/HandFlow/output/dominoes_viz/pred.npz
OUT=tmp/traj_videos
mkdir -p $OUT
run() { python scripts/scripts_eval/viz_traj3d.py "$@" || echo "VFAIL $*"; }
# ARCTIC clip 1 (floor 0.8691): ours + gt
run --dataset ARCTICEGO-CLIP-VAL --clip_idx 1 --model_path $NANO --up_axis 2 --floor_y 0.8691 --video_out $OUT/arctic1_rlhnd_3d.mp4
run --dataset ARCTICEGO-CLIP-VAL --clip_idx 1 --model_path $NANO --up_axis 2 --floor_y 0.8691 --gt --video_out $OUT/arctic1_gt_3d.mp4
# HOT3D-ARIA clip 11 (floor -0.5068): ours
run --dataset HOT3D-ARIA-CLIP-TEST --clip_idx 11 --model_path $NANO --up_axis 2 --floor_y -0.5068 --video_out $OUT/aria11_rlhnd_3d.mp4
# EgoDex dominoes clip 1 (floor 1.0787, y-up): ours + handflow
run --dataset EGODEX-TEST15 --clip_idx 1 --model_path $NANO --up_axis 1 --rad_scale 3.0 --floor_y 1.0787 --video_out $OUT/egodex1_rlhnd_3d.mp4
run --dataset EGODEX-TEST15 --clip_idx 1 --model_path $NANO --up_axis 1 --rad_scale 3.0 --floor_y 1.0787 --handflow $HFD --video_out $OUT/egodex1_handflow_3d.mp4
echo TRAJ_VIDEO_DEMO_DONE
