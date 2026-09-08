#!/bin/bash
# qualitative batch: raw overlays (GT+6 models) + v7-standard trajectories + concat grid per clip
# usage: qual_runner.sh <DATASET> <UP_AXIS> <BASELINE_TAG> <clip> [clip ...]
set -u
cd /path/to/rlhnd
source .venv/bin/activate
DS=$1; UP=$2; TAG=$3; shift 3
NANO=logs/realhand_v2_cosmos3_nano_hot3dfix_h100_260901/checkpoints/stage1_20k.ckpt
DH=logs/dreamhand_standard_260831/checkpoints/wan22_20k.ckpt
PB=/path/to/hand_tracking_ablation/results/pred_cache_baselines
OUT=tmp/qualitative
mkdir -p $OUT/$DS
declare -A SRC=( [gt]="--gt" [rlhnd]="" [dreamhand]="--exp dreamhand_standard_260831" \
  [hawor]="--baseline $PB/hawor__$TAG.npz" [hamer]="--baseline $PB/hamer__$TAG.npz" \
  [wilor]="--baseline $PB/wilor__$TAG.npz" [haptic]="--baseline $PB/haptic__$TAG.npz" )
for C in "$@"; do
  P=$OUT/$DS/$(printf %05d "$C")
  echo "=== $DS clip $C"
  [ -f "${P}_concat.mp4" ] && { echo "ALREADY_DONE $DS $C"; continue; }
  # gt overlay is written last, so its presence means all 7 overlay videos exist
  if ls ${P}_*__gt.mp4 >/dev/null 2>&1; then
    echo "OVERLAYS_EXIST $DS $C"
  else
    python scripts/scripts_eval/viz_compare.py --dataset "$DS" --clips "$C" --raw_overlay --gt_row \
      --model_path $NANO --model_label RLHND --model2_path $DH --model2_label DreamHand \
      --baseline_rows "HaWoR=$PB/hawor__$TAG.npz;HaMeR=$PB/hamer__$TAG.npz;WiLoR=$PB/wilor__$TAG.npz;Haptic=$PB/haptic__$TAG.npz" \
      --out_dir $OUT || { echo "OVERLAY_FAIL $DS $C"; continue; }
  fi
  ls ${P}_*__rlhnd.mp4 >/dev/null 2>&1 || { echo "CLIP_SKIPPED $DS $C"; continue; }
  # trajectory pass 1: shared floor across all methods (v7 standard)
  MINS=/tmp/qual_mins_${DS}_${C}.txt; rm -f "$MINS"
  for M in gt rlhnd dreamhand hawor hamer wilor haptic; do
    V=$(python scripts/scripts_eval/viz_traj3d.py --dataset "$DS" --clip_idx "$C" --model_path $NANO \
        --up_axis "$UP" ${SRC[$M]} --res 320 --out /tmp/qual_probe.png 2>/dev/null \
        | grep PRED_WORLD_MIN | awk '{print $2}')
    [ -n "$V" ] && echo "$V" >> "$MINS" || echo "TRAJ_PROBE_FAIL $DS $C $M"
  done
  [ -s "$MINS" ] || { echo "NO_TRAJ $DS $C"; continue; }
  GMIN=$(python -c "print(min(float(l) for l in open('$MINS')))")
  echo "FLOOR $DS $C $GMIN"
  # trajectory pass 2: full renders on the shared floor
  for M in gt rlhnd dreamhand hawor hamer wilor haptic; do
    [ -f "${P}_traj3d_${M}.png" ] && continue
    python scripts/scripts_eval/viz_traj3d.py --dataset "$DS" --clip_idx "$C" --model_path $NANO \
      --up_axis "$UP" ${SRC[$M]} --floor_y "$GMIN" --out "${P}_traj3d_${M}.png" \
      || echo "TRAJ_FAIL $DS $C $M"
  done
  python tmp/make_concat.py "$P" "${P}_concat.mp4" || echo "CONCAT_FAIL $DS $C"
done
echo "QUAL_RUNNER_DONE $DS $*"
