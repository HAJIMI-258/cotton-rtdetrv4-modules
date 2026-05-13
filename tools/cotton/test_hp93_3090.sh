#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-outputs/rtv4_hgnetv2_m_cotton_hp93_3090/best_stg2.pth}
if [ ! -f "$CKPT" ]; then
  CKPT=outputs/rtv4_hgnetv2_m_cotton_hp93_3090/best_stg1.pth
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python train.py   -c configs/cotton/rtv4_hgnetv2_m_cotton_hp93_3090.yml   --test-only   -r "$CKPT"
