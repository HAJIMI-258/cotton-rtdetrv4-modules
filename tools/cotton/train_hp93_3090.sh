#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python train.py   -c configs/cotton/rtv4_hgnetv2_m_cotton_hp93_3090.yml   --use-amp   --seed=0
