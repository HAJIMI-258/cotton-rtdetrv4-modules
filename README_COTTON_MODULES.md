# Cotton RT-DETRv4 Training Modules

This repository is a research working copy of RT-DETRv4 with cotton disease and pest detection modules added for code review and follow-up optimization.

## Main Modified Files

- `engine/rtv4/hybrid_encoder.py`
  - Prewitt-Franklin edge-guided enhancement
  - CARAFE upsampling
  - BiFPN-style fusion
  - Coordinate Attention, ECA, CBAM, LSK
  - FasterNet partial convolution block
  - RepVGG enhancement block
  - Multi-scale gather-distribute context
  - Frozen safe residual gates for stable from-scratch training
- `engine/rtv4/rtv4_criterion.py`
  - NWD box loss
  - More stable distillation loss handling
  - Auxiliary distillation skip logic
- `engine/core/yaml_config.py`
  - Teacher model construction support
- `configs/cotton/rtv4_hgnetv2_s_cotton_teacher_paper_plus_all.yml`
  - Main cotton all-module training config

## Validation Already Run

- Full module forward pass on CUDA.
- Full module backward pass on CUDA with no NaN gradients.
- Official `train.py` smoke test on a local RTX 4060 Laptop GPU:
  - 100 training images
  - 20 validation images
  - 1 epoch
  - AMP enabled
  - all modules enabled
  - NWD enabled
  - completed without crash or NaN

## Notes

Dataset, pretrained weights, training outputs, and local machine paths are intentionally not committed.

Before full training, set the dataset and teacher/pretrained paths in the cotton config:

```yaml
train_dataloader:
  dataset:
    img_folder: ./data/cotton_rtdetr_dataset/images/train
    ann_file: ./data/cotton_coco_zero_based/annotations/instances_train.json
```

For stability, the safe residual scale gates are frozen by default:

```yaml
safe_module_init_scale: 0.001
safe_module_trainable_scale: False
safe_module_max_scale: 0.005
```

This avoids the gradient-clipping failure mode observed when the scale gates were trainable from scratch.
