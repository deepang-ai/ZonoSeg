# ZonoSeg: A Zonal-aware Prostate Segmentation Framework on 3D MRI Images

> This work is accepted by [IEEE Transactions on Medical Imaging](https://ieeexplore.ieee.org/)

[![GitHub](https://img.shields.io/badge/Code-GitHub-181717?logo=github)](https://github.com/deepang-ai/ZonoSeg)
[![Hugging Face](https://img.shields.io/badge/Weights-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/deepang/ZonoSeg)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

ZonoSeg is a hierarchical encoder-decoder framework for zonal-aware prostate
segmentation. It focuses on the ambiguous interface between the central gland
and peripheral zone through probabilistic boundary modeling and explicit zone
decoupling.

# Network Architecture

The ZonoSeg Block combines multi-scale Gaussian smoothing and edge cues to
construct a continuous probabilistic boundary field. The zone-adaptive
decoupling head separates boundary and region representations for more precise
central-gland and peripheral-zone delineation.

![Overview](./figures/Overview.png)

# Data Description

### Dataset: Multi-center prostate MRI

Modality: 3D MRI

The experiments cover eight multi-center prostate MRI datasets, including
ProstateX, PROMISE MRI BMC, and MSD Task05 Prostate. The target regions are the
central gland and peripheral zone.

Convert each dataset to the standard nnU-Net v2 format. See
`documentation/dataset_format.md` for the expected directory layout.

# Training

This repository follows the standard nnU-Net v2 planning, preprocessing, and
training workflow. The ZonoSeg network is implemented in
`nnunetv2/nets/ZonoSeg.py`, with trainer variants in
`nnunetv2/training/nnUNetTrainer/variants/zonoseg/`.
The Triton kernels used by the model are in
`nnunetv2/nets/zonoseg_kernel/`.

Install the package and configure the nnU-Net environment variables described
in `documentation/setting_up_paths.md`.

The model requires a CUDA-enabled PyTorch environment with Triton installed.

Plan and preprocess a dataset:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

Train a ZonoSeg model:

```bash
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr ZonoSeg_128Trainer
```

# Evaluation

Use the standard nnU-Net v2 prediction command:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres
```

Pretrained weights are available in the companion
[ZonoSeg-hf](https://huggingface.co/deepang/ZonoSeg) repository.

# Results

![Benchmark](./figures/Benchmark.png)

# Visualization

![Visualization](./figures/Visualization.png)

![Boundary visualization](./figures/Visualization2.png)

# BibTeX

```bibtex
@article{li2026zonoseg,
  title={ZonoSeg: A Zonal-aware Prostate Segmentation Framework on 3D MRI Images},
  author={Li, Yunhao and Wang, Aoying and Wang, Qiong and Hu, Ying and Qin, Jing and Pang, Yan},
  journal={IEEE Transactions on Medical Imaging},
  year={2026}
}
```
