# DIAF-Net

Official clean implementation of **DIAF-Net: Dispersion-Informed Alignment and Fusion Network for Multi-Contrast MRI Super-Resolution**.

This release contains only the full DIAF-Net used in the paper. Legacy IDM names, unrelated comparison networks, capacity-matched ablation replacements, diagnostic probes, training-time heatmap export, and unused medical-segmentation/data-loading code have been removed.

## Method

DIAF-Net takes a low-resolution target contrast and a potentially misaligned high-resolution reference contrast. It contains three named components:

- **SFDE — Spatial Feature Dispersion Estimator:** estimates a Gaussian feature distribution and converts its scale into a Spatial Feature Dispersion Map (SFDM).
- **DGCA — Dispersion-Guided Coarse-to-Fine Alignment:** corrects dominant displacement with a bounded base flow, followed by SFDM-modulated deformable alignment.
- **DGFR — Dispersion-Guided Fusion Refinement:** performs target-anchored cross-contrast aggregation, representative-token routing, and adaptive local-global interaction.

SFDE is evaluated once on the highest-resolution target feature. The resulting SFDM is resized into a four-scale pyramid and shared by DGCA and DGFR at each encoder level.

The training objective is

```text
L = L1 + 1e-4 * LFCL
```

where **LFCL** is the Likelihood-based Feature Coding Loss.

## Repository structure

```text
DIAF-Net/
├── configs/
│   └── config.py                  # Paper/default configuration
├── datasets/
│   └── multi_contrast_mri.py      # BraTS2020 and fastMRI pair loader
├── models/
│   ├── sfde.py                    # SFDE and SFDM construction
│   ├── dgca.py                    # DGCA
│   ├── dgfr.py                    # DGFR
│   └── diaf_net.py                # Full DIAF-Net
├── losses.py                      # LFCL and complete training objective
├── train.py                       # Training and validation
├── test.py                        # Robustness evaluation and image export
├── utils.py                       # Metrics, seeds, and image I/O
└── requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
conda create -n diafnet python=3.10 -y
conda activate diafnet
pip install -r requirements.txt
```

DGCA uses `torchvision.ops.deform_conv2d`. PyTorch and torchvision must be installed as matching builds. Install the appropriate CUDA wheels from the official PyTorch instructions before installing the remaining packages when necessary.

## Data preparation

The repository expects preprocessed 2D image pairs. Dataset downloading and conversion are not included because the original datasets have their own access and redistribution conditions.

All paired folders must use identical filenames.

### BraTS2020

```text
datasets/BraTS2020/
├── train/
│   ├── oriT2/                     # HR target ground truth
│   ├── oriT1/                     # HR reference
│   └── orLRbicT2/x4/              # x4 LR target input
├── val/
│   ├── oriT2/
│   ├── oriT1/
│   └── orLRbicT2/x4/
└── val_idm/                       # Synthetic robustness evaluation
    ├── oriT2/
    ├── oriT1/                     # clean
    ├── oriT1_mis3/
    ├── oriT1_mis6/
    ├── oriT1_mis9/
    └── orLRbicT2/x4/
```

The model resizes the LR target to the reference resolution internally. Therefore, the LR folder may contain native-resolution LR images or pre-upsampled LR images, provided that the preprocessing protocol is consistent across all methods.

### fastMRI

The supplied project convention uses PDFS/FSPD as the target and PD as the reference:

```text
datasets/fastMRI/
├── train/
│   ├── oriFSPD/                   # HR PDFS/FSPD target
│   ├── oriPD/                     # HR PD reference
│   └── orLRbicFSPD/x4/            # x4 LR target input
└── val/
    ├── oriFSPD/
    ├── oriPD/
    └── orLRbicFSPD/x4/
```

## Training

Paper-style BraTS2020 training:

```bash
python train.py \
  --data-root datasets/BraTS2020 \
  --dataset brats2020 \
  --scale 4 \
  --batch-size 2 \
  --epochs 200 \
  --learning-rate 1e-4 \
  --lfcl-weight 1e-4 \
  --device cuda \
  --output-dir results/diaf_brats_x4
```

The default training augmentation uses 50% clean references and 50% randomly misaligned references with rotations and translations bounded by 5 degrees and 5 pixels. Disable it with:

```bash
python train.py ... --no-train-misalignment
```

Resume training:

```bash
python train.py \
  --data-root datasets/BraTS2020 \
  --dataset brats2020 \
  --resume results/diaf_brats_x4/checkpoints/latest.pth
```

Checkpoints and training history are saved under the selected output directory:

```text
results/diaf_brats_x4/
├── config.json
├── history.json
└── checkpoints/
    ├── best.pth
    └── latest.pth
```

## Evaluation

Evaluate the same BraTS2020 checkpoint on clean, mis3, mis6, and mis9 conditions:

```bash
python test.py \
  --checkpoint results/diaf_brats_x4/checkpoints/best.pth \
  --data-root datasets/BraTS2020 \
  --dataset brats2020 \
  --scale 4 \
  --conditions clean mis3 mis6 mis9 \
  --device cuda \
  --save-dir results/diaf_brats_x4/test_outputs
```

Evaluate fastMRI with natural misalignment:

```bash
python test.py \
  --checkpoint results/diaf_fastmri_x4/checkpoints/best.pth \
  --data-root datasets/fastMRI \
  --dataset fastmri \
  --scale 4 \
  --conditions native \
  --device cuda
```

The evaluation script reports PSNR, SSIM, and RMSE. When `--save-dir` is specified, reconstructed images and `metrics.json` are also written.

## Model API

```python
import torch
from models import DIAFNet

model = DIAFNet().cuda().eval()
target_lr = torch.randn(1, 3, 56, 56, device="cuda")
reference_hr = torch.randn(1, 3, 224, 224, device="cuda")

with torch.inference_mode():
    output = model(target_lr, reference_hr)

print(output.shape)  # [1, 3, 224, 224]
```

To access the SFDE output and SFDM pyramid during research:

```python
with torch.inference_mode():
    output, auxiliary = model(target_lr, reference_hr, return_aux=True)
    sfdm = auxiliary["sfdm"]
    pyramid = auxiliary["sfdm_pyramid"]
```

The default model contains approximately **8.27 M trainable parameters**, matching the paper configuration.

## Reproducibility notes

- The public release focuses on the DIAF-Net training and evaluation path.
- The default model uses dimensions `(64, 128, 192, 288)`, encoder/decoder depths `(2, 2, 2, 2)`, window sizes `(8, 8, 8, 4)`, attention heads `(4, 8, 8, 12)`, and representative tokens `(1, 1, 2, 2)`.
- Images are normalized from `[0, 1]` to `[-1, 1]`.
- Model selection uses validation PSNR.
- Results can depend on dataset preprocessing, subject-level splits, PyTorch/CUDA versions, and random seeds.

## Citation

Please cite the paper after publication:

```bibtex
@inproceedings{diafnet,
  title     = {DIAF-Net: Dispersion-Informed Alignment and Fusion Network for Multi-Contrast MRI Super-Resolution},
  author    = {Jiang, Cunang and Ding, Chenhao and Li, Teng and Zhou, Yuchen and Dai, Yu},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {TBD}
}
```

Update the year, page numbers, and DOI after the final bibliographic record becomes available.
