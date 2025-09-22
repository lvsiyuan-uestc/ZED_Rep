# ZED_Rep

Reproduction of the paper "Zero-Shot Detection of AI-Generated Images" (ZED) with lightweight PyTorch tooling for training conditional pixel models and evaluating the Δ01 detector used in the publication.

## Features

- **Hierarchical pixel heads** – scripts to train level-0 and level-1 conditional logistic heads (`x_0 \mid y_1` and `x_1 \mid y_2`) with either the residual or mixture formulation described in the paper.
- **Turn-key evaluation utilities** – batch and single-image evaluators that build the multi-scale pyramid, run the trained heads, and report D-scores, Δ01, and |Δ01| with optional visualisations.
- **Threshold selection helper** – compute ROC statistics and balanced-accuracy operating points from exported CSV score tables to derive production-ready decision thresholds.

## Repository structure

| Path | Description |
| --- | --- |
| `train_level0.py` | Training loop for the level-0 conditional head (`x_0 \mid y_1`) with cosine LR schedule, AMP support, and optional logistic-mixture head (`--mix_k`). |
| `train_level1.py` | Training loop for the level-1 conditional head (`x_1 \mid y_2`) featuring warm-up, AMP, gradient clipping, and mixture support. |
| `eval_delta01.py` | Single-image evaluator that rescales the image pyramid per head, prints Δ01 statistics, and optionally writes diagnostic maps. |
| `eval_batch_scores.py` | Batch evaluator that walks a directory or list of images, exports CSV score tables, and summarises |Δ01| statistics. |
| `choose_threshold.py` | Utility for sweeping thresholds on exported scores, computing ROC/AUC, and reporting the best balanced-accuracy point. |
| `src/` | Core library: pixel pyramid builders, logistic likelihoods, mixture heads, and shared evaluation helpers. |

## Getting started

### Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support recommended for training and evaluation
- NumPy, pandas, Pillow

Install dependencies in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pillow
```

### Data preparation

The training scripts expect a dataset loader named `RealPairsDataset` located at `src/dset_realpairs.py`. The original implementation is not distributed here; implement your own version that yields `(x_l, y_{l+1})` pyramid pairs from real images (the authors used COCO 2017 with JPEG augmentation). Place your dataset or list files under `data/` or pass custom paths via `--data_list`.

### Checkpoints

Saved weights are stored in the `runs/` directory (created automatically). Provide `--save` paths for training and reuse the resulting checkpoints for evaluation scripts. No pre-trained weights are bundled with this repository.

## Training

> Ensure that your dataset loader returns tensors compatible with the multi-scale pyramid utilities in `src/pyramid.py` (uint8 tensors for `x_l` and float tensors for `y_{l+1}`).

### Level-0 head (`x_0 \mid y_1`)

Train the residual head (K=1) with automatic mixed precision:

```bash
python train_level0.py \
  --steps 3000 \
  --batch_size 4 \
  --amp \
  --save runs/level0_head_res_coco.pt
```

Switch to the mixture-of-logistics head with `K=5` components and NHWC memory format for better GPU throughput:

```bash
python train_level0.py \
  --steps 3000 \
  --batch_size 2 \
  --mix_k 5 \
  --amp \
  --channels_last \
  --save runs/level0_head_mixK5.pt
```

Key options include the cosine learning-rate schedule, gradient clipping, EMA logging, and resume support via `--resume` checkpoints.

### Level-1 head (`x_1 \mid y_2`)

Level-1 training mirrors the level-0 pipeline but uses a linear warm-up schedule and allows configuring weight decay:

```bash
python train_level1.py \
  --steps 3000 \
  --batch_size 8 \
  --amp \
  --save runs/level1_head_res.pt
```

Enable the mixture head in the same way with `--mix_k 5`. Training logs the exponential moving average of the loss and periodically persists checkpoints defined by `--ckpt_every`.

## Evaluation workflow

### Single-image inspection

Use `eval_delta01.py` to analyse one image, automatically resizing the pyramid and aligning per-head scales:

```bash
python eval_delta01.py \
  --img path/to/sample.jpg \
  --ckpt1 runs/level1_head_mixK5_v21.pt \
  --scale1 uint8 \
  --ckpt0 runs/level0_head_mixK5_h96b4.pt \
  --scale0 unit \
  --save_maps
```

The script prints D-scores, Δ01, and |Δ01|, and when `--save_maps` is provided, writes diagnostic PNGs (`D1_native`, `D0_native`, `Delta01_vis`). Tune `--max_side` to limit the processing resolution and `--pi_temp*` to temperature-adjust mixture logits.

### Batch scoring

Score an image directory (or list file) and export per-image statistics as CSV for downstream analysis:

```bash
python eval_batch_scores.py \
  --dir data/fake \
  --ckpt1 runs/level1_head_mixK5_v21.pt --scale1 uint8 \
  --ckpt0 runs/level0_head_mixK5_h128b6_v21.pt --scale0 uint8 \
  --csv runs/fake_scores_u8.csv \
  --label 0
```

The evaluator summarises valid rows, reports |Δ01| distribution statistics, and records any per-image errors alongside the failing paths. Supported `--scale*` presets are `uint8`, `unit`, and `tanh`, matching the expected input scaling of each head.

### Threshold tuning

Given the CSV outputs for real and fake images, combine them and call `choose_threshold.py` to compute ROC/AUC metrics and the balanced-accuracy optimum:

```bash
python choose_threshold.py \
  --csv runs/scores_mixK5_T13_P90.csv \
  --feature AbsDelta01_P90 \
  --pos_label 0 \
  --higher_is_better
```

The script orients feature scores, builds the ROC curve, calculates AUC with the trapezoidal rule, and reports the confusion matrix at the best balanced-accuracy threshold.

## Implementation notes

- Image loading utilities normalise palettes/transparency to RGB, optionally resize the longest side, and crop to multiples of eight to keep pyramid construction aligned.
- The pyramid builder produces paired uint8 `x_l` and float `y_{l+1}` tensors entirely on CPU to minimise GPU memory pressure during data loading.
- Logistic and mixture likelihood helpers operate in pixel units (0–255) and expose entropy estimates required for Δ01 scoring.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
