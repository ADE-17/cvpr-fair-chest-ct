# CVPR Fair Disease Diagnosis (Competition Code)

This folder contains a cleaned version of the code used for the chest CT fair diagnosis competition.

## Overview

- **Task**: 4-way volume-level classification (Adenocarcinoma, Squamous Cell Carcinoma, Covid-19, Healthy) with a fairness-aware metric (average of per-gender macro F1).
- **Model**: 2D ConvNeXt backbone + attention-based Multiple Instance Learning (MIL) with an optional adversarial gender head (Gradient Reversal Layer).
- **Training**: stratified k-fold by (class × gender), focal loss, subgroup oversampling, gradient accumulation, two-stage freeze/unfreeze schedule.
- **Post-processing**: soft-ensemble across folds with flip TTA, and optional per-class threshold optimization (validation or out-of-fold).

## Contents

- `config.py` – paths, hyperparameters, and training/inference switches.
- `dataset.py` – slice-level and scan-level datasets, including MIL batching.
- `models.py` – ConvNeXt slice classifier and ScanLevelMIL + GRL gender head.
- `losses.py` – focal loss implementation.
- `metrics.py` – per-gender macro F1, per-class F1, confusion matrix.
- `train.py` – k-fold training loop with MIL, GRL, and fairness-aware sampling.
- `inference.py` – validation/in-house evaluation with optional TTA.
- `eval_val.py` – fold-wise validation metrics, plots, and threshold optimization (including OOF thresholds).
- `test_infer/test_infer.py` – final test-time inference script that writes `A.csv`, `G.csv`, `covid.csv`, `normal.csv` for submission.
- `requirements.txt` – Python dependencies.

## Basic usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train all folds:

```bash
python train.py
```

Evaluate folds and run threshold optimization:

```bash
python eval_val.py --optimize_thresholds
python eval_val.py --oof_optimize_thresholds
```

Run ensemble + TTA inference on test set (edit `TEST_ROOT` in `test_infer/test_infer.py` to point to your test folder):

```bash
python test_infer/test_infer.py \
  --thresholds_path outputs/convnext_tiny_grl/checkpoints/val_eval/thresholds_oof.json
```

