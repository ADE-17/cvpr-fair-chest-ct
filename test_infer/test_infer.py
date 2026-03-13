"""
Run inference on the test set and write A.csv, covid.csv, G.csv, normal.csv.

Supports ensembling across all folds by averaging per-scan logits from multiple checkpoints.

Assumes test structure like:
    TEST_ROOT/ct_scan_xx/*.jpg
or
    TEST_ROOT/**/ct_scan_xx/*.jpg

If your test is nested differently, just update `build_test_scan_records`.

python /home/adipa/cvpr_fair/test_infer/test_infer.py \
  --checkpoints /scratch/adipa/cvpr_hack/outputs/checkpoints/best_fold2.pt \
  /scratch/adipa/cvpr_hack/outputs/checkpoints/best_fold4.pt

"""

import os
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append("/home/adipa/cvpr_fair")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F
import json

from config import (
    CLASSES,
    N_CLASSES,
    IMG_SIZE,
    BATCH_SIZE,
    MIL_BATCH_SIZE,
    NUM_WORKERS,
    CHECKPOINT_DIR,
    MAX_SLICES_PER_SCAN,
    USE_MIL,
)

from dataset import (
    ScanLevelDataset,
    SliceDataset,
    collate_scan_batch,
    collate_with_scan_id,
    get_val_transforms,
    build_slice_records,
)
from models import ScanLevelMIL, SliceClassifier

# --------------------------------------------------------------------------
# CONFIG – CHANGE THIS
# --------------------------------------------------------------------------
TEST_ROOT = Path("/scratch/adipa/cvpr_hack/test_for_participants")  # <-- set to your test folder
OUTPUT_DIR = Path("/scratch/adipa/cvpr_hack/test_predictions")  # where A.csv etc. go
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Build scan list for TEST (no labels / no gender)
# --------------------------------------------------------------------------
def build_test_scan_records(root_dir: Path):
    """
    One record per CT scan folder in TEST_ROOT.
    We don't know labels or gender for test, so we fill dummy values.
    """
    records = []
    root_dir = Path(root_dir)

    # Heuristic: any directory named ct_scan_* is treated as a scan
    for scan_dir in sorted(root_dir.rglob("ct_scan_*")):
        if not scan_dir.is_dir():
            continue
        slices = sorted([f for f in scan_dir.glob("*.jpg") if not f.name.startswith("._")])
        if not slices:
            slices = sorted([f for f in scan_dir.glob("*.png") if not f.name.startswith("._")])
        if not slices:
            continue

        records.append(
            {
                "scan_path": str(scan_dir),
                "scan_id": scan_dir.name,  # use folder name for CSV
                "class": None,
                "label": 0,          # dummy
                "gender": None,
                "gender_idx": 0,     # dummy
                "slice_paths": [str(p) for p in slices],
                "split": "test",
            }
        )
    return records


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------
def make_hflip_transform(base_transform):
    """Return a transform that applies a horizontal flip before the base transform."""

    def _tf(img):
        img = F.hflip(img)
        return base_transform(img)

    return _tf


def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def apply_class_thresholds_from_logits(logits, thresholds, class_names=CLASSES):
    """
    logits: (C,)
    thresholds: dict class_name->threshold on probabilities (after softmax)
    rule: choose among classes passing threshold the one with highest prob else argmax
    """
    if thresholds is None:
        return int(np.argmax(logits))
    if "thresholds" in thresholds and isinstance(thresholds["thresholds"], dict):
        thresholds = thresholds["thresholds"]
    probs = softmax_np(np.asarray(logits)[None, :], axis=1)[0]
    thr = np.array([float(thresholds.get(c, 0.0)) for c in class_names], dtype=np.float32)
    passed = np.where(probs >= thr)[0]
    if len(passed) == 0:
        return int(np.argmax(probs))
    best = passed[np.argmax(probs[passed])]
    return int(best)


def predict_mil_logits(model, loader, device):
    """
    MIL model: one logit vector per scan already (batch['scan_id'] per item).
    Returns: dict scan_id -> np.ndarray logits shape (C,)
    """
    model.eval()
    logits_by_scan = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test inference (MIL)"):
            images = batch["images"].to(device)
            mask = batch["mask"].to(device)
            logits = model(images, mask)
            logits = logits.cpu().numpy()
            for sid, l in zip(batch["scan_id"], logits):
                logits_by_scan[sid] = l
    return logits_by_scan


def predict_slice_level_logits_max(model, loader, device):
    """
    Slice-level model: aggregate slice logits per scan with max (AGGREGATION='max').
    Returns: dict scan_id -> np.ndarray logits shape (C,)
    """
    model.eval()
    from collections import defaultdict

    scan_logits = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test inference (slice-level)"):
            images = batch["image"].to(device)
            scan_ids = batch["scan_id"]
            logits = model(images).cpu().numpy()
            for i, sid in enumerate(scan_ids):
                scan_logits[sid].append(logits[i])

    logits_by_scan = {}
    for sid, log_list in scan_logits.items():
        logits_by_scan[sid] = np.max(log_list, axis=0)  # max aggregation
    return logits_by_scan


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def list_default_fold_checkpoints():
    # best_fold0.pt ... best_fold4.pt
    ckpts = sorted(CHECKPOINT_DIR.glob("best_fold*.pt"))
    return ckpts


def main(checkpoints=None, thresholds_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build test scan records
    scan_records = build_test_scan_records(TEST_ROOT)
    print(f"Found {len(scan_records)} test scans under {TEST_ROOT}")

    # 2) Resolve checkpoints for ensembling
    if checkpoints is None or len(checkpoints) == 0:
        ckpt_paths = list_default_fold_checkpoints()
    else:
        ckpt_paths = [Path(p) for p in checkpoints]
    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}.")

    # decide MIL vs slice-level from first checkpoint (must match across folds)
    first_ckpt = torch.load(ckpt_paths[0], map_location=device)
    use_mil = first_ckpt.get("use_mil", USE_MIL)

    val_transform = get_val_transforms(IMG_SIZE)
    flip_transform = make_hflip_transform(get_val_transforms(IMG_SIZE))

    # 3) Build loaders once (reused across fold models), for original and flipped scans
    if use_mil:
        dataset_orig = ScanLevelDataset(
            scan_records,
            transform=val_transform,
            max_slices=MAX_SLICES_PER_SCAN,
            is_training=False,
        )
        loader_orig = DataLoader(
            dataset_orig,
            batch_size=MIL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_scan_batch,
        )
        dataset_flip = ScanLevelDataset(
            scan_records,
            transform=flip_transform,
            max_slices=MAX_SLICES_PER_SCAN,
            is_training=False,
        )
        loader_flip = DataLoader(
            dataset_flip,
            batch_size=MIL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_scan_batch,
        )
    else:
        # Slice-level: flatten to slices, cap slices per scan
        slice_records = build_slice_records(
            scan_records, slices_per_scan=None, max_slices=MAX_SLICES_PER_SCAN
        )
        dataset_orig = SliceDataset(slice_records, transform=val_transform, is_training=False)
        loader_orig = DataLoader(
            dataset_orig,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_with_scan_id,
        )
        dataset_flip = SliceDataset(slice_records, transform=flip_transform, is_training=False)
        loader_flip = DataLoader(
            dataset_flip,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_with_scan_id,
        )

    # 4) Ensemble across folds and TTA views by averaging logits
    logits_sum = {}
    n_models_effective = 0

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt.get("use_mil", USE_MIL) != use_mil:
            raise ValueError("Mixed MIL/non-MIL checkpoints. Ensemble requires consistent model type.")

        if use_mil:
            model = ScanLevelMIL(num_classes=N_CLASSES, model_name=ckpt.get("model_name", "convnext_tiny"), pretrained=False)
            model.load_state_dict(ckpt["model"])
            model = model.to(device)
            logits_orig = predict_mil_logits(model, loader_orig, device)
            logits_flip = predict_mil_logits(model, loader_flip, device)
        else:
            model = SliceClassifier(num_classes=N_CLASSES, model_name=ckpt.get("model_name", "convnext_tiny"), pretrained=False)
            model.load_state_dict(ckpt["model"])
            model = model.to(device)
            logits_orig = predict_slice_level_logits_max(model, loader_orig, device)
            logits_flip = predict_slice_level_logits_max(model, loader_flip, device)

        # combine original + flipped for this model
        for sid in logits_orig.keys():
            l = logits_orig[sid] + logits_flip[sid]
            if sid not in logits_sum:
                logits_sum[sid] = l.astype(np.float64)
            else:
                logits_sum[sid] += l.astype(np.float64)
        n_models_effective += 2  # original + flipped

    thresholds = None
    if thresholds_path:
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)

    preds_by_scan = {
        sid: apply_class_thresholds_from_logits(logits_sum[sid] / n_models_effective, thresholds, CLASSES)
        for sid in logits_sum.keys()
    }

    # 3) Group by predicted class and write CSVs
    scans_by_class = defaultdict(list)
    # map scan_id back to folder path so we can get folder name
    path_by_scan_id = {rec["scan_id"]: rec["scan_path"] for rec in scan_records}

    for sid, cls_idx in preds_by_scan.items():
        cls_name = CLASSES[cls_idx]  # "A", "G", "normal", "covid"
        folder_name = Path(path_by_scan_id[sid]).name
        scans_by_class[cls_name].append(folder_name)

    # Ensure we always create the 4 required csv files
    name_map = {
        "A": "A.csv",
        "covid": "covid.csv",
        "G": "G.csv",
        "normal": "normal.csv",
    }

    for cls, csv_name in name_map.items():
        rows = scans_by_class.get(cls, [])
        out_path = OUTPUT_DIR / csv_name
        with out_path.open("w") as f:
            for scan_name in rows:
                f.write(f"{scan_name}\n")
        print(f"Wrote {len(rows)} predictions to {out_path}")

    print("Done. Submit the four CSVs in", OUTPUT_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of checkpoint paths. If omitted, uses all best_fold*.pt in CHECKPOINT_DIR.",
    )
    parser.add_argument(
        "--thresholds_path",
        type=str,
        default=None,
        help="Optional thresholds JSON from eval_val.py (e.g., val_eval/thresholds_ensemble.json).",
    )
    args = parser.parse_args()
    main(args.checkpoints, thresholds_path=args.thresholds_path)