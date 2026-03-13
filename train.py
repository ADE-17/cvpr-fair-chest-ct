"""
Training script for CVPR Fair Disease Diagnosis.
Stratified k-fold, focal loss, volume-level validation, WandB logging.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DATA_ROOT,
    CHECKPOINT_DIR,
    CLASSES,
    N_CLASSES,
    MODEL_NAME,
    IMG_SIZE,
    BATCH_SIZE,
    MIL_BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    LR_BACKBONE,
    WEIGHT_DECAY,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    LABEL_SMOOTHING,
    ACCUM_STEPS,
    FREEZE_EPOCHS,
    N_FOLDS,
    RANDOM_SEED,
    AGGREGATION,
    TRAIN_SLICES_PER_SCAN,
    USE_MIL,
    MAX_SLICES_PER_SCAN,
    MINORITY_BOOST_WEIGHT,
    SCC_CLASS_IDX,
    FEMALE_GENDER_IDX,
    USE_GENDER_ADV,
    GRL_LAMBDA,
)
from dataset import (
    build_scan_records,
    build_slice_records,
    SliceDataset,
    ScanLevelDataset,
    get_train_transforms,
    get_val_transforms,
    collate_with_scan_id,
    collate_scan_batch,
)
from models import SliceClassifier, ScanLevelMIL
from losses import FocalLoss
from metrics import per_gender_macro_f1, full_metrics_report

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_scan_level_df(scan_records):
    """DataFrame for stratification: one row per scan."""
    import pandas as pd
    rows = []
    for r in scan_records:
        rows.append({
            "scan_id": r["scan_id"],
            "class": r["class"],
            "label": r["label"],
            "gender": r["gender"],
        })
    return pd.DataFrame(rows)


def stratified_fold_indices(scan_records, n_folds=5, seed=42):
    """
    Stratify by (class, gender). Return list of (train_scan_ids, val_scan_ids) per fold.
    """
    import pandas as pd
    df = get_scan_level_df(scan_records)
    df["stratify_key"] = df["class"].astype(str) + "_" + df["gender"].astype(str)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scan_ids = df["scan_id"].tolist()
    stratify = df["stratify_key"].tolist()

    folds = []
    for train_idx, val_idx in skf.split(scan_ids, stratify):
        train_ids = [scan_ids[i] for i in train_idx]
        val_ids = [scan_ids[i] for i in val_idx]
        folds.append((train_ids, val_ids))
    return folds


def slice_indices_by_scan_id(slice_records, scan_ids_set):
    """Indices of slice_records whose scan_id is in scan_ids_set."""
    return [i for i, r in enumerate(slice_records) if r["scan_id"] in scan_ids_set]


def get_fairness_weights(records, minority_boost=MINORITY_BOOST_WEIGHT):
    """
    Weights for WeightedRandomSampler: massive weight for Female SCC so they appear in almost every batch.
    records: list of dicts with 'label' and 'gender_idx' (slice_records or scan_records).
    """
    weights = []
    for r in records:
        if r["label"] == SCC_CLASS_IDX and r["gender_idx"] == FEMALE_GENDER_IDX:
            weights.append(float(minority_boost))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.double)


def aggregate_logits(logits, scan_ids, strategy="mean"):
    """Aggregate slice logits per scan. Returns {scan_id: (agg_logits, label, gender_idx)}."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for i, sid in enumerate(scan_ids):
        grouped[sid].append(i)

    result = {}
    for sid, indices in grouped.items():
        L = torch.stack([logits[i] for i in indices])
        if strategy == "mean":
            agg = L.mean(dim=0)
        elif strategy == "max":
            agg = L.max(dim=0)[0]
        else:
            agg = L.mean(dim=0)
        result[sid] = agg
    return result


def evaluate_volume_level(model, loader, device, aggregation="mean"):
    """
    Evaluate at volume level: aggregate slice logits per scan, then compute metrics.
    """
    model.eval()
    scan_preds = {}
    scan_labels = {}
    scan_genders = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"]
            gender_idxs = batch["gender_idx"]
            scan_ids = batch["scan_id"]

            logits = model(images)
            agg_dict = aggregate_logits(logits.cpu(), scan_ids, aggregation)

            for sid, agg_logits in agg_dict.items():
                pred = int(torch.argmax(agg_logits).item())
                # Get label/gender from first occurrence
                idx = next(i for i, s in enumerate(scan_ids) if s == sid)
                scan_preds[sid] = pred
                scan_labels[sid] = int(labels[idx].item())
                scan_genders[sid] = int(gender_idxs[idx].item())

    y_true = [scan_labels[sid] for sid in sorted(scan_preds.keys())]
    y_pred = [scan_preds[sid] for sid in sorted(scan_preds.keys())]
    genders = [scan_genders[sid] for sid in sorted(scan_preds.keys())]

    return per_gender_macro_f1(y_true, y_pred, genders), full_metrics_report(
        y_true, y_pred, genders, CLASSES
    )


def train_one_epoch(model, loader, criterion, optimizer, device, use_mil=False, accum_steps=1, gender_criterion=None):
    model.train()
    total_loss = 0.0
    n = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        if use_mil:
            images = batch["images"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)
            if USE_GENDER_ADV and gender_criterion is not None:
                gender_labels = batch["gender_idx"].to(device)
                logits, gender_logits = model(images, mask, return_gender=True)
            else:
                logits = model(images, mask)
        else:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
        loss = criterion(logits, labels)
        if use_mil and USE_GENDER_ADV and gender_criterion is not None:
            gender_loss = gender_criterion(gender_logits, gender_labels)
            loss = loss + GRL_LAMBDA * gender_loss
        loss = loss / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        n += 1
    return total_loss / n if n else 0.0


def evaluate_volume_level_mil(model, loader, device):
    """Evaluate MIL model at scan level (one prediction per scan already)."""
    model.eval()
    all_preds, all_labels, all_genders = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            images = batch["images"].to(device)
            mask = batch["mask"].to(device)
            logits = model(images, mask)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())
            all_genders.extend(batch["gender_idx"].numpy())
    pg = per_gender_macro_f1(all_labels, all_preds, all_genders)
    full = full_metrics_report(all_labels, all_preds, all_genders, CLASSES)
    return pg, full


def main(args):
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load scan records (train only for k-fold)
    train_scan_records = build_scan_records(str(DATA_ROOT), split="train")
    val_scan_records = build_scan_records(str(DATA_ROOT), split="val")

    # Build slice records (use all for now; sampling happens in Dataset via build_slice_records)
    def make_slice_records(scan_records, slices_per_scan):
        return build_slice_records(scan_records, slices_per_scan=slices_per_scan)

    # Stratified k-fold on train scans
    folds = stratified_fold_indices(train_scan_records, n_folds=N_FOLDS, seed=RANDOM_SEED)

    for fold, (train_scan_ids, val_scan_ids) in enumerate(folds):
        if args.fold is not None and fold != args.fold:
            continue

        train_ids_set = set(train_scan_ids)
        val_ids_set = set(val_scan_ids)
        train_scans = [r for r in train_scan_records if r["scan_id"] in train_ids_set]
        val_scans = [r for r in train_scan_records if r["scan_id"] in val_ids_set]

        if USE_MIL:
            train_ds = ScanLevelDataset(
                train_scans,
                transform=get_train_transforms(IMG_SIZE),
                max_slices=MAX_SLICES_PER_SCAN,
                is_training=True,
            )
            val_ds = ScanLevelDataset(
                val_scans,
                transform=get_val_transforms(IMG_SIZE),
                max_slices=MAX_SLICES_PER_SCAN,
                is_training=False,
            )
            train_weights = get_fairness_weights(train_scans)
            sampler = WeightedRandomSampler(
                train_weights,
                num_samples=len(train_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=MIL_BATCH_SIZE,
                sampler=sampler,
                num_workers=NUM_WORKERS,
                collate_fn=collate_scan_batch,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=MIL_BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                collate_fn=collate_scan_batch,
            )
            model = ScanLevelMIL(num_classes=N_CLASSES, model_name=MODEL_NAME, pretrained=True)
        else:
            train_slice_records = make_slice_records(train_scans, TRAIN_SLICES_PER_SCAN)
            val_slice_records = make_slice_records(val_scans, None)
            train_ds = SliceDataset(
                train_slice_records,
                transform=get_train_transforms(IMG_SIZE),
                is_training=True,
            )
            val_ds = SliceDataset(
                val_slice_records,
                transform=get_val_transforms(IMG_SIZE),
                is_training=False,
            )
            train_weights = get_fairness_weights(train_slice_records)
            sampler = WeightedRandomSampler(
                train_weights,
                num_samples=len(train_weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=NUM_WORKERS,
                collate_fn=collate_with_scan_id,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                collate_fn=collate_with_scan_id,
            )
            model = SliceClassifier(num_classes=N_CLASSES, model_name=MODEL_NAME, pretrained=True)

        model = model.to(device)
        criterion = FocalLoss(
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            label_smoothing=LABEL_SMOOTHING,
        )
        gender_criterion = nn.CrossEntropyLoss() if USE_GENDER_ADV and USE_MIL else None
        # Phase 1 (epochs 1–FREEZE_EPOCHS): freeze backbone for MIL, train attention + classifier only
        if USE_MIL:
            for p in model.backbone.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        run_name = f"fold{fold}" + ("_mil_new" if USE_MIL else "")
        if HAS_WANDB:
            wandb.init(
                project=args.wandb_project or "cvpr-fair-diagnosis",
                name=run_name,
                config={
                    "fold": fold,
                    "model": MODEL_NAME,
                    "use_mil": USE_MIL,
                    "epochs": EPOCHS,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "focal_gamma": FOCAL_GAMMA,
                    "aggregation": AGGREGATION,
                    "minority_boost": MINORITY_BOOST_WEIGHT,
                },
            )

        best_score = 0.0
        for epoch in range(EPOCHS):
            # At epoch == FREEZE_EPOCHS, unfreeze backbone for MIL and lower its LR
            if USE_MIL and epoch == FREEZE_EPOCHS:
                for p in model.backbone.parameters():
                    p.requires_grad = True
                # New optimizer with separate LR for backbone vs attention/classifier head
                optimizer = torch.optim.AdamW(
                    [
                        {"params": model.backbone.parameters(), "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
                        {"params": model.mil.parameters(), "lr": LR, "weight_decay": WEIGHT_DECAY},
                    ]
                )
                # New scheduler for remaining epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS - FREEZE_EPOCHS
                )

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                use_mil=USE_MIL,
                accum_steps=ACCUM_STEPS,
                gender_criterion=gender_criterion,
            )
            scheduler.step()
            if USE_MIL:
                pg, full = evaluate_volume_level_mil(model, val_loader, device)
            else:
                pg, full = evaluate_volume_level(model, val_loader, device, AGGREGATION)

            # Use validation macro F1 (across all classes) as checkpoint selection metric
            score = full["macro_f1"]
            if score > best_score:
                best_score = score
                ckpt_path = CHECKPOINT_DIR / f"best_fold{fold}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "score": score,
                        "fold": fold,
                        "use_mil": USE_MIL,
                    },
                    ckpt_path,
                )

            log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_overall_f1": score,
                "val_male_macro_f1": pg["male_macro_f1"],
                "val_female_macro_f1": pg["female_macro_f1"],
                "val_macro_f1": full["macro_f1"],
            }
            for c in CLASSES:
                log[f"val_f1_{c}"] = full.get(f"f1_{c}", 0)
            if HAS_WANDB:
                wandb.log(log, step=epoch)
            print(
                f"Fold {fold} Epoch {epoch} | loss={train_loss:.4f} | "
                f"overall_f1={score:.4f} | male={pg['male_macro_f1']:.4f} | "
                f"female={pg['female_macro_f1']:.4f}"
            )

        if HAS_WANDB:
            wandb.finish()

    print("Training complete. Best checkpoints saved to", CHECKPOINT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Train only this fold (0 to N_FOLDS-1)")
    parser.add_argument("--wandb_project", type=str, default="cvpr-fair-diagnosis")
    args = parser.parse_args()
    main(args)
