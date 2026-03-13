"""
Evaluate fold checkpoints on the validation split and save detailed metrics + plots.

Usage:
  python /home/adipa/cvpr_fair/eval_val.py
  python /home/adipa/cvpr_fair/eval_val.py --checkpoints /scratch/adipa/cvpr_hack/outputs/checkpoints/best_fold0.pt ...

Outputs (under CHECKPOINT_DIR/val_eval):
  - metrics.csv with per-fold metrics (macro F1, per-gender macro F1, per-class F1)
  - confusion_matrix_foldX.png heatmaps
  - f1_per_class_bar.png comparing per-class F1 across folds
  - summary.txt with mean/std across folds

python eval_val.py --optimize_thresholds

python /home/adipa/cvpr_fair/eval_val.py \
  --use_thresholds \
  --thresholds_path /scratch/adipa/cvpr_hack/outputs/convnext_tiny_grl/checkpoints/val_eval/thresholds_ensemble.json

python /home/adipa/cvpr_fair/eval_val.py --oof_optimize_thresholds

python /home/adipa/cvpr_fair/test_infer/test_infer.py \
  --thresholds_path /scratch/adipa/cvpr_hack/outputs/convnext_tiny_grl/checkpoints/val_eval/thresholds_oof.json
  
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from config import (
    DATA_ROOT,
    CHECKPOINT_DIR,
    CLASSES,
    N_CLASSES,
    IMG_SIZE,
    BATCH_SIZE,
    MIL_BATCH_SIZE,
    NUM_WORKERS,
    AGGREGATION,
    MAX_SLICES_PER_SCAN,
    USE_MIL,
)
from dataset import (
    build_scan_records,
    build_slice_records,
    SliceDataset,
    ScanLevelDataset,
    collate_with_scan_id,
    collate_scan_batch,
    get_val_transforms,
)
from models import SliceClassifier, ScanLevelMIL
from metrics import per_gender_macro_f1, full_metrics_report


def list_default_fold_checkpoints():
    return sorted(Path(CHECKPOINT_DIR).glob("best_fold*.pt"))

def parse_fold_idx(path: Path):
    # expects names like best_fold0.pt
    name = path.stem
    if "fold" not in name:
        return None
    try:
        return int(name.split("fold")[-1].split("_")[0])
    except Exception:
        return None


def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def apply_class_thresholds(probs, thresholds, class_names=CLASSES):
    """
    Convert probabilities (N,C) to single predicted class index using per-class thresholds.
    Rule: select among classes with p_c >= t_c the one with largest p_c. If none pass, argmax.
    thresholds: dict class_name -> float
    """
    # allow either {"thresholds": {...}} or flat dict
    if "thresholds" in thresholds and isinstance(thresholds["thresholds"], dict):
        thresholds = thresholds["thresholds"]
    thr = np.array([float(thresholds.get(c, 0.0)) for c in class_names], dtype=np.float32)
    preds = []
    for p in probs:
        passed = np.where(p >= thr)[0]
        if len(passed) == 0:
            preds.append(int(np.argmax(p)))
        else:
            # choose best among passed
            best = passed[np.argmax(p[passed])]
            preds.append(int(best))
    return np.array(preds, dtype=int)


def optimize_thresholds_one_vs_rest(probs, y_true, class_names=CLASSES, grid=None):
    """
    Optimize per-class thresholds independently to maximize one-vs-rest F1 on validation.
    Returns: dict class_name -> threshold, and dict class_name -> best_f1.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)  # step=0.01
    y_true = np.asarray(y_true)
    thresholds = {}
    best_f1s = {}
    for c_idx, c_name in enumerate(class_names):
        y_bin = (y_true == c_idx).astype(int)
        p = probs[:, c_idx]
        best_t, best = 0.5, -1.0
        for t in grid:
            y_hat = (p >= t).astype(int)
            f1 = f1_score(y_bin, y_hat, zero_division=0)
            if f1 > best:
                best = f1
                best_t = float(t)
        thresholds[c_name] = best_t
        best_f1s[c_name] = float(best)
    return thresholds, best_f1s


def stratified_fold_scan_ids(train_scan_records, n_folds=5, seed=42):
    """
    Recreate the same stratified k-fold split used in training.
    Returns list of (train_scan_ids, val_scan_ids) for each fold.
    Stratify by (class, gender).
    """
    import pandas as pd

    rows = []
    for r in train_scan_records:
        rows.append(
            {
                "scan_id": r["scan_id"],
                "class": r["class"],
                "gender": r["gender"],
            }
        )
    df = pd.DataFrame(rows)
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


def run_mil_val(model, loader, device):
    model.eval()
    all_logits, all_labels, all_genders = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val (MIL)", leave=False):
            images = batch["images"].to(device)
            mask = batch["mask"].to(device)
            logits = model(images, mask).cpu().numpy()
            all_logits.append(logits)
            all_labels.extend(batch["label"].numpy())
            all_genders.extend(batch["gender_idx"].numpy())
    logits = np.concatenate(all_logits, axis=0)
    return np.array(all_labels), np.array(all_genders), logits


def run_slice_val_max(model, loader, device):
    model.eval()
    scan_logits = defaultdict(list)
    scan_labels = {}
    scan_genders = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val (slice)", leave=False):
            images = batch["image"].to(device)
            scan_ids = batch["scan_id"]
            labels = batch["label"]
            genders = batch["gender_idx"]
            logits = model(images).cpu().numpy()
            for i, sid in enumerate(scan_ids):
                scan_logits[sid].append(logits[i])
                scan_labels[sid] = int(labels[i].item())
                scan_genders[sid] = int(genders[i].item())

    scan_logits_agg = {}
    for sid, log_list in scan_logits.items():
        if AGGREGATION == "max":
            agg = np.max(log_list, axis=0)
        else:
            agg = np.mean(log_list, axis=0)
        scan_logits_agg[sid] = agg

    keys = sorted(scan_logits_agg.keys())
    y_true = np.array([scan_labels[k] for k in keys])
    genders = np.array([scan_genders[k] for k in keys])
    logits = np.stack([scan_logits_agg[k] for k in keys], axis=0)
    return y_true, genders, logits


def main(
    checkpoints=None,
    optimize_thresholds=False,
    use_thresholds=False,
    thresholds_path=None,
    oof_optimize_thresholds=False,
    n_folds=5,
    seed=42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = list_default_fold_checkpoints() if not checkpoints else [Path(p) for p in checkpoints]
    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}.")

    out_dir = Path(CHECKPOINT_DIR) / "val_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validation data comes from provided val split
    scan_records = build_scan_records(str(DATA_ROOT), split="val")
    train_scan_records = build_scan_records(str(DATA_ROOT), split="train")
    val_transform = get_val_transforms(IMG_SIZE)

    # Pre-build loaders for each mode
    mil_loader = None
    slice_loader = None

    results = []
    # For optional ensemble threshold optimization
    ensemble_logits_sum = None
    ensemble_count = 0
    ensemble_y_true = None
    ensemble_genders = None

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device)
        use_mil = ckpt.get("use_mil", USE_MIL)

        if use_mil:
            if mil_loader is None:
                ds = ScanLevelDataset(
                    scan_records,
                    transform=val_transform,
                    max_slices=MAX_SLICES_PER_SCAN,
                    is_training=False,
                )
                mil_loader = DataLoader(
                    ds,
                    batch_size=MIL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    collate_fn=collate_scan_batch,
                )
            model = ScanLevelMIL(num_classes=N_CLASSES, model_name=ckpt.get("model_name", "convnext_tiny"), pretrained=False)
            model.load_state_dict(ckpt["model"])
            model = model.to(device)
            y_true, genders, logits = run_mil_val(model, mil_loader, device)
        else:
            if slice_loader is None:
                slice_records = build_slice_records(scan_records, slices_per_scan=None, max_slices=MAX_SLICES_PER_SCAN)
                ds = SliceDataset(slice_records, transform=val_transform, is_training=False)
                slice_loader = DataLoader(
                    ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    collate_fn=collate_with_scan_id,
                )
            model = SliceClassifier(num_classes=N_CLASSES, model_name=ckpt.get("model_name", "convnext_tiny"), pretrained=False)
            model.load_state_dict(ckpt["model"])
            model = model.to(device)
            y_true, genders, logits = run_slice_val_max(model, slice_loader, device)

        probs = softmax_np(logits, axis=1)

        # Optional: apply loaded thresholds for evaluation
        applied_thresholds = None
        if use_thresholds and thresholds_path:
            with open(thresholds_path, "r") as f:
                applied_thresholds = json.load(f)
            y_pred = apply_class_thresholds(probs, applied_thresholds, CLASSES)
        else:
            y_pred = probs.argmax(axis=1)

        # Optional: optimize thresholds for this checkpoint and save
        thr_file = None
        thr_scores = None
        if optimize_thresholds:
            thresholds, best_f1s = optimize_thresholds_one_vs_rest(probs, y_true, CLASSES)
            thr_file = out_dir / f"thresholds_{ckpt_path.stem}.json"
            with open(thr_file, "w") as f:
                json.dump(
                    {"thresholds": thresholds, "best_one_vs_rest_f1": best_f1s},
                    f,
                    indent=2,
                )
            thr_scores = best_f1s

        # Update ensemble accumulators (for optional ensemble threshold optimization)
        if ensemble_logits_sum is None:
            ensemble_logits_sum = logits.astype(np.float64)
            ensemble_y_true = y_true
            ensemble_genders = genders
        else:
            ensemble_logits_sum += logits.astype(np.float64)
        ensemble_count += 1

        pg = per_gender_macro_f1(y_true, y_pred, genders)
        full = full_metrics_report(y_true, y_pred, genders, CLASSES)

        # Per-class F1s from full_metrics_report (keys: f1_A, f1_G, ...)
        row = {
            "ckpt": ckpt_path.name,
            "macro_f1": full["macro_f1"],
            "male_macro_f1": pg["male_macro_f1"],
            "female_macro_f1": pg["female_macro_f1"],
            "competition_overall_f1": pg["overall_f1"],
        }
        for c in CLASSES:
            row[f"f1_{c}"] = full.get(f"f1_{c}", 0.0)
        if thr_file is not None:
            row["thresholds_file"] = str(thr_file)
        results.append(row)
        print(row)

        # Save confusion matrix heatmap for this fold
        cm = full["confusion_matrix"]
        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion matrix - {ckpt_path.name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_matrix_{ckpt_path.stem}.png", dpi=200)
        plt.close()

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "metrics.csv", index=False)

    # If requested, optimize thresholds on ENSEMBLE logits and save a single file for test-time use
    if optimize_thresholds and ensemble_count > 0:
        ensemble_logits = ensemble_logits_sum / float(ensemble_count)
        ensemble_probs = softmax_np(ensemble_logits, axis=1)
        thresholds, best_f1s = optimize_thresholds_one_vs_rest(ensemble_probs, ensemble_y_true, CLASSES)
        ensemble_path = out_dir / "thresholds_ensemble.json"
        with open(ensemble_path, "w") as f:
            json.dump(
                {"thresholds": thresholds, "best_one_vs_rest_f1": best_f1s},
                f,
                indent=2,
            )
        print(f"\nSaved ensemble thresholds to: {ensemble_path}")

    # --- True OOF threshold optimization (recommended) ---
    # Build OOF predictions by evaluating fold-k checkpoint ONLY on the held-out fold-k scans.
    if oof_optimize_thresholds:
        folds = stratified_fold_scan_ids(train_scan_records, n_folds=n_folds, seed=seed)

        # Map scan_id -> record for quick subset selection
        rec_by_id = {r["scan_id"]: r for r in train_scan_records}

        oof_logits_list = []
        oof_y_true_list = []
        oof_gender_list = []

        for ckpt_path in ckpt_paths:
            fold_idx = parse_fold_idx(ckpt_path)
            if fold_idx is None or fold_idx < 0 or fold_idx >= len(folds):
                continue
            _, val_ids = folds[fold_idx]
            fold_scans = [rec_by_id[sid] for sid in val_ids if sid in rec_by_id]

            if len(fold_scans) == 0:
                continue

            ckpt = torch.load(ckpt_path, map_location=device)
            use_mil = ckpt.get("use_mil", USE_MIL)

            if use_mil:
                ds = ScanLevelDataset(
                    fold_scans,
                    transform=val_transform,
                    max_slices=MAX_SLICES_PER_SCAN,
                    is_training=False,
                )
                loader = DataLoader(
                    ds,
                    batch_size=MIL_BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    collate_fn=collate_scan_batch,
                )
                model = ScanLevelMIL(
                    num_classes=N_CLASSES,
                    model_name=ckpt.get("model_name", "convnext_tiny"),
                    pretrained=False,
                )
                model.load_state_dict(ckpt["model"])
                model = model.to(device)
                y_true, genders, logits = run_mil_val(model, loader, device)
            else:
                slice_records = build_slice_records(
                    fold_scans, slices_per_scan=None, max_slices=MAX_SLICES_PER_SCAN
                )
                ds = SliceDataset(slice_records, transform=val_transform, is_training=False)
                loader = DataLoader(
                    ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    collate_fn=collate_with_scan_id,
                )
                model = SliceClassifier(
                    num_classes=N_CLASSES,
                    model_name=ckpt.get("model_name", "convnext_tiny"),
                    pretrained=False,
                )
                model.load_state_dict(ckpt["model"])
                model = model.to(device)
                y_true, genders, logits = run_slice_val_max(model, loader, device)

            oof_logits_list.append(logits.astype(np.float64))
            oof_y_true_list.append(y_true.astype(int))
            oof_gender_list.append(genders.astype(int))

        if len(oof_logits_list) > 0:
            oof_logits = np.concatenate(oof_logits_list, axis=0)
            oof_y_true = np.concatenate(oof_y_true_list, axis=0)
            oof_genders = np.concatenate(oof_gender_list, axis=0)
            oof_probs = softmax_np(oof_logits, axis=1)

            thresholds, best_f1s = optimize_thresholds_one_vs_rest(oof_probs, oof_y_true, CLASSES)
            oof_path = out_dir / "thresholds_oof.json"
            with open(oof_path, "w") as f:
                json.dump(
                    {
                        "thresholds": thresholds,
                        "best_one_vs_rest_f1": best_f1s,
                        "note": "Optimized on OOF predictions from k-fold train split (no leakage from provided val set).",
                        "n_folds": n_folds,
                        "seed": seed,
                    },
                    f,
                    indent=2,
                )
            # Report OOF score with and without thresholds
            preds_argmax = oof_probs.argmax(axis=1)
            preds_thr = apply_class_thresholds(oof_probs, thresholds, CLASSES)
            pg_argmax = per_gender_macro_f1(oof_y_true, preds_argmax, oof_genders)
            pg_thr = per_gender_macro_f1(oof_y_true, preds_thr, oof_genders)
            full_argmax = full_metrics_report(oof_y_true, preds_argmax, oof_genders, CLASSES)
            full_thr = full_metrics_report(oof_y_true, preds_thr, oof_genders, CLASSES)
            print(f"\nSaved OOF thresholds to: {oof_path}")
            print(
                f"OOF argmax: macro_f1={full_argmax['macro_f1']:.4f} comp={pg_argmax['overall_f1']:.4f} "
                f"(male={pg_argmax['male_macro_f1']:.4f}, female={pg_argmax['female_macro_f1']:.4f})"
            )
            print(
                f"OOF thresholds: macro_f1={full_thr['macro_f1']:.4f} comp={pg_thr['overall_f1']:.4f} "
                f"(male={pg_thr['male_macro_f1']:.4f}, female={pg_thr['female_macro_f1']:.4f})"
            )
        else:
            print("\nOOF threshold optimization requested, but no fold checkpoints matched best_fold{k}.pt naming.")

    # Summary (mean/std) printed and saved
    def mean_std(key):
        vals = np.array([r[key] for r in results], dtype=float)
        return float(vals.mean()), float(vals.std())

    lines = ["=== Summary across folds ==="]
    for k in ["macro_f1", "competition_overall_f1", "male_macro_f1", "female_macro_f1"]:
        m, s = mean_std(k)
        line = f"{k}: mean={m:.4f} std={s:.4f}"
        print(line)
        lines.append(line)

    # Also summarize per-class F1 mean/std
    for c in CLASSES:
        key = f"f1_{c}"
        m, s = mean_std(key)
        line = f"{key}: mean={m:.4f} std={s:.4f}"
        print(line)
        lines.append(line)

    with open(out_dir / "summary.txt", "w") as f:
        f.write("\n".join(lines))

    # Plot per-class F1 across folds
    if len(results) > 1:
        plt.figure(figsize=(6, 4))
        for c in CLASSES:
            plt.plot(df["ckpt"], df[f"f1_{c}"], marker="o", label=c)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("F1 score")
        plt.title("Per-class F1 by fold")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "f1_per_class_by_fold.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*", default=None, help="Optional list of checkpoint paths.")
    p.add_argument("--optimize_thresholds", action="store_true", help="Grid-search best per-class thresholds on val and save JSON.")
    p.add_argument("--use_thresholds", action="store_true", help="Evaluate using thresholds loaded from --thresholds_path.")
    p.add_argument("--thresholds_path", type=str, default=None, help="Path to thresholds JSON (expects {'thresholds': {...}} or flat dict).")
    p.add_argument("--oof_optimize_thresholds", action="store_true", help="Optimize thresholds on out-of-fold (OOF) predictions from the k-fold train split and save thresholds_oof.json.")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(
        args.checkpoints,
        optimize_thresholds=args.optimize_thresholds,
        use_thresholds=args.use_thresholds,
        thresholds_path=args.thresholds_path,
        oof_optimize_thresholds=args.oof_optimize_thresholds,
        n_folds=args.n_folds,
        seed=args.seed,
    )

