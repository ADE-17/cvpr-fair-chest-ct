"""
Inference with TTA and volume aggregation.
Predict on original, flipped, and shifted slices; average logits; aggregate per volume.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    TTA_FLIP,
    TTA_SHIFTS,
    TTA_N_SHIFTS,
    INFERENCE_SLICES_PER_SCAN,
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


class TTATransform:
    """Apply TTA: optional flip and shifts. Returns list of (transformed_img,) for averaging."""

    def __init__(self, base_transform, flip=True, shifts=None, n_shifts=3):
        self.base = base_transform
        self.flip = flip
        self.shifts = shifts or [(0, 0)]
        self.n_shifts = min(n_shifts, len(self.shifts))

    def __call__(self, img):
        from PIL import Image
        imgs = []
        # Original
        for dx, dy in self.shifts[: self.n_shifts]:
            if dx == 0 and dy == 0:
                if self.flip:
                    imgs.append(self.base(img))
                    imgs.append(self.base(img.transpose(Image.FLIP_LEFT_RIGHT)))
                else:
                    imgs.append(self.base(img))
            else:
                w, h = img.size
                shifted = img.transform(
                    (w, h),
                    Image.AFFINE,
                    (1, 0, dx, 0, 1, dy),
                    fill=0,
                )
                imgs.append(self.base(shifted))
        return imgs


def get_tta_val_transforms(img_size=IMG_SIZE, flip=True, shifts=None, n_shifts=3):
    base = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return TTATransform(base, flip=flip, shifts=shifts, n_shifts=n_shifts)


# Custom dataset that returns multiple TTA views per slice
class TTASliceDataset(torch.utils.data.Dataset):
    """Returns (list of augmented images, label, gender_idx, scan_id) per slice."""

    def __init__(self, slice_records, tta_transform):
        self.records = slice_records
        self.tta_transform = tta_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["path"]).convert("RGB")
        tta_imgs = self.tta_transform(image)
        return {
            "images": tta_imgs,
            "label": rec["label"],
            "gender_idx": rec["gender_idx"],
            "scan_id": rec["scan_id"],
        }


def tta_collate(batch):
    """Collate: stack TTA views, flatten batch."""
    all_imgs = []
    all_labels = []
    all_genders = []
    all_scan_ids = []
    for b in batch:
        for img in b["images"]:
            all_imgs.append(img)
            all_labels.append(b["label"])
            all_genders.append(b["gender_idx"])
            all_scan_ids.append(b["scan_id"])
    return {
        "image": torch.stack(all_imgs),
        "label": torch.tensor(all_labels, dtype=torch.long),
        "gender_idx": torch.tensor(all_genders, dtype=torch.long),
        "scan_id": all_scan_ids,
    }


def predict_with_tta(
    model,
    loader,
    device,
    n_tta_per_slice,
    aggregation="mean",
):
    """
    For each slice, average logits over TTA views. Then aggregate per scan.
    """
    model.eval()
    slice_logits = {}  # scan_id -> list of (mean logits over TTA)

    with torch.no_grad():
        for batch in tqdm(loader, desc="TTA inference"):
            images = batch["image"].to(device)
            scan_ids = batch["scan_id"]
            logits = model(images)
            logits = logits.cpu().numpy()

            # Reshape: (B * n_tta, C) -> group by n_tta, average
            n = len(scan_ids) // n_tta_per_slice
            for i in range(n):
                start = i * n_tta_per_slice
                end = start + n_tta_per_slice
                sid = scan_ids[start]
                L = logits[start:end]
                mean_logits = np.mean(L, axis=0)
                if sid not in slice_logits:
                    slice_logits[sid] = []
                slice_logits[sid].append(mean_logits)

    # Aggregate per scan
    scan_preds = {}
    scan_labels = {}
    scan_genders = {}
    for sid, logits_list in slice_logits.items():
        if aggregation == "max":
            agg = np.max(logits_list, axis=0)
        else:
            agg = np.mean(logits_list, axis=0)
        pred = int(np.argmax(agg))
        scan_preds[sid] = pred
        # Get label/gender from first record (we need to pass them through - from loader we have batch)
        # We'll get labels/genders from the dataset separately
        scan_labels[sid] = None
        scan_genders[sid] = None

    return scan_preds, scan_labels, scan_genders


def predict_with_tta_simple(
    model,
    loader,
    device,
    n_tta,
    aggregation="mean",
):
    """
    Simpler: loader returns batches with (image, scan_id, label, gender_idx).
    For each batch, we get logits. Group by scan_id and average.
    """
    model.eval()
    from collections import defaultdict
    scan_logits = defaultdict(list)
    scan_labels = {}
    scan_genders = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="TTA inference"):
            images = batch["image"].to(device)
            scan_ids = batch["scan_id"]
            labels = batch["label"]
            genders = batch["gender_idx"]
            logits = model(images)
            logits = logits.cpu().numpy()
            for i, sid in enumerate(scan_ids):
                scan_logits[sid].append(logits[i])
                scan_labels[sid] = int(labels[i].item())
                scan_genders[sid] = int(genders[i].item())

    scan_preds = {}
    for sid, log_list in scan_logits.items():
        if aggregation == "max":
            agg = np.max(log_list, axis=0)
        else:
            agg = np.mean(log_list, axis=0)
        scan_preds[sid] = int(np.argmax(agg))

    y_true = [scan_labels[sid] for sid in sorted(scan_preds.keys())]
    y_pred = [scan_preds[sid] for sid in sorted(scan_preds.keys())]
    genders = [scan_genders[sid] for sid in sorted(scan_preds.keys())]
    return y_true, y_pred, genders


def run_inference_no_tta(model, loader, device, aggregation="mean"):
    """Standard inference without TTA (for validation)."""
    model.eval()
    from collections import defaultdict
    scan_logits = defaultdict(list)
    scan_labels = {}
    scan_genders = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            images = batch["image"].to(device)
            scan_ids = batch["scan_id"]
            labels = batch["label"]
            genders = batch["gender_idx"]
            logits = model(images)
            logits = logits.cpu().numpy()
            for i, sid in enumerate(scan_ids):
                scan_logits[sid].append(logits[i])
                scan_labels[sid] = int(labels[i].item())
                scan_genders[sid] = int(genders[i].item())

    scan_preds = {}
    for sid, log_list in scan_logits.items():
        if aggregation == "max":
            agg = np.max(log_list, axis=0)
        else:
            agg = np.mean(log_list, axis=0)
        scan_preds[sid] = int(np.argmax(agg))

    y_true = [scan_labels[sid] for sid in sorted(scan_preds.keys())]
    y_pred = [scan_preds[sid] for sid in sorted(scan_preds.keys())]
    genders = [scan_genders[sid] for sid in sorted(scan_preds.keys())]
    return y_true, y_pred, genders


def run_inference_mil(model, loader, device):
    """MIL model: one prediction per scan from (images, mask)."""
    model.eval()
    all_preds, all_labels, all_genders = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            images = batch["images"].to(device)
            mask = batch["mask"].to(device)
            logits = model(images, mask)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())
            all_genders.extend(batch["gender_idx"].numpy())
    return all_labels, all_preds, all_genders


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.split == "val":
        scan_records = build_scan_records(str(DATA_ROOT), split="val")
    else:
        scan_records = build_scan_records(str(DATA_ROOT), split="train")

    val_transform = get_val_transforms(IMG_SIZE)
    ckpt_path = CHECKPOINT_DIR / (args.checkpoint or "best_fold0.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    use_mil = ckpt.get("use_mil", USE_MIL)

    if use_mil:
        dataset = ScanLevelDataset(
            scan_records,
            transform=val_transform,
            max_slices=MAX_SLICES_PER_SCAN,
            is_training=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=MIL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_scan_batch,
        )
        model = ScanLevelMIL(num_classes=N_CLASSES, model_name=args.model, pretrained=False)
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        y_true, y_pred, genders = run_inference_mil(model, loader, device)
    else:
        slice_records = build_slice_records(
            scan_records, slices_per_scan=INFERENCE_SLICES_PER_SCAN
        )
        dataset = SliceDataset(slice_records, transform=val_transform, is_training=False)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_with_scan_id,
        )
        model = SliceClassifier(num_classes=N_CLASSES, model_name=args.model, pretrained=False)
        model.load_state_dict(ckpt["model"])
        model = model.to(device)
        if args.tta:
            y_true, y_pred, genders = run_tta_inference(
                model, slice_records, device, val_transform, aggregation=AGGREGATION
            )
        else:
            y_true, y_pred, genders = run_inference_no_tta(
                model, loader, device, AGGREGATION
            )

    pg = per_gender_macro_f1(y_true, y_pred, genders)
    full = full_metrics_report(y_true, y_pred, genders, CLASSES)
    print("Per-gender macro F1:", pg)
    print("Overall (competition metric):", pg["overall_f1"])
    print("\nClassification report:\n", full["report"])
    print("Confusion matrix:\n", full["confusion_matrix"])


def run_tta_inference(model, slice_records, device, base_transform, aggregation="mean"):
    """
    Run TTA: for each slice, predict on original + flip + shifts, average logits.
    Then aggregate per scan.
    """
    model.eval()
    from collections import defaultdict
    scan_logits = defaultdict(list)
    scan_labels = {}
    scan_genders = {}

    def get_tta_views(img):
        views = [img]
        if TTA_FLIP:
            views.append(img.transpose(Image.FLIP_LEFT_RIGHT))
        for dx, dy in TTA_SHIFTS[:TTA_N_SHIFTS]:
            if dx != 0 or dy != 0:
                w, h = img.size
                shifted = img.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, dy), fill=0)
                views.append(shifted)
        return views

    for rec in tqdm(slice_records, desc="TTA"):
        img = Image.open(rec["path"]).convert("RGB")
        views = get_tta_views(img)
        logits_list = []
        for v in views:
            x = base_transform(v).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x).cpu().numpy()[0]
            logits_list.append(logits)
        mean_logits = np.mean(logits_list, axis=0)
        sid = rec["scan_id"]
        scan_logits[sid].append(mean_logits)
        scan_labels[sid] = rec["label"]
        scan_genders[sid] = rec["gender_idx"]

    scan_preds = {}
    for sid, log_list in scan_logits.items():
        if aggregation == "max":
            agg = np.max(log_list, axis=0)
        else:
            agg = np.mean(log_list, axis=0)
        scan_preds[sid] = int(np.argmax(agg))

    y_true = [scan_labels[sid] for sid in sorted(scan_preds.keys())]
    y_pred = [scan_preds[sid] for sid in sorted(scan_preds.keys())]
    genders = [scan_genders[sid] for sid in sorted(scan_preds.keys())]
    return y_true, y_pred, genders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta_n", type=int, default=3)
    args = parser.parse_args()
    main(args)
