"""
Slice-level dataset for CVPR Fair Disease Diagnosis.
Supports class/gender/ct_scan_XX/slices structure with scan_id for volume aggregation.
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from config import (
    CLASS_TO_IDX,
    GENDER_TO_IDX,
    IMG_SIZE,
    CLASSES,
    GENDERS,
    TRAIN_SLICES_PER_SCAN,
    MAX_SLICES_PER_SCAN,
)


def build_scan_records(root_dir, split="train"):
    """Build list of (scan_path, class, gender, slice_paths) per scan."""
    root = Path(root_dir)
    base = root if split == "train" else root / "val"
    records = []

    for class_name in CLASSES:
        class_path = base / class_name
        if not class_path.is_dir():
            continue
        for gender in GENDERS:
            gender_path = class_path / gender
            if not gender_path.is_dir():
                continue
            for scan_dir in sorted(gender_path.iterdir()):
                if not scan_dir.is_dir():
                    continue
                slices = sorted(
                    [f for f in scan_dir.glob("*.jpg") if not f.name.startswith("._")]
                )
                if not slices:
                    slices = sorted(
                        [f for f in scan_dir.glob("*.png") if not f.name.startswith("._")]
                    )
                if not slices:
                    continue
                records.append({
                    "scan_path": str(scan_dir),
                    "scan_id": f"{class_name}_{gender}_{scan_dir.name}",
                    "class": class_name,
                    "label": CLASS_TO_IDX[class_name],
                    "gender": gender,
                    "gender_idx": GENDER_TO_IDX[gender],
                    "slice_paths": [str(p) for p in slices],
                    "split": split,
                })
    return records


def build_slice_records(scan_records, slices_per_scan=None, max_slices=None):
    """
    Flatten to slice-level: one row per slice, with scan_id for aggregation.
    If slices_per_scan, randomly sample that many per scan (for training).
    If max_slices, cap total slices per scan (e.g. for inference).
    """
    max_slices = max_slices or MAX_SLICES_PER_SCAN
    rows = []
    for rec in scan_records:
        paths = rec["slice_paths"]
        if slices_per_scan and len(paths) > slices_per_scan:
            paths = random.sample(paths, slices_per_scan)
        elif not slices_per_scan and len(paths) > max_slices:
            # Cap for inference: uniform sample
            step = len(paths) / max_slices
            indices = [min(int(i * step), len(paths) - 1) for i in range(max_slices)]
            paths = [paths[i] for i in indices]
        for p in paths:
            rows.append({
                "path": p,
                "scan_id": rec["scan_id"],
                "label": rec["label"],
                "gender": rec["gender"],
                "gender_idx": rec["gender_idx"],
            })
    return rows


class SliceDataset(Dataset):
    """Slice-level dataset. Returns (image, label, gender_idx, scan_id)."""

    def __init__(self, slice_records, transform=None, is_training=True):
        self.records = slice_records
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": rec["label"],
            "gender_idx": rec["gender_idx"],
            "scan_id": rec["scan_id"],
        }


def get_train_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# Collate that preserves scan_id for grouped inference
def collate_with_scan_id(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    gender_idxs = torch.tensor([b["gender_idx"] for b in batch], dtype=torch.long)
    scan_ids = [b["scan_id"] for b in batch]
    return {
        "image": images,
        "label": labels,
        "gender_idx": gender_idxs,
        "scan_id": scan_ids,
    }


# --- Scan-level dataset for MIL (one sample = one scan, up to MAX_SLICES_PER_SCAN slices) ---
class ScanLevelDataset(Dataset):
    """
    One item = one scan: list of slice images (up to MAX_SLICES_PER_SCAN), same label/gender for all.
    For MIL: model sees all slices and learns to weight them.
    """

    def __init__(self, scan_records, transform=None, max_slices=None, is_training=True):
        self.records = scan_records
        self.transform = transform or get_val_transforms()
        self.max_slices = max_slices or MAX_SLICES_PER_SCAN
        self.is_training = is_training

    def __len__(self):
        return len(self.records)

    def _sample_slice_paths(self, paths):
        n = min(len(paths), self.max_slices)
        if self.is_training and len(paths) > n:
            return random.sample(paths, n)
        if not self.is_training and len(paths) > n:
            step = len(paths) / n
            indices = [min(int(i * step), len(paths) - 1) for i in range(n)]
            return [paths[i] for i in indices]
        return paths

    def __getitem__(self, idx):
        rec = self.records[idx]
        paths = self._sample_slice_paths(rec["slice_paths"])
        images = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        n_slices = len(images)
        # Pad to max_slices with zeros so collate can stack
        images_tensor = torch.stack(images)
        if n_slices < self.max_slices:
            padding = torch.zeros(
                self.max_slices - n_slices,
                images_tensor.shape[1],
                images_tensor.shape[2],
                images_tensor.shape[3],
                dtype=images_tensor.dtype,
            )
            images_tensor = torch.cat([images_tensor, padding], dim=0)
        return {
            "images": images_tensor,
            "mask": torch.tensor([1] * n_slices + [0] * (self.max_slices - n_slices), dtype=torch.float32),
            "label": rec["label"],
            "gender_idx": rec["gender_idx"],
            "scan_id": rec["scan_id"],
            "n_slices": n_slices,
        }


def collate_scan_batch(batch):
    """Batch of scans: images (B, N_max, C, H, W), mask (B, N_max)."""
    images = torch.stack([b["images"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    gender_idxs = torch.tensor([b["gender_idx"] for b in batch], dtype=torch.long)
    scan_ids = [b["scan_id"] for b in batch]
    return {
        "images": images,
        "mask": mask,
        "label": labels,
        "gender_idx": gender_idxs,
        "scan_id": scan_ids,
    }
