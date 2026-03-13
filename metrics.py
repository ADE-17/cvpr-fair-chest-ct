"""
Metrics for fair disease diagnosis.
Competition metric: average of per-gender macro F1-scores.
"""

import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from collections import defaultdict


def per_gender_macro_f1(y_true, y_pred, gender_indices):
    """
    Compute macro F1 per gender, then average.
    gender_indices: 0=male, 1=female (or array of same length as y_true)
    Returns: dict with male_f1, female_f1, overall (average)
    """
    gender_indices = np.asarray(gender_indices)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(np.unique(y_true)) if len(np.unique(y_true)) > 1 else max(y_true.max(), y_pred.max()) + 1

    results = {}
    for g, name in enumerate(["male", "female"]):
        mask = gender_indices == g
        if mask.sum() == 0:
            results[f"{name}_macro_f1"] = 0.0
            results[f"{name}_n"] = 0
            continue
        gt, pr = y_true[mask], y_pred[mask]
        f1 = f1_score(gt, pr, average="macro", zero_division=0)
        results[f"{name}_macro_f1"] = float(f1)
        results[f"{name}_n"] = int(mask.sum())

    results["overall_f1"] = (
        (results["male_macro_f1"] + results["female_macro_f1"]) / 2
        if (results.get("male_n", 0) > 0 and results.get("female_n", 0) > 0)
        else results.get("male_macro_f1", 0) or results.get("female_macro_f1", 0)
    )
    return results


def aggregate_slices_to_volume(logits_or_preds, scan_ids, strategy="mean"):
    """
    Aggregate slice-level predictions to volume-level.
    logits_or_preds: (N,) for class indices, or (N, C) for logits/probs
    scan_ids: list of scan identifiers
    strategy: mean or max
    Returns: dict scan_id -> aggregated prediction (class index)
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for i, sid in enumerate(scan_ids):
        grouped[sid].append(i)

    scan_preds = {}
    for sid, indices in grouped.items():
        vals = np.array([logits_or_preds[i] for i in indices])
        if vals.ndim == 1:
            # class indices: majority vote
            from collections import Counter
            pred = Counter(vals).most_common(1)[0][0]
        else:
            # logits or probs: mean or max over slices
            agg = np.mean(vals, axis=0) if strategy == "mean" else np.max(vals, axis=0)
            pred = int(np.argmax(agg))
        scan_preds[sid] = pred
    return scan_preds


def full_metrics_report(y_true, y_pred, gender_indices, class_names=None):
    """
    Full report: per-gender macro F1, per-class F1, confusion matrix.
    y_true, y_pred, gender_indices: at volume level (after aggregation).
    """
    class_names = class_names or ["A", "G", "normal", "covid"]
    report = {}

    # Competition metric
    pg = per_gender_macro_f1(y_true, y_pred, gender_indices)
    report.update(pg)

    # Per-class F1
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        report[f"f1_{name}"] = float(per_class[i]) if i < len(per_class) else 0.0

    # Macro F1 (overall, not per-gender)
    report["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Classification report string
    report["report"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return report
