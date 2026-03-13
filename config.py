"""Configuration for CVPR Fair Disease Diagnosis pipeline."""

from pathlib import Path

# Paths
DATA_ROOT = Path("/scratch/adipa/cvpr_hack")
OUTPUT_DIR = Path("/scratch/adipa/cvpr_hack/outputs/convnext_tiny_grl")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Classes (A=Adenocarcinoma, G=Squamous Cell Carcinoma)
CLASSES = ["A", "G", "normal", "covid"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
N_CLASSES = 4

GENDERS = ["male", "female"]
GENDER_TO_IDX = {"male": 0, "female": 1}
# Minority group for oversampling: Female Squamous Cell Carcinoma (G)
SCC_CLASS_IDX = CLASS_TO_IDX["G"]
FEMALE_GENDER_IDX = GENDER_TO_IDX["female"]

# Model
MODEL_NAME = "convnext_tiny"  # timm: convnext_tiny, efficientnetv2_s, etc.
IMG_SIZE = 224  # 224 uses less GPU memory than 256
PRETRAINED = True

# Training
BATCH_SIZE = 16
MIL_BATCH_SIZE = 2  # MIL: each sample = MAX_SLICES_PER_SCAN slices; keep small to avoid OOM
NUM_WORKERS = 4
EPOCHS = 50
LR = 1e-4
# Lower LR for backbone when unfrozen (epochs 6–30)
LR_BACKBONE = 5e-5
WEIGHT_DECAY = 0.05
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.15
# Gradient accumulation (effective batch size = batch_size * ACCUM_STEPS)
ACCUM_STEPS = 8
# Freeze backbone for first FREEZE_EPOCHS epochs (1-indexed, so 5 => epochs 1–5)
FREEZE_EPOCHS = 5
#
# Adversarial gender-invariance via Gradient Reversal:
# train an auxiliary gender head to predict gender, but reverse gradients
# so the shared features are discouraged from encoding gender information.
USE_GENDER_ADV = True
GRL_LAMBDA = 0.1  # weight of gender adversarial loss relative to main disease loss

# Stratified K-Fold
N_FOLDS = 5
RANDOM_SEED = 42

# Volume aggregation: how to combine slice logits per scan
# "max" = any slice can trigger positive (good for rare tumor slices); "attention" = learned MIL
AGGREGATION = "max"

# Max slices per scan (cap for training and inference; tumor may appear in only 5–10 of 150)
# Lower = less GPU memory (MIL batch uses batch_size * this many images)
MAX_SLICES_PER_SCAN = 32
USE_MIL = True  # If True, use Attention-based MIL instead of simple max/mean

# Oversample minority: Female SCC (Squamous Cell Carcinoma) gets this weight so they appear in almost every batcht
MINORITY_BOOST_WEIGHT = 10.0  # weight for (class=G, gender=female)

# TTA
TTA_FLIP = True
TTA_SHIFTS = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]  # (dx, dy) in pixels
TTA_N_SHIFTS = 3  # use first N shifts if many (0=center only)

# Slice sampling per scan (capped by MAX_SLICES_PER_SCAN)
TRAIN_SLICES_PER_SCAN = 32  # slices per scan during training (≤ MAX_SLICES_PER_SCAN)
INFERENCE_SLICES_PER_SCAN = None  # None = use all slices up to MAX_SLICES_PER_SCAN

# Ensure dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
