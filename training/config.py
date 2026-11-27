import os


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")

HR_PATH = os.path.join(DATASET_DIR, "HR")
LR_PATH = os.path.join(DATASET_DIR, "LR")

# Dedicated Train / Validation / Test Directories (80% - 10% - 10% Split)
TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
TRAIN_HR_PATH = os.path.join(TRAIN_DIR, "HR")
TRAIN_LR_PATH = os.path.join(TRAIN_DIR, "LR")

VAL_DIR = os.path.join(DATASET_DIR, "Validation")
VAL_HR_PATH = os.path.join(VAL_DIR, "HR")
VAL_LR_PATH = os.path.join(VAL_DIR, "LR")

TEST_DIR = os.path.join(DATASET_DIR, "Test")
TEST_HR_PATH = os.path.join(TEST_DIR, "HR")
TEST_LR_PATH = os.path.join(TEST_DIR, "LR")

MODEL_PATH = os.path.join(PROJECT_ROOT, "model")
RESULT_PATH = os.path.join(PROJECT_ROOT, "result")
INPUT_PATH = os.path.join(PROJECT_ROOT, "input")


# ============================================================
# DATASET & SPLITS (80% Train | 10% Validation | 10% Test)
# ============================================================

SCALE = 4

PATCH_SIZE = 16

BATCH_SIZE = 1

NUM_WORKERS = 0

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def get_split_files(manifest_path=None):
    """
    Returns (train_files, val_files, test_files) based on 80-10-10 split.
    Reads Dataset/split_manifest.json if available, or calculates dynamically with seed=42.
    """
    if manifest_path is None:
        manifest_path = os.path.join(DATASET_DIR, "split_manifest.json")
    if os.path.exists(manifest_path):
        try:
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d.get("train_stems", []), d.get("val_stems", []), d.get("test_stems", [])
        except Exception:
            pass

    # Dynamic 80-10-10 calculation based on HR_PATH files
    import random
    all_files = sorted([f.lower().replace(".mrc", "") for f in os.listdir(HR_PATH) if f.lower().endswith(".mrc")]) if os.path.exists(HR_PATH) else []
    if not all_files:
        return [], ["emd_7081", "emd_7769", "emd_8923"], ["emd_8983", "emd_9718", "emd_10064"]

    rng = random.Random(42)
    shuffled = all_files.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, round(n * 0.10))
    n_test = max(1, round(n * 0.10))
    test_f = shuffled[:n_test]
    val_f = shuffled[n_test : n_test + n_val]
    train_f = shuffled[n_test + n_val :]
    return train_f, val_f, test_f


TRAIN_SPLIT_FILES, VAL_SPLIT_FILES, TEST_SPLIT_FILES = get_split_files()

# Output folders for graphs & metrics connected to evaluate.py parent folder (PROJECT_ROOT)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
METRICS_LOG_DIR = os.path.join(PROJECT_ROOT, "metrics_logs")
CHECKPOINT_EVAL_DIR = os.path.join(PROJECT_ROOT, "checkpoint_evaluations")


# ============================================================
# GENERATOR (3D ResNet-50 Architecture - He et al. 2016 / Med3D)
# ============================================================

IMG_FEAT = 1

N_FEATS = 64

# ResNet-50 exact layer distribution: [3, 4, 6, 3] = 16 Bottleneck blocks (48 conv layers)
# Total depth = 48 (bottleneck) + 1 (conv01) + 1 (conv02) = 50 convolutional layers
RES_NUM = 16
RESNET_LAYERS = [3, 4, 6, 3]
RESNET_BASE_PLANES = 64


# ============================================================
# DISCRIMINATOR
# ============================================================

NUM_D_BLOCK = 3


# ============================================================
# TRAINING
# ============================================================

PRE_TRAIN_EPOCH = 30

FINE_TRAIN_EPOCH = 30

LEARNING_RATE = 5e-4


# ============================================================
# LOSS WEIGHTS
# ============================================================

L2_COEFF = 1.0

ADV_COEFF = 1e-3

GRAD_COEFF = 1.0

SPECTRAL_COEFF = 0.1

TV_LOSS_COEFF = 1e-4


# ============================================================
# CHECKPOINT NAMES
# ============================================================

GENERATOR_PRETRAINED = os.path.join(
    MODEL_PATH,
    "generator_pretrained.pt"
)

GENERATOR_FINAL = os.path.join(
    MODEL_PATH,
    "generator_final.pt"
)

DISCRIMINATOR_FINAL = os.path.join(
    MODEL_PATH,
    "discriminator_final.pt"
)

BEST_GENERATOR = os.path.join(
    MODEL_PATH,
    "best_generator.pt"
)
