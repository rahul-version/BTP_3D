import os


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")

HR_PATH = os.path.join(DATASET_DIR, "HR")
LR_PATH = os.path.join(DATASET_DIR, "LR")

# Dedicated Validation Directory
VAL_DIR = os.path.join(DATASET_DIR, "Validation")
VAL_HR_PATH = os.path.join(VAL_DIR, "HR")
VAL_LR_PATH = os.path.join(VAL_DIR, "LR")

MODEL_PATH = os.path.join(PROJECT_ROOT, "model")
RESULT_PATH = os.path.join(PROJECT_ROOT, "result")
INPUT_PATH = os.path.join(PROJECT_ROOT, "input")


# ============================================================
# DATASET & SPLITS (Train, Validation, Test)
# ============================================================

SCALE = 4

PATCH_SIZE = 16

BATCH_SIZE = 1

NUM_WORKERS = 0

# Split definitions:
# Out of 19 Cryo-EM MRC volumes:
# - Training set: 13 volumes
# - Validation set: 3 volumes (emd_7081, emd_7769, emd_8923) in Dataset/Validation/
# - Testing set: 3 volumes (emd_8983, emd_9718, emd_10064)
VAL_SPLIT_FILES = ["emd_7081", "emd_7769", "emd_8923"]
TEST_SPLIT_FILES = ["emd_8983", "emd_9718", "emd_10064"]

# Output folders for graphs & metrics connected to evaluate.py parent folder (PROJECT_ROOT)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
METRICS_LOG_DIR = os.path.join(PROJECT_ROOT, "metrics_logs")
CHECKPOINT_EVAL_DIR = os.path.join(PROJECT_ROOT, "checkpoint_evaluations")


# ============================================================
# GENERATOR
# ============================================================

IMG_FEAT = 1

N_FEATS = 64

RES_NUM = 16


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
