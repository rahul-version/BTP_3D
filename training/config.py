import os


# PROJECT PATHS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")

HR_PATH = os.path.join(DATASET_DIR, "HR")
LR_PATH = os.path.join(DATASET_DIR, "LR")

MODEL_PATH = os.path.join(PROJECT_ROOT, "model")
RESULT_PATH = os.path.join(PROJECT_ROOT, "result")
INPUT_PATH = os.path.join(PROJECT_ROOT, "input")


# DATASET

SCALE = 4

PATCH_SIZE = 16

BATCH_SIZE = 1

NUM_WORKERS = 0


# GENERATOR

IMG_FEAT = 1

N_FEATS = 64

RES_NUM = 16


# DISCRIMINATOR

NUM_D_BLOCK = 3


# TRAINING

PRE_TRAIN_EPOCH = 30

FINE_TRAIN_EPOCH = 30

LEARNING_RATE = 5e-2


# LOSS WEIGHTS

L2_COEFF = 1.0

ADV_COEFF = 1e-3

GRAD_COEFF = 1.0

SPECTRAL_COEFF = 0.1

TV_LOSS_COEFF = 1e-4


# CHECKPOINT NAMES

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