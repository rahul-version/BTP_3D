import os
import random

import numpy as np
import torch


# RANDOM SEED

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# DEVICE

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")


# CREATE DIRECTORIES

def ensure_dirs(*paths):

    for path in paths:

        os.makedirs(
            path,
            exist_ok=True
        )


# PARAMETER COUNT

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# SAVE CHECKPOINT

def save_checkpoint(
    model,
    path
):

    torch.save(
        model.state_dict(),
        path
    )