
"""
Configuration parameters for the XAI Counterfactuals project.
"""

from pathlib import Path
import torch


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_NAME = "resnet50_exp10_scheduler_earlystop.pth"
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME


# Weights & Biases configuration
WANDB_PROJECT = "xai-counterfactuals"


# Dataset configuration
IMAGE_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["fake", "real"]


# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01     ### TODO: provare 0.1????
RANDOM_SEED = 42
DROPOUT = 0.08
# Learning rate scheduler (ReduceLROnPlateau)
SCHEDULER_MODE = "min"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
SCHEDULER_THRESHOLD = 1e-4
SCHEDULER_THRESHOLD_MODE = "rel"
SCHEDULER_MIN_LR = 1e-7
# Early stopping
EARLY_STOPPING_PATIENCE = 8



# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
