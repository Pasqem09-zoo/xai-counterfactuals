
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
CHECKPOINT_NAME = "resnet18full_lr1e-7_try2.pth"
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME


# Weights & Biases configuration
WANDB_PROJECT = "xai-counterfactuals"


# Dataset configuration
IMAGE_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["real", "fake"]


# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-7
RANDOM_SEED = 42


# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
