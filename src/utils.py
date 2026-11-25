
"""
Utility functions for experiment tracking with Weights & Biases.
"""

from pathlib import Path
import wandb
import torch

from src.config import (
    WANDB_PROJECT,
    CHECKPOINT_NAME,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    RANDOM_SEED,
    WEIGHT_DECAY,
    DROPOUT
)


def init_wandb(train_size, val_size):
    """Initialize a W&B run and log experiment configuration."""

    # Store W&B local files outside the project directory
    wandb_dir = Path.home() / "wandb-runs"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=WANDB_PROJECT,
        name=Path(CHECKPOINT_NAME).stem,
        dir=str(wandb_dir),
        config={
            "model": "resnet18",
            "training_strategy": "full_fine_tuning",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "random_seed": RANDOM_SEED,
            "train_samples": train_size,
            "validation_samples": val_size,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT
        }
    )

    return run


def log_epoch(run, epoch, train_loss, train_acc, val_loss, val_acc, learning_rate):
    """Log training and validation metrics after each epoch."""

    run.log(
        {
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "train/learning_rate": learning_rate,
        },
        step=epoch
    )


def finish_wandb(run):
    """Finish the current W&B run."""

    run.finish()



def get_scheduler(optimizer, mode, factor, patience,
                  threshold, threshold_mode, min_lr):
    """Create a ReduceLROnPlateau learning rate scheduler."""

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode=threshold_mode,
        min_lr=min_lr
    )

    return scheduler



class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True, False

        self.counter += 1

        return False, self.counter >= self.patience
