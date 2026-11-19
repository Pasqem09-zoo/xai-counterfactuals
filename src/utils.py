
"""
Utility functions for experiment tracking with Weights & Biases.
"""

from pathlib import Path
import wandb

from src.config import (
    WANDB_PROJECT,
    CHECKPOINT_NAME,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    RANDOM_SEED,
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
        }
    )

    return run


def log_epoch(run, epoch, train_loss, train_acc, val_loss, val_acc):
    """Log training and validation metrics after each epoch."""

    run.log(
        {
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        },
        step=epoch
    )


def finish_wandb(run):
    """Finish the current W&B run."""

    run.finish()
