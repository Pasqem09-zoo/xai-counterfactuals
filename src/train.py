
"""
Training ResNet18 for real/fake image classification.
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn

from src.config import (
    DEVICE,
    RANDOM_SEED,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    SCHEDULER_MODE,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_THRESHOLD,
    SCHEDULER_THRESHOLD_MODE,
    SCHEDULER_MIN_LR,
    EARLY_STOPPING_PATIENCE
)
from src.dataset import get_dataloaders
from src.models import *
from src.utils import (
    init_wandb,
    log_epoch,
    finish_wandb,
    get_scheduler,
    EarlyStopping
)


def main():

    # ========================================================
    # INITIALIZATION
    # ========================================================

    # Set random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Load dataset
    train_loader, val_loader, test_loader = get_dataloaders()

    # Load pretrained ResNet18 (full fine-tuning)
    model = ResNet50Classifier().to(DEVICE)     # ResNet50Classifier().to(DEVICE)  ResNet18Classifier().to(DEVICE)

    for parameter in model.parameters():
        parameter.requires_grad = True

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = get_scheduler(
        optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        threshold=SCHEDULER_THRESHOLD,
        threshold_mode=SCHEDULER_THRESHOLD_MODE,
        min_lr=SCHEDULER_MIN_LR
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE
    )

    # Print training configuration
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    optimizer_params = sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group["params"]
    )

    print("Trainable parameters:", trainable_params)
    print("Optimizer parameters:", optimizer_params)
    print("Optimizer:", type(optimizer).__name__)
    print("Learning rate:", optimizer.param_groups[0]["lr"])
    print("Weight decay:", optimizer.param_groups[0]["weight_decay"])

    # Prepare checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    run = init_wandb(
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset)
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(NUM_EPOCHS):

        epoch_start = time.perf_counter()
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward and backward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Training statistics
            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += batch_size

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                # Validation statistics
                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += batch_size

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        epoch_time = time.perf_counter() - epoch_start

        # ====================================================
        # LOGGING AND RESULTS
        # ====================================================

        # Learning rate used during this epoch
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics to W&B
        log_epoch(
            run,
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_accuracy,
            val_loss=val_loss,
            val_acc=val_accuracy,
            learning_rate=current_lr
        )

        # Print epoch results
        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ====================================================
        # CHECKPOINT, SCHEDULER AND EARLY STOPPING
        # ====================================================

        # Check validation loss improvement
        improved, should_stop = early_stopping.step(val_loss)

        # Save best model
        if improved:
            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

        # Update learning rate for the next epoch
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        print(f"Learning rate: {current_lr:.2e} -> {new_lr:.2e}")

        # Stop training if validation loss has not improved
        if should_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print("\nTraining completed.")
    print("Best validation loss:", early_stopping.best_loss)

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    # Load the best model selected on validation loss
    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Test statistics
            batch_size = labels.size(0)
            test_loss += loss.item() * batch_size
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()
            test_total += batch_size

    test_loss /= test_total
    test_accuracy = test_correct / test_total

    print("\nFinal test results (best checkpoint):")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Finish W&B run
    finish_wandb(run)


if __name__ == "__main__":
    main()
