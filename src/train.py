
"""
Training ResNet18 for real/fake image classification.
"""

import torch
import torch.nn as nn
import random
import numpy as np
import time

from src.utils import init_wandb, log_epoch, finish_wandb
from src.config import (
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    RANDOM_SEED,
    CHECKPOINT_DIR,
    CHECKPOINT_PATH
)
from src.dataset import get_dataloaders
from src.models import ResNet18Classifier


def main():

    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)                ### controlla il generatore pseudocasuale standard di Python
    np.random.seed(RANDOM_SEED)             ### controlla il generatore pseudocasuale di NumPy
    torch.manual_seed(RANDOM_SEED)          ### imposta il seed dei generatori casuali di PyTorch, compresi quelli utilizzati per inizializzare i parametri

    # Load dataset
    train_loader, val_loader, test_loader = get_dataloaders()

    # Load pretrained ResNet18
    model = ResNet18Classifier().to(DEVICE)

    # ### CODICE PER SCONGELARE SOLTANTO I PARAMETRI DELL'ULTIMO STRATO DI CLASSIFICAZIONE
    # # Freeze pretrained parameters
    # for parameter in model.parameters():
    #     parameter.requires_grad = False
    # # Unfreeze the final classification layer
    # for parameter in model.model.fc.parameters():
    #     parameter.requires_grad = True

    # ### CODICE PER SCONGELARE TUTTI I PARAMETRI
    # Enable training for all model parameters
    for parameter in model.parameters():
        parameter.requires_grad = True

    ### CODICE CON SOLO I PARAMETRI DELL'ULTIMO STRATO DI CLASSIFICAZIONE
    # Keep the pretrained feature extractor in evaluation mode
    # to avoid updating BatchNorm running statistics
    # model.eval()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        # model.model.fc.parameters(),
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
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

        ### CODICE CON TUTTI I PARAMETRI SCONGELATI
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # 1. Reset gradients
            optimizer.zero_grad()

            # 2. Forward pass
            outputs = model(images)

            # 3. Compute loss
            loss = criterion(outputs, labels)

            # 4. Backward pass
            loss.backward()

            # 5. Update parameters
            optimizer.step()

            # Training statistics
            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += batch_size

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        # ====================================================
        # VALIDATION
        # ====================================================

        ### CODICE CON TUTTI I PARAMETRI SCONGELATI
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

                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size

                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += batch_size

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        epoch_time = time.perf_counter() - epoch_start

        # Log epoch metrics to W&B
        log_epoch(
            run,
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_accuracy,
            val_loss=val_loss,
            val_acc=val_accuracy
        )

        # ====================================================
        # RESULTS AND CHECKPOINT
        # ====================================================

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )


        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

    print("\nTraining completed.")
    print("Best validation loss:", best_val_loss)


    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    # Load the best model selected on the validation set
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

            batch_size = labels.size(0)
            test_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            test_correct += (predictions == labels).sum().item()
            test_total += batch_size

    test_loss /= test_total
    test_accuracy = test_correct / test_total

    print("\nFinal test results (best checkpoint):")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


    finish_wandb(run)

if __name__ == "__main__":
    main()
