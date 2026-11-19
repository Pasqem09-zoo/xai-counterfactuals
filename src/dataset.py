
"""
Dataset loading and preprocessing for real/fake face classification.
"""

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from src.config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED


def get_transforms():
    """Define the preprocessing transformations for ResNet18."""

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform



def get_dataloaders():
    """Load the dataset, split it into training, validation, and test sets, and return DataLoaders for each set.
    With train_test_split from sklearn.model_selection, we can stratify the split based on the class labels; so that the proportion of real and fake images is maintained in each subset."""

    # Load the complete dataset
    dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=get_transforms()
    )

    # Image indices and corresponding class labels
    indices = list(range(len(dataset)))
    labels = dataset.targets

    # First split: 70% training, 30% remaining
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,            # Stratify the split based on class labels to maintain the proportion of real and fake images in each subset.
        random_state=RANDOM_SEED
    )

    # Labels of the remaining 30%
    temp_labels = [labels[i] for i in temp_idx]

    # Second split: 15% validation, 15% test
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )

    # Create dataset subsets
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader



if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")
