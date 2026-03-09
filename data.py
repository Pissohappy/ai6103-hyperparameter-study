"""
Data loading and preprocessing for Food-11 dataset
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from config import DATA_DIR, IMAGE_SIZE, DEFAULT_BATCH_SIZE, DEVICE


def get_transforms(mean=None, std=None, augment=True):
    """
    Get data transforms for training and validation
    
    Args:
        mean: RGB mean values for normalization
        std: RGB std values for normalization
        augment: Whether to apply data augmentation
    
    Returns:
        train_transform, val_transform
    """
    normalize = None
    if mean is not None and std is not None:
        normalize = transforms.Normalize(mean=mean, std=std)
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(IMAGE_SIZE, padding=12),
            transforms.ToTensor(),
        ])
        if normalize:
            train_transform.transforms.append(normalize)
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        if normalize:
            train_transform.transforms.append(normalize)
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    if normalize:
        val_transform.transforms.append(normalize)
    
    return train_transform, val_transform


def compute_dataset_stats(data_path):
    """
    Compute mean and std for each RGB channel on training set
    
    Args:
        data_path: Path to training data directory
    
    Returns:
        mean, std: Lists of mean and std for R, G, B channels
    """
    print("Computing dataset statistics...")
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize accumulators
    mean = 0.0
    std = 0.0
    total_images = 0
    
    for images, _ in tqdm(loader, desc="Computing stats"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    
    print(f"\nDataset Statistics:")
    print(f"  Mean (R, G, B): {mean.tolist()}")
    print(f"  Std (R, G, B): {std.tolist()}")
    
    return mean.tolist(), std.tolist()


def get_dataloaders(data_dir=DATA_DIR, batch_size=DEFAULT_BATCH_SIZE, mean=None, std=None, 
                    augment=True, num_workers=4):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root directory of dataset
        batch_size: Batch size
        mean: Normalization mean
        std: Normalization std
        augment: Whether to use data augmentation
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_path = os.path.join(data_dir, 'training')
    val_path = os.path.join(data_dir, 'validation')
    test_path = os.path.join(data_dir, 'evaluation')
    
    train_transform, val_transform = get_transforms(mean, std, augment=augment)
    
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=val_transform)
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    print(f"  Classes: {train_dataset.classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


class MixupDataLoader:
    """
    DataLoader wrapper that applies mixup augmentation
    """
    def __init__(self, dataloader, alpha=0.2, device='cuda'):
        self.dataloader = dataloader
        self.alpha = alpha
        self.device = device
    
    def __iter__(self):
        for images, labels in self.dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Sample lambda from Beta distribution
            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1.0
            
            # Shuffle indices
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(self.device)
            
            # Mix images
            mixed_images = lam * images + (1 - lam) * images[index]
            
            yield mixed_images, labels, labels[index], lam
    
    def __len__(self):
        return len(self.dataloader)


if __name__ == "__main__":
    # Test data loading and compute statistics
    train_path = os.path.join(DATA_DIR, 'training')
    if os.path.exists(train_path):
        mean, std = compute_dataset_stats(train_path)
        print(f"\nAdd these to your config:")
        print(f"MEAN = {mean}")
        print(f"STD = {std}")
    else:
        print(f"Data directory not found: {train_path}")
        print("Please run download_data.py first")
