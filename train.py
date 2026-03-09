"""
Training utilities
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import os
from config import DEVICE, OUTPUT_DIR


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.2):
    """
    Train for one epoch
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    if use_mixup:
        for mixed_images, labels_a, labels_b, lam in pbar:
            mixed_images = mixed_images.to(device)
            labels_a = labels_a.to(device)
            labels_b = labels_b.to(device)
            
            optimizer.zero_grad()
            outputs = model(mixed_images)
            
            # Mixup loss
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels_a.size(0)
            # Approximate accuracy (use labels_a for simplicity)
            correct += predicted.eq(labels_a).sum().item()
    else:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """
    Validate the model
    
    Returns:
        avg_loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


class Trainer:
    """
    Trainer class that handles the full training loop
    """
    def __init__(self, model, train_loader, val_loader, device=DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def train(self, epochs, lr, momentum=0.9, weight_decay=0.0, 
              scheduler_type=None, use_mixup=False, mixup_alpha=0.2, 
              save_dir=None, experiment_name="experiment"):
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            lr: Initial learning rate
            momentum: SGD momentum
            weight_decay: Weight decay coefficient
            scheduler_type: None or 'cosine'
            use_mixup: Whether to use mixup augmentation
            mixup_alpha: Alpha parameter for mixup
            save_dir: Directory to save results
            experiment_name: Name for saving results
        
        Returns:
            history: Training history dictionary
        """
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
        
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        else:
            scheduler = None
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
        
        print(f"\n{'='*60}")
        print(f"Training: {experiment_name}")
        print(f"  Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")
        print(f"  Scheduler: {scheduler_type}, Mixup: {use_mixup}")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = train_one_epoch(
                self.model, self.train_loader, self.criterion, optimizer,
                self.device, use_mixup, mixup_alpha
            )
            
            # Validate
            val_loss, val_acc = validate(
                self.model, self.val_loader, self.criterion, self.device
            )
            
            # Update scheduler
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Track best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f}")
        
        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{experiment_name}.json")
            with open(save_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"\nResults saved to {save_path}")
        
        print(f"\nBest Val Acc: {best_val_acc:.2f}%, Best Val Loss: {best_val_loss:.4f}")
        
        return self.history
    
    def reset_history(self):
        """Reset training history"""
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def reset_model(self):
        """Reset model weights"""
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
