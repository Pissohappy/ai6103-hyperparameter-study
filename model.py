"""
Model definition - EfficientNet-B0
"""
import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES


def get_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True):
    """
    Create EfficientNet-B0 model
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: EfficientNet-B0 model
    """
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Modify the classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_efficientnet_b0(pretrained=False)
    print(f"Model: EfficientNet-B0")
    print(f"Trainable parameters: {count_parameters(model):,}")
