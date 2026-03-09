"""
Configuration for AI6103 experiments
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Dataset
IMAGE_SIZE = 100
NUM_CLASSES = 11

# Training defaults
DEFAULT_BATCH_SIZE = 128
DEFAULT_MOMENTUM = 0.9
DEFAULT_EPOCHS = 15

# Device
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
