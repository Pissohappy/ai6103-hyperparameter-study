"""
Utility functions for plotting and analysis
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import OUTPUT_DIR

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


def plot_training_curves(history, title, save_path=None, show=True):
    """
    Plot training loss and accuracy curves
    
    Args:
        history: Dictionary with train_loss, train_acc, val_loss, val_acc
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{title} - Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{title} - Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_lr_comparison(histories, labels, title, save_path=None, show=True):
    """
    Compare training curves for different learning rates
    
    Args:
        histories: List of history dictionaries
        labels: List of labels for each experiment
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for i, (history, label, color) in enumerate(zip(histories, labels, colors)):
        epochs = range(1, len(history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, history['train_loss'], color=color, linewidth=2, label=label)
        axes[0, 1].plot(epochs, history['val_loss'], color=color, linewidth=2, label=label)
        axes[1, 0].plot(epochs, history['train_acc'], color=color, linewidth=2, label=label)
        axes[1, 1].plot(epochs, history['val_acc'], color=color, linewidth=2, label=label)
    
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Validation Loss', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Training Accuracy', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Validation Accuracy', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_beta_distribution(alpha=0.2, save_path=None, show=True):
    """
    Plot the Beta distribution PDF for mixup
    
    Args:
        alpha: Alpha parameter for Beta distribution
        save_path: Path to save the figure
        show: Whether to display the plot
    """
    x = np.linspace(0, 1, 1000)
    pdf = stats.beta.pdf(x, alpha, alpha)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, pdf, 'b-', linewidth=2, label=f'Beta(α={alpha}, β={alpha})')
    plt.fill_between(x, pdf, alpha=0.3)
    plt.xlabel('λ', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title(f'Beta Distribution for Mixup (α={alpha})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, max(pdf) * 1.1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_rate_schedule(history, title, save_path=None, show=True):
    """
    Plot learning rate over epochs
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history['lr']) + 1), history['lr'], 'g-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def get_final_results(history):
    """
    Get final training and validation results
    
    Returns:
        Dictionary with final metrics
    """
    return {
        'final_train_loss': history['train_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'best_val_loss': min(history['val_loss'])
    }


def print_results_table(results_dict):
    """
    Print a formatted table of results
    
    Args:
        results_dict: Dictionary mapping experiment names to results
    """
    print("\n" + "="*80)
    print(f"{'Experiment':<30} {'Train Loss':>10} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>10}")
    print("="*80)
    
    for name, results in results_dict.items():
        print(f"{name:<30} {results['final_train_loss']:>10.4f} "
              f"{results['final_train_acc']:>9.2f}% "
              f"{results['final_val_loss']:>10.4f} "
              f"{results['final_val_acc']:>9.2f}%")
    
    print("="*80)


def save_results(results, save_path):
    """Save results dictionary to JSON"""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")


def load_results(load_path):
    """Load results dictionary from JSON"""
    with open(load_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test beta distribution plot
    plot_beta_distribution(alpha=0.2, save_path=os.path.join(OUTPUT_DIR, 'beta_distribution.png'))
