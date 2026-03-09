"""
Experiment runner for AI6103 homework
"""
import os
import json
import torch
from datetime import datetime
from config import DEVICE, OUTPUT_DIR, DATA_DIR
from data import compute_dataset_stats, get_dataloaders, MixupDataLoader
from model import get_efficientnet_b0
from train import Trainer
from utils import (plot_training_curves, plot_lr_comparison, plot_beta_distribution,
                   get_final_results, print_results_table, save_results)


def run_section2():
    """
    Section 2: Data Preprocessing
    Compute dataset statistics
    """
    print("\n" + "="*60)
    print("SECTION 2: DATA PREPROCESSING")
    print("="*60)
    
    train_path = os.path.join(DATA_DIR, 'training')
    mean, std = compute_dataset_stats(train_path)
    
    # Save statistics
    stats = {'mean': mean, 'std': std}
    save_path = os.path.join(OUTPUT_DIR, 'dataset_stats.json')
    save_results(stats, save_path)
    
    return mean, std


def run_section3(mean, std):
    """
    Section 3: Learning Rate Experiments
    Test LR = 0.1, 0.025, 0.001 for 15 epochs
    """
    print("\n" + "="*60)
    print("SECTION 3: LEARNING RATE EXPERIMENTS")
    print("="*60)
    
    learning_rates = [0.1, 0.025, 0.001]
    epochs = 15
    
    train_loader, val_loader, test_loader = get_dataloaders(
        mean=mean, std=std, augment=True
    )
    
    results = {}
    histories = []
    labels = []
    
    for lr in learning_rates:
        print(f"\n{'='*40}")
        print(f"Learning Rate: {lr}")
        print(f"{'='*40}")
        
        model = get_efficientnet_b0(pretrained=False)
        trainer = Trainer(model, train_loader, val_loader, DEVICE)
        
        experiment_name = f"lr_{lr}"
        history = trainer.train(
            epochs=epochs,
            lr=lr,
            momentum=0.9,
            weight_decay=0.0,
            scheduler_type=None,
            save_dir=OUTPUT_DIR,
            experiment_name=experiment_name
        )
        
        histories.append(history)
        labels.append(f"LR={lr}")
        results[experiment_name] = get_final_results(history)
        
        # Plot individual curves
        plot_training_curves(
            history, 
            title=f"Learning Rate = {lr}",
            save_path=os.path.join(OUTPUT_DIR, f"section3_{experiment_name}.png"),
            show=False
        )
        
        # Clean up
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Comparison plot
    plot_lr_comparison(
        histories, labels,
        title="Learning Rate Comparison (Section 3)",
        save_path=os.path.join(OUTPUT_DIR, "section3_comparison.png"),
        show=False
    )
    
    # Print results table
    print_results_table(results)
    
    # Save results
    save_results(results, os.path.join(OUTPUT_DIR, "section3_results.json"))
    
    # Find best learning rate
    best_lr = max(results.items(), key=lambda x: x[1]['final_val_acc'])
    print(f"\n🏆 Best Learning Rate: {best_lr[0]} with Val Acc: {best_lr[1]['final_val_acc']:.2f}%")
    
    return best_lr[0], results


def run_section4(mean, std, best_lr_name):
    """
    Section 4: Learning Rate Schedule
    Compare fixed LR vs Cosine Annealing for 300 epochs
    """
    print("\n" + "="*60)
    print("SECTION 4: LEARNING RATE SCHEDULE")
    print("="*60)
    
    # Extract LR value from name (e.g., "lr_0.025" -> 0.025)
    best_lr = float(best_lr_name.split('_')[1])
    epochs = 300
    
    train_loader, val_loader, test_loader = get_dataloaders(
        mean=mean, std=std, augment=True
    )
    
    results = {}
    histories = []
    labels = []
    
    # Experiment 1: Fixed LR
    print(f"\n{'='*40}")
    print("Experiment 1: Fixed Learning Rate")
    print(f"{'='*40}")
    
    model = get_efficientnet_b0(pretrained=False)
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    
    history_fixed = trainer.train(
        epochs=epochs,
        lr=best_lr,
        momentum=0.9,
        weight_decay=0.0,
        scheduler_type=None,
        save_dir=OUTPUT_DIR,
        experiment_name="section4_fixed_lr"
    )
    
    histories.append(history_fixed)
    labels.append("Fixed LR")
    results["Fixed LR"] = get_final_results(history_fixed)
    
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Experiment 2: Cosine Annealing
    print(f"\n{'='*40}")
    print("Experiment 2: Cosine Annealing")
    print(f"{'='*40}")
    
    model = get_efficientnet_b0(pretrained=False)
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    
    history_cosine = trainer.train(
        epochs=epochs,
        lr=best_lr,
        momentum=0.9,
        weight_decay=0.0,
        scheduler_type='cosine',
        save_dir=OUTPUT_DIR,
        experiment_name="section4_cosine"
    )
    
    histories.append(history_cosine)
    labels.append("Cosine Annealing")
    results["Cosine Annealing"] = get_final_results(history_cosine)
    
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Comparison plot
    plot_lr_comparison(
        histories, labels,
        title="Learning Rate Schedule Comparison (Section 4)",
        save_path=os.path.join(OUTPUT_DIR, "section4_comparison.png"),
        show=False
    )
    
    # Print results table
    print_results_table(results)
    
    # Save results
    save_results(results, os.path.join(OUTPUT_DIR, "section4_results.json"))
    
    return results


def run_section5(mean, std, best_lr_name):
    """
    Section 5: Weight Decay
    Test λ = 5e-4 and 1e-4 with Cosine Annealing
    """
    print("\n" + "="*60)
    print("SECTION 5: WEIGHT DECAY")
    print("="*60)
    
    best_lr = float(best_lr_name.split('_')[1])
    epochs = 300
    weight_decays = [5e-4, 1e-4]
    
    train_loader, val_loader, test_loader = get_dataloaders(
        mean=mean, std=std, augment=True
    )
    
    results = {}
    histories = []
    labels = []
    
    for wd in weight_decays:
        print(f"\n{'='*40}")
        print(f"Weight Decay: {wd}")
        print(f"{'='*40}")
        
        model = get_efficientnet_b0(pretrained=False)
        trainer = Trainer(model, train_loader, val_loader, DEVICE)
        
        experiment_name = f"wd_{wd}"
        history = trainer.train(
            epochs=epochs,
            lr=best_lr,
            momentum=0.9,
            weight_decay=wd,
            scheduler_type='cosine',
            save_dir=OUTPUT_DIR,
            experiment_name=f"section5_{experiment_name}"
        )
        
        histories.append(history)
        labels.append(f"WD={wd}")
        results[f"WD={wd}"] = get_final_results(history)
        
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Comparison plot
    plot_lr_comparison(
        histories, labels,
        title="Weight Decay Comparison (Section 5)",
        save_path=os.path.join(OUTPUT_DIR, "section5_comparison.png"),
        show=False
    )
    
    # Print results table
    print_results_table(results)
    
    # Save results
    save_results(results, os.path.join(OUTPUT_DIR, "section5_results.json"))
    
    return results


def run_section6(mean, std, best_lr_name):
    """
    Section 6: Mixup Data Augmentation
    """
    print("\n" + "="*60)
    print("SECTION 6: MIXUP DATA AUGMENTATION")
    print("="*60)
    
    best_lr = float(best_lr_name.split('_')[1])
    epochs = 300
    mixup_alpha = 0.2
    
    # Plot Beta distribution
    plot_beta_distribution(
        alpha=mixup_alpha,
        save_path=os.path.join(OUTPUT_DIR, "section6_beta_distribution.png"),
        show=False
    )
    
    train_loader, val_loader, test_loader = get_dataloaders(
        mean=mean, std=std, augment=True
    )
    
    results = {}
    histories = []
    labels = []
    
    # Experiment 1: Without Mixup (baseline)
    print(f"\n{'='*40}")
    print("Experiment 1: Without Mixup")
    print(f"{'='*40}")
    
    model = get_efficientnet_b0(pretrained=False)
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    
    history_baseline = trainer.train(
        epochs=epochs,
        lr=best_lr,
        momentum=0.9,
        weight_decay=5e-4,  # Use best from Section 5
        scheduler_type='cosine',
        use_mixup=False,
        save_dir=OUTPUT_DIR,
        experiment_name="section6_no_mixup"
    )
    
    histories.append(history_baseline)
    labels.append("No Mixup")
    results["No Mixup"] = get_final_results(history_baseline)
    
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Experiment 2: With Mixup
    print(f"\n{'='*40}")
    print("Experiment 2: With Mixup")
    print(f"{'='*40}")
    
    model = get_efficientnet_b0(pretrained=False)
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    
    history_mixup = trainer.train(
        epochs=epochs,
        lr=best_lr,
        momentum=0.9,
        weight_decay=5e-4,
        scheduler_type='cosine',
        use_mixup=True,
        mixup_alpha=mixup_alpha,
        save_dir=OUTPUT_DIR,
        experiment_name="section6_mixup"
    )
    
    histories.append(history_mixup)
    labels.append("Mixup")
    results["Mixup"] = get_final_results(history_mixup)
    
    # Comparison plot
    plot_lr_comparison(
        histories, labels,
        title="Mixup Comparison (Section 6)",
        save_path=os.path.join(OUTPUT_DIR, "section6_comparison.png"),
        show=False
    )
    
    # Print results table
    print_results_table(results)
    
    # Save results
    save_results(results, os.path.join(OUTPUT_DIR, "section6_results.json"))
    
    return results


def run_all_experiments():
    """
    Run all experiments in sequence
    """
    print("\n" + "="*80)
    print("AI6103 HYPERPARAMETER STUDY - FULL EXPERIMENT RUN")
    print(f"Device: {DEVICE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Section 2: Data preprocessing
    mean, std = run_section2()
    
    # Section 3: Learning rate
    best_lr_name, section3_results = run_section3(mean, std)
    
    # Section 4: Learning rate schedule
    section4_results = run_section4(mean, std, best_lr_name)
    
    # Section 5: Weight decay
    section5_results = run_section5(mean, std, best_lr_name)
    
    # Section 6: Mixup
    section6_results = run_section6(mean, std, best_lr_name)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")


if __name__ == "__main__":
    run_all_experiments()
