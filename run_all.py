#!/usr/bin/env python3
"""
Main script to run all AI6103 experiments
Usage: python run_all.py [--section SECTION]
"""
import argparse
import os
from config import OUTPUT_DIR
from data import compute_dataset_stats, get_dataloaders
from experiments import (
    run_section2, run_section3, run_section4, 
    run_section5, run_section6, run_all_experiments
)


def main():
    parser = argparse.ArgumentParser(description='AI6103 Hyperparameter Study')
    parser.add_argument('--section', type=int, choices=[2, 3, 4, 5, 6],
                        help='Run only a specific section')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset first')
    args = parser.parse_args()
    
    # Download data if requested
    if args.download:
        from download_data import download_food11
        download_food11()
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if args.section:
        # Run specific section
        # First get dataset stats
        from config import DATA_DIR
        train_path = os.path.join(DATA_DIR, 'training')
        mean, std = compute_dataset_stats(train_path)
        
        if args.section == 2:
            run_section2()
        elif args.section == 3:
            run_section3(mean, std)
        elif args.section == 4:
            # Need best LR from section 3
            best_lr_name = "lr_0.025"  # Default, should be updated from section 3 results
            run_section4(mean, std, best_lr_name)
        elif args.section == 5:
            best_lr_name = "lr_0.025"
            run_section5(mean, std, best_lr_name)
        elif args.section == 6:
            best_lr_name = "lr_0.025"
            run_section6(mean, std, best_lr_name)
    else:
        # Run all experiments
        run_all_experiments()


if __name__ == "__main__":
    main()
