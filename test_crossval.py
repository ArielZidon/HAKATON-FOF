"""
Cross-Validation Testing for Grid2D Classifier

This script performs cross-validation testing using the CSV files in the coordinates directory.
It trains on all grids except one (leave-one-out), then tests on the held-out grid.
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from model import Grid2DClassifier
from utils import load_csv_grids, visualize_comparison, split_labeled_unlabeled
import matplotlib.pyplot as plt
import seaborn as sns


def perform_leave_one_out_cv(grid_files, model_type='random_forest',
                              n_estimators=150, labeled_fraction=0.2,
                              visualize=True):
    """
    Perform leave-one-out cross-validation on the provided grids.

    For each grid:
    1. Train on all other grids
    2. Simulate partial labeling (only keep labeled_fraction of points as labeled)
    3. Predict the remaining unlabeled points
    4. Evaluate accuracy

    Args:
        grid_files: List of CSV file paths
        model_type: Type of model to use
        n_estimators: Number of estimators
        labeled_fraction: Fraction of test grid points to use as "known" labels
        visualize: Whether to show visualizations

    Returns:
        Dictionary with results for each test grid
    """
    print("="*60)
    print("LEAVE-ONE-OUT CROSS-VALIDATION TEST")
    print("="*60)
    print(f"Grid files: {len(grid_files)}")
    print(f"Model type: {model_type}")
    print(f"Labeled fraction for test grids: {labeled_fraction}")
    print("="*60)

    # Load all grids
    all_grids = load_csv_grids(grid_files)

    results = {}

    for test_idx, (test_grid_name, test_grid_points) in enumerate(all_grids):
        print(f"\n{'='*60}")
        print(f"Test {test_idx + 1}/{len(all_grids)}: Testing on '{test_grid_name}'")
        print(f"{'='*60}")

        # Prepare training grids (all except test grid)
        training_grids = [g for i, g in enumerate(all_grids) if i != test_idx]

        print(f"Training grids: {[g[0] for g in training_grids]}")

        # Create and train classifier
        classifier = Grid2DClassifier(model_type=model_type)
        classifier.add_training_grids(training_grids)
        classifier.train(n_estimators=n_estimators, random_state=42)

        # Split test grid into labeled and unlabeled
        labeled_points, unlabeled_coords, true_labels = split_labeled_unlabeled(
            test_grid_points,
            labeled_fraction=labeled_fraction,
            random_state=42
        )

        print(f"\nTest grid '{test_grid_name}':")
        print(f"  - Total points: {len(test_grid_points)}")
        print(f"  - Labeled (known) points: {len(labeled_points)}")
        print(f"  - Unlabeled (to predict) points: {len(unlabeled_coords)}")

        # Predict unlabeled points (with and without refinement)
        print("\nPredicting without refinement...")
        predictions_no_refine = classifier.predict_grid(
            unlabeled_coords,
            labeled_points=None,
            refine_with_labeled=False
        )
        accuracy_no_refine = accuracy_score(true_labels, predictions_no_refine)

        print("Predicting with refinement using labeled points...")
        predictions_with_refine = classifier.predict_grid(
            unlabeled_coords,
            labeled_points=labeled_points,
            refine_with_labeled=True
        )
        accuracy_with_refine = accuracy_score(true_labels, predictions_with_refine)

        print(f"\nResults for '{test_grid_name}':")
        print(f"  Accuracy (no refinement):   {accuracy_no_refine:.4f}")
        print(f"  Accuracy (with refinement): {accuracy_with_refine:.4f}")
        print(f"  Improvement: {accuracy_with_refine - accuracy_no_refine:+.4f}")

        # Store results
        results[test_grid_name] = {
            'accuracy_no_refine': accuracy_no_refine,
            'accuracy_with_refine': accuracy_with_refine,
            'predictions_no_refine': predictions_no_refine,
            'predictions_with_refine': predictions_with_refine,
            'true_labels': true_labels,
            'unlabeled_coords': unlabeled_coords,
            'labeled_points': labeled_points,
            'n_total': len(test_grid_points),
            'n_labeled': len(labeled_points),
            'n_unlabeled': len(unlabeled_coords)
        }

        # Visualize if requested
        if visualize:
            # Visualization 1: With refinement
            visualize_comparison(
                unlabeled_coords,
                true_labels,
                predictions_with_refine,
                title=f"{test_grid_name} - With Refinement (Acc: {accuracy_with_refine:.4f})"
            )

            # Visualization 2: Comparison of refinement effect
            if accuracy_with_refine != accuracy_no_refine:
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # Without refinement
                errors_no = true_labels != predictions_no_refine
                axes[0].scatter(unlabeled_coords[~errors_no, 0], unlabeled_coords[~errors_no, 1],
                              c='lightgreen', s=50, alpha=0.5, label='Correct')
                axes[0].scatter(unlabeled_coords[errors_no, 0], unlabeled_coords[errors_no, 1],
                              c='red', s=100, alpha=0.8, marker='X', label='Error')
                # Show labeled points
                axes[0].scatter(labeled_points[:, 0], labeled_points[:, 1],
                              c='gold', s=100, alpha=0.7, marker='s', edgecolors='black',
                              linewidth=2, label='Known Labels')
                axes[0].set_title(f'Without Refinement\nAcc: {accuracy_no_refine:.4f}',
                                fontsize=12, fontweight='bold')
                axes[0].set_xlabel('X')
                axes[0].set_ylabel('Y')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # With refinement
                errors_with = true_labels != predictions_with_refine
                axes[1].scatter(unlabeled_coords[~errors_with, 0], unlabeled_coords[~errors_with, 1],
                              c='lightgreen', s=50, alpha=0.5, label='Correct')
                axes[1].scatter(unlabeled_coords[errors_with, 0], unlabeled_coords[errors_with, 1],
                              c='red', s=100, alpha=0.8, marker='X', label='Error')
                # Show labeled points
                axes[1].scatter(labeled_points[:, 0], labeled_points[:, 1],
                              c='gold', s=100, alpha=0.7, marker='s', edgecolors='black',
                              linewidth=2, label='Known Labels')
                axes[1].set_title(f'With Refinement\nAcc: {accuracy_with_refine:.4f}',
                                fontsize=12, fontweight='bold')
                axes[1].set_xlabel('X')
                axes[1].set_ylabel('Y')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plt.suptitle(f'{test_grid_name} - Refinement Comparison', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()

    return results


def print_summary(results):
    """Print a summary of cross-validation results."""
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)

    # Create summary table
    summary_data = []
    for grid_name, result in results.items():
        summary_data.append({
            'Grid': grid_name,
            'Total Points': result['n_total'],
            'Labeled': result['n_labeled'],
            'Unlabeled': result['n_unlabeled'],
            'Acc (No Refine)': f"{result['accuracy_no_refine']:.4f}",
            'Acc (With Refine)': f"{result['accuracy_with_refine']:.4f}",
            'Improvement': f"{result['accuracy_with_refine'] - result['accuracy_no_refine']:+.4f}"
        })

    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))

    # Calculate averages
    avg_no_refine = np.mean([r['accuracy_no_refine'] for r in results.values()])
    avg_with_refine = np.mean([r['accuracy_with_refine'] for r in results.values()])
    avg_improvement = avg_with_refine - avg_no_refine

    print("\n" + "="*60)
    print("AVERAGE RESULTS")
    print("="*60)
    print(f"Average Accuracy (No Refinement):   {avg_no_refine:.4f}")
    print(f"Average Accuracy (With Refinement): {avg_with_refine:.4f}")
    print(f"Average Improvement:                {avg_improvement:+.4f}")
    print("="*60)

    # Plot summary
    grid_names = list(results.keys())
    acc_no_refine = [results[g]['accuracy_no_refine'] for g in grid_names]
    acc_with_refine = [results[g]['accuracy_with_refine'] for g in grid_names]

    x = np.arange(len(grid_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, acc_no_refine, width, label='No Refinement', alpha=0.8)
    bars2 = ax.bar(x + width/2, acc_with_refine, width, label='With Refinement', alpha=0.8)

    ax.set_xlabel('Test Grid', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Cross-Validation Results by Grid', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grid_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Cross-validation test for Grid2D Classifier')

    parser.add_argument('--grid-files', nargs='+', type=str,
                       help='CSV files to use for cross-validation')
    parser.add_argument('--coordinates-dir', type=str, default='coordinates',
                       help='Directory containing coordinate CSV files')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'neural_network'],
                       help='Type of model to use')
    parser.add_argument('--n-estimators', type=int, default=150,
                       help='Number of estimators')
    parser.add_argument('--labeled-fraction', type=float, default=0.2,
                       help='Fraction of test grid points to use as labeled')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualizations')

    args = parser.parse_args()

    # Determine grid files
    if args.grid_files:
        grid_files = args.grid_files
    else:
        # Use all CSV files in coordinates directory
        coords_dir = Path(args.coordinates_dir)
        if not coords_dir.exists():
            print(f"Error: Coordinates directory '{coords_dir}' not found!")
            return

        grid_files = list(coords_dir.glob('*.csv'))
        if not grid_files:
            print(f"Error: No CSV files found in '{coords_dir}'!")
            return

        grid_files = [str(f) for f in grid_files]

    print(f"Found {len(grid_files)} grid files:")
    for gf in grid_files:
        print(f"  - {gf}")

    # Run cross-validation
    results = perform_leave_one_out_cv(
        grid_files=grid_files,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        labeled_fraction=args.labeled_fraction,
        visualize=not args.no_visualize
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    # Example: run on coordinates directory
    print("Running cross-validation on coordinates directory...\n")

    coords_dir = Path('coordinates')
    if coords_dir.exists():
        grid_files = [str(f) for f in coords_dir.glob('*.csv')]
        if grid_files:
            results = perform_leave_one_out_cv(
                grid_files=grid_files,
                model_type='random_forest',
                n_estimators=150,
                labeled_fraction=0.3,  # Use 30% as labeled
                visualize=True
            )
            print_summary(results)
        else:
            print("No CSV files found in coordinates directory!")
    else:
        print("Coordinates directory not found! Please run from the project root.")
