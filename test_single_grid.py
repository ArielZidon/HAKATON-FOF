"""
Cross-Validation Testing for Single Grid with Mixed Classes

This script performs k-fold cross-validation when you have a single grid file
containing multiple classes. It uses a 75-25 train-test split.
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from model import Grid2DClassifier
from utils import load_csv_grids, visualize_comparison
import matplotlib.pyplot as plt
import seaborn as sns


def perform_kfold_cv(grid_file, n_splits=4, labeled_fraction=0.2,
                     model_type='random_forest', n_estimators=150,
                     visualize=True):
    """
    Perform k-fold cross-validation on a single grid file.

    Args:
        grid_file: Path to CSV file containing the grid
        n_splits: Number of folds for cross-validation
        labeled_fraction: Fraction of test points to use as "known" labels
        model_type: Type of model to use
        n_estimators: Number of estimators
        visualize: Whether to show visualizations

    Returns:
        Dictionary with results for each fold
    """
    print("="*60)
    print("K-FOLD CROSS-VALIDATION TEST (Single Grid)")
    print("="*60)
    print(f"Grid file: {grid_file}")
    print(f"Model type: {model_type}")
    print(f"Number of folds: {n_splits}")
    print(f"Train-Test ratio: {(n_splits-1)/n_splits:.2%}-{1/n_splits:.2%}")
    print(f"Labeled fraction for test: {labeled_fraction}")
    print("="*60)

    # Load the grid
    grids = load_csv_grids([grid_file])
    grid_name, grid_points = grids[0]

    print(f"\nLoaded grid '{grid_name}': {len(grid_points)} points")

    # Extract coordinates and labels
    coords = grid_points[:, :2]
    labels = grid_points[:, 2].astype(int)

    # Show class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} points ({count/len(labels)*100:.1f}%)")

    # Perform stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(coords, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_splits}")
        print(f"{'='*60}")

        # Split data
        train_coords = coords[train_idx]
        train_labels = labels[train_idx]
        test_coords = coords[test_idx]
        test_labels = labels[test_idx]

        print(f"Training set: {len(train_coords)} points")
        print(f"Test set: {len(test_coords)} points")

        # Show class distribution in splits
        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        print(f"\nTrain class distribution: ", end="")
        for lbl, cnt in zip(train_unique, train_counts):
            print(f"Class {lbl}: {cnt} ({cnt/len(train_labels)*100:.1f}%) ", end="")
        print(f"\nTest class distribution: ", end="")
        for lbl, cnt in zip(test_unique, test_counts):
            print(f"Class {lbl}: {cnt} ({cnt/len(test_labels)*100:.1f}%) ", end="")
        print()

        # Create training grid
        train_grid_points = np.column_stack([train_coords, train_labels])
        training_grids = [(f'train_fold_{fold_idx+1}', train_grid_points)]

        # Train classifier
        classifier = Grid2DClassifier(model_type=model_type)
        classifier.add_training_grids(training_grids)
        print("\nTraining model...")
        classifier.train(n_estimators=n_estimators, random_state=42)

        # Simulate partial labeling on test set
        n_test = len(test_coords)
        n_labeled = int(labeled_fraction * n_test)
        n_unlabeled = n_test - n_labeled

        # Randomly select labeled points (stratified by class)
        np.random.seed(42 + fold_idx)
        labeled_indices = []
        for label in test_unique:
            label_mask = test_labels == label
            label_indices = np.where(label_mask)[0]
            n_label_to_select = max(1, int(labeled_fraction * len(label_indices)))
            selected = np.random.choice(label_indices, n_label_to_select, replace=False)
            labeled_indices.extend(selected)

        unlabeled_indices = np.setdiff1d(np.arange(n_test), labeled_indices)

        labeled_coords = test_coords[labeled_indices]
        labeled_classes = test_labels[labeled_indices]
        labeled_points = np.column_stack([labeled_coords, labeled_classes])

        unlabeled_coords = test_coords[unlabeled_indices]
        true_labels_unlabeled = test_labels[unlabeled_indices]

        print(f"\nTest set split:")
        print(f"  - Labeled (known): {len(labeled_points)} points")
        print(f"  - Unlabeled (predict): {len(unlabeled_coords)} points")

        # Predict without refinement
        print("\nPredicting without refinement...")
        predictions_no_refine = classifier.predict_grid(
            unlabeled_coords,
            labeled_points=None,
            refine_with_labeled=False
        )
        accuracy_no_refine = accuracy_score(true_labels_unlabeled, predictions_no_refine)

        # Predict with refinement
        print("Predicting with refinement using labeled points...")
        predictions_with_refine = classifier.predict_grid(
            unlabeled_coords,
            labeled_points=labeled_points,
            refine_with_labeled=True
        )
        accuracy_with_refine = accuracy_score(true_labels_unlabeled, predictions_with_refine)

        print(f"\nResults for Fold {fold_idx + 1}:")
        print(f"  Accuracy (no refinement):   {accuracy_no_refine:.4f}")
        print(f"  Accuracy (with refinement): {accuracy_with_refine:.4f}")
        print(f"  Improvement: {accuracy_with_refine - accuracy_no_refine:+.4f}")

        # Confusion matrix
        cm_with_refine = confusion_matrix(true_labels_unlabeled, predictions_with_refine)
        print(f"\nConfusion Matrix (with refinement):")
        print(cm_with_refine)

        # Store results
        results[f'fold_{fold_idx+1}'] = {
            'fold': fold_idx + 1,
            'accuracy_no_refine': accuracy_no_refine,
            'accuracy_with_refine': accuracy_with_refine,
            'predictions_no_refine': predictions_no_refine,
            'predictions_with_refine': predictions_with_refine,
            'true_labels': true_labels_unlabeled,
            'unlabeled_coords': unlabeled_coords,
            'labeled_points': labeled_points,
            'confusion_matrix': cm_with_refine,
            'n_train': len(train_coords),
            'n_test_labeled': len(labeled_points),
            'n_test_unlabeled': len(unlabeled_coords)
        }

        # Visualize if requested
        if visualize:
            visualize_fold_results(
                unlabeled_coords, true_labels_unlabeled,
                predictions_no_refine, predictions_with_refine,
                labeled_points, fold_idx + 1,
                accuracy_no_refine, accuracy_with_refine
            )

    return results


def visualize_fold_results(unlabeled_coords, true_labels, pred_no_refine,
                           pred_with_refine, labeled_points, fold_num,
                           acc_no_refine, acc_with_refine):
    """Visualize results for a single fold."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors_map = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}

    # Map labels to colors
    def get_colors(labels):
        return [colors_map.get(int(l), '#888888') for l in labels]

    # True labels
    axes[0].scatter(unlabeled_coords[:, 0], unlabeled_coords[:, 1],
                   c=get_colors(true_labels), s=80, alpha=0.7,
                   edgecolors='black', linewidth=0.5)
    if len(labeled_points) > 0:
        axes[0].scatter(labeled_points[:, 0], labeled_points[:, 1],
                       c=get_colors(labeled_points[:, 2]), s=150, alpha=0.9,
                       marker='s', edgecolors='gold', linewidth=2, label='Known')
    axes[0].set_title(f'True Labels', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Predictions without refinement
    errors_no = true_labels != pred_no_refine
    axes[1].scatter(unlabeled_coords[~errors_no, 0], unlabeled_coords[~errors_no, 1],
                   c='lightgreen', s=80, alpha=0.6, label='Correct')
    axes[1].scatter(unlabeled_coords[errors_no, 0], unlabeled_coords[errors_no, 1],
                   c='red', s=120, alpha=0.9, marker='X', label='Error')
    if len(labeled_points) > 0:
        axes[1].scatter(labeled_points[:, 0], labeled_points[:, 1],
                       c='gold', s=120, alpha=0.7, marker='s',
                       edgecolors='black', linewidth=2, label='Known')
    axes[1].set_title(f'Without Refinement\nAcc: {acc_no_refine:.4f}',
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Predictions with refinement
    errors_with = true_labels != pred_with_refine
    axes[2].scatter(unlabeled_coords[~errors_with, 0], unlabeled_coords[~errors_with, 1],
                   c='lightgreen', s=80, alpha=0.6, label='Correct')
    axes[2].scatter(unlabeled_coords[errors_with, 0], unlabeled_coords[errors_with, 1],
                   c='red', s=120, alpha=0.9, marker='X', label='Error')
    if len(labeled_points) > 0:
        axes[2].scatter(labeled_points[:, 0], labeled_points[:, 1],
                       c='gold', s=120, alpha=0.7, marker='s',
                       edgecolors='black', linewidth=2, label='Known')
    axes[2].set_title(f'With Refinement\nAcc: {acc_with_refine:.4f}',
                     fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Fold {fold_num} Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def print_summary(results):
    """Print summary of cross-validation results."""
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)

    # Create summary table
    summary_data = []
    for fold_name, result in results.items():
        summary_data.append({
            'Fold': result['fold'],
            'Train': result['n_train'],
            'Test (Labeled)': result['n_test_labeled'],
            'Test (Unlabeled)': result['n_test_unlabeled'],
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
    print("AVERAGE RESULTS ACROSS ALL FOLDS")
    print("="*60)
    print(f"Average Accuracy (No Refinement):   {avg_no_refine:.4f}")
    print(f"Average Accuracy (With Refinement): {avg_with_refine:.4f}")
    print(f"Average Improvement:                {avg_improvement:+.4f}")

    # Standard deviations
    std_no_refine = np.std([r['accuracy_no_refine'] for r in results.values()])
    std_with_refine = np.std([r['accuracy_with_refine'] for r in results.values()])
    print(f"\nStd Dev (No Refinement):   {std_no_refine:.4f}")
    print(f"Std Dev (With Refinement): {std_with_refine:.4f}")
    print("="*60)

    # Plot summary
    fold_nums = [results[f]['fold'] for f in sorted(results.keys())]
    acc_no_refine = [results[f]['accuracy_no_refine'] for f in sorted(results.keys())]
    acc_with_refine = [results[f]['accuracy_with_refine'] for f in sorted(results.keys())]

    x = np.arange(len(fold_nums))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, acc_no_refine, width, label='No Refinement', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, acc_with_refine, width, label='With Refinement', alpha=0.8, color='lightcoral')

    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('K-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in fold_nums])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Add average lines
    ax.axhline(y=avg_no_refine, color='blue', linestyle='--', linewidth=2,
              label=f'Avg No Refine: {avg_no_refine:.3f}', alpha=0.7)
    ax.axhline(y=avg_with_refine, color='red', linestyle='--', linewidth=2,
              label=f'Avg With Refine: {avg_with_refine:.3f}', alpha=0.7)

    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='K-Fold CV test for single grid')

    parser.add_argument('--grid-file', type=str, default='coordinates/field1.csv',
                       help='CSV file containing the grid')
    parser.add_argument('--n-splits', type=int, default=4,
                       help='Number of folds (default: 4 for 75-25 split)')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'neural_network'],
                       help='Type of model to use')
    parser.add_argument('--n-estimators', type=int, default=150,
                       help='Number of estimators')
    parser.add_argument('--labeled-fraction', type=float, default=0.2,
                       help='Fraction of test points to use as labeled')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualizations')

    args = parser.parse_args()

    # Run k-fold cross-validation
    results = perform_kfold_cv(
        grid_file=args.grid_file,
        n_splits=args.n_splits,
        labeled_fraction=args.labeled_fraction,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        visualize=not args.no_visualize
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    # Run with default settings: 4-fold CV (75-25 split)
    print("Running 4-Fold Cross-Validation on field1.csv...\n")

    results = perform_kfold_cv(
        grid_file='coordinates/field1.csv',
        n_splits=4,  # 75-25 split
        labeled_fraction=0.2,
        model_type='random_forest',
        n_estimators=150,
        visualize=True
    )

    print_summary(results)
