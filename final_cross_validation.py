"""
Final Cross-Validation on 22 Real Field Files

This script:
1. Runs leave-one-out cross-validation on all 22 field files
2. Tests multiple model types (Random Forest, Gradient Boosting, Neural Network)
3. Compares models and selects the best one
4. Saves the winning model
5. Generates comprehensive results
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

from model import Grid2DClassifier
from utils import load_csv_grids


def run_cross_validation_all_models(grid_files, model_types, n_estimators=150):
    """
    Run leave-one-out cross-validation for multiple model types.

    Args:
        grid_files: List of grid CSV files
        model_types: List of model types to test
        n_estimators: Number of estimators for ensemble models

    Returns:
        Dictionary with results for each model type
    """
    print("="*70)
    print("CROSS-VALIDATION: COMPARING MODEL TYPES")
    print("="*70)
    print(f"Grid files: {len(grid_files)}")
    print(f"Model types: {model_types}")
    print(f"Strategy: Leave-One-Out Cross-Validation")
    print("="*70)

    # Load all grids once
    all_grids = load_csv_grids(grid_files)
    print(f"\nLoaded {len(all_grids)} grids")

    # Calculate total points
    total_points = sum(len(g[1]) for g in all_grids)
    print(f"Total data points: {total_points}")

    results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Testing Model: {model_type.upper()}")
        print(f"{'='*70}")

        start_time = time.time()
        fold_results = []

        # Leave-one-out cross-validation
        for test_idx, (test_grid_name, test_grid_points) in enumerate(all_grids):
            # Training grids (all except test grid)
            training_grids = [g for i, g in enumerate(all_grids) if i != test_idx]

            # Create and train classifier
            classifier = Grid2DClassifier(model_type=model_type)
            classifier.add_training_grids(training_grids)
            classifier.train(n_estimators=n_estimators, random_state=42)

            # Test on held-out grid
            test_coords = test_grid_points[:, :2]
            test_labels = test_grid_points[:, 2].astype(int)

            # Predict
            predictions = classifier.predict_entire_grid(test_coords)

            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)

            fold_results.append({
                'fold': test_idx + 1,
                'test_grid': test_grid_name,
                'n_test_points': len(test_labels),
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': predictions,
                'true_labels': test_labels
            })

            print(f"  Fold {test_idx+1:2d}/{len(all_grids)} - {test_grid_name:12s}: "
                  f"Acc={accuracy:.4f}, F1={f1:.4f}, N={len(test_labels):2d}")

        elapsed_time = time.time() - start_time

        # Calculate overall statistics
        accuracies = [r['accuracy'] for r in fold_results]
        f1_scores = [r['f1_score'] for r in fold_results]

        results[model_type] = {
            'fold_results': fold_results,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'training_time': elapsed_time
        }

        print(f"\n  Summary for {model_type}:")
        print(f"    Mean Accuracy: {results[model_type]['mean_accuracy']:.4f} ± {results[model_type]['std_accuracy']:.4f}")
        print(f"    Mean F1 Score: {results[model_type]['mean_f1']:.4f} ± {results[model_type]['std_f1']:.4f}")
        print(f"    Range: [{results[model_type]['min_accuracy']:.4f}, {results[model_type]['max_accuracy']:.4f}]")
        print(f"    Training Time: {elapsed_time:.2f}s")

    return results, all_grids


def print_comparison_table(results):
    """Print comparison table of all models."""
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)

    data = []
    for model_type, res in results.items():
        data.append({
            'Model': model_type.upper(),
            'Mean Accuracy': f"{res['mean_accuracy']:.4f}",
            'Std Dev': f"{res['std_accuracy']:.4f}",
            'Mean F1': f"{res['mean_f1']:.4f}",
            'Min Acc': f"{res['min_accuracy']:.4f}",
            'Max Acc': f"{res['max_accuracy']:.4f}",
            'Time (s)': f"{res['training_time']:.1f}"
        })

    df = pd.DataFrame(data)
    print("\n" + df.to_string(index=False))

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['mean_accuracy'])
    print(f"\n🏆 BEST MODEL: {best_model[0].upper()}")
    print(f"   Mean Accuracy: {best_model[1]['mean_accuracy']:.4f}")
    print(f"   Mean F1 Score: {best_model[1]['mean_f1']:.4f}")

    return best_model[0]


def visualize_results(results):
    """Create comprehensive visualizations."""

    # 1. Accuracy comparison across models
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Mean accuracy comparison
    ax = axes[0, 0]
    model_names = list(results.keys())
    accuracies = [results[m]['mean_accuracy'] for m in model_names]
    std_devs = [results[m]['std_accuracy'] for m in model_names]

    bars = ax.bar(model_names, accuracies, yerr=std_devs, capsize=10, alpha=0.7,
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Mean Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: F1 Score comparison
    ax = axes[0, 1]
    f1_scores = [results[m]['mean_f1'] for m in model_names]
    std_f1 = [results[m]['std_f1'] for m in model_names]

    bars = ax.bar(model_names, f1_scores, yerr=std_f1, capsize=10, alpha=0.7,
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('Mean F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Mean F1 Score', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Per-fold accuracy for best model
    ax = axes[1, 0]
    best_model = max(results.items(), key=lambda x: x[1]['mean_accuracy'])[0]
    fold_accs = [r['accuracy'] for r in results[best_model]['fold_results']]
    fold_nums = range(1, len(fold_accs) + 1)

    ax.plot(fold_nums, fold_accs, 'o-', linewidth=2, markersize=8, color='#4ECDC4', alpha=0.7)
    ax.axhline(y=np.mean(fold_accs), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(fold_accs):.4f}')
    ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Fold Accuracy: {best_model.upper()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1.1])

    # Plot 4: Training time comparison
    ax = axes[1, 1]
    times = [results[m]['training_time'] for m in model_names]
    bars = ax.bar(model_names, times, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Model Training Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{t:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Saved visualization to: cross_validation_results.png")
    plt.show()


def train_and_save_best_model(best_model_type, all_grids, save_path='models/best_model.pkl'):
    """Train the best model on all data and save it."""

    print("\n" + "="*70)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("="*70)
    print(f"Model type: {best_model_type.upper()}")
    print(f"Training on ALL {len(all_grids)} grids")

    # Create classifier
    classifier = Grid2DClassifier(model_type=best_model_type)
    classifier.add_training_grids(all_grids)

    # Train on all data
    training_results = classifier.train(n_estimators=150, random_state=42)

    print(f"\nTraining complete!")
    print(f"Total samples: {training_results['n_samples']}")
    print(f"Class distribution: {training_results['class_distribution']}")

    # Save model
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(classifier, f)

    print(f"\n💾 Model saved to: {save_path}")

    # Get model info
    info = classifier.get_model_info()
    print(f"\nModel Information:")
    print(f"  Model type: {info['model_type']}")
    print(f"  Training grids: {info['n_training_grids']}")
    print(f"  Total training points: {info['total_training_points']}")

    return classifier


def generate_final_report(results, best_model_type, all_grids):
    """Generate final markdown report."""

    report = f"""# Cross-Validation Results - 22 Field Files

## Executive Summary

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Grids**: {len(all_grids)}
**Total Data Points**: {sum(len(g[1]) for g in all_grids)}
**Cross-Validation Method**: Leave-One-Out (LOOCV)
**Best Model**: **{best_model_type.upper()}**

## Model Comparison

| Model | Mean Accuracy | Std Dev | Mean F1 | Min Acc | Max Acc | Time (s) |
|-------|---------------|---------|---------|---------|---------|----------|
"""

    for model_type, res in results.items():
        report += f"| {model_type.upper()} | {res['mean_accuracy']:.4f} | {res['std_accuracy']:.4f} | "
        report += f"{res['mean_f1']:.4f} | {res['min_accuracy']:.4f} | {res['max_accuracy']:.4f} | "
        report += f"{res['training_time']:.1f} |\n"

    report += f"\n## Winner: {best_model_type.upper()}\n\n"
    best_res = results[best_model_type]
    report += f"- **Mean Accuracy**: {best_res['mean_accuracy']:.4f} ± {best_res['std_accuracy']:.4f}\n"
    report += f"- **Mean F1 Score**: {best_res['mean_f1']:.4f} ± {best_res['std_f1']:.4f}\n"
    report += f"- **Accuracy Range**: [{best_res['min_accuracy']:.4f}, {best_res['max_accuracy']:.4f}]\n"
    report += f"- **Training Time**: {best_res['training_time']:.2f} seconds\n"

    report += "\n## Per-Fold Results (Best Model)\n\n"
    report += "| Fold | Test Grid | Points | Accuracy | F1 Score |\n"
    report += "|------|-----------|--------|----------|----------|\n"

    for fold_res in best_res['fold_results']:
        report += f"| {fold_res['fold']:2d} | {fold_res['test_grid']:12s} | "
        report += f"{fold_res['n_test_points']:3d} | {fold_res['accuracy']:.4f} | "
        report += f"{fold_res['f1_score']:.4f} |\n"

    report += f"\n## Saved Model\n\n"
    report += f"- **Location**: `models/best_model.pkl`\n"
    report += f"- **Model Type**: {best_model_type}\n"
    report += f"- **Trained On**: All {len(all_grids)} grids ({sum(len(g[1]) for g in all_grids)} points)\n"

    report += "\n## Usage\n\n"
    report += "```python\n"
    report += "import pickle\n\n"
    report += "# Load model\n"
    report += "with open('models/best_model.pkl', 'rb') as f:\n"
    report += "    classifier = pickle.load(f)\n\n"
    report += "# Predict new points\n"
    report += "new_points = [[35.550, 33.270], [35.548, 33.286]]\n"
    report += "predictions = classifier.predict_grid(new_points)\n"
    report += "print(predictions)  # [0, 1] for example\n"
    report += "```\n"

    # Save report
    with open('FINAL_CV_RESULTS.md', 'w') as f:
        f.write(report)

    print("\n📄 Saved report to: FINAL_CV_RESULTS.md")


def main():
    """Main execution function."""

    # Get all field files
    coords_dir = Path('coordinates')
    grid_files = sorted(coords_dir.glob('field*.csv'), key=lambda x: int(x.stem[5:]))
    grid_files = [str(f) for f in grid_files]

    print(f"Found {len(grid_files)} field files")

    # Model types to compare
    model_types = ['random_forest', 'gradient_boosting', 'neural_network']

    # Run cross-validation for all models
    results, all_grids = run_cross_validation_all_models(grid_files, model_types, n_estimators=150)

    # Print comparison table
    best_model_type = print_comparison_table(results)

    # Visualize results
    visualize_results(results)

    # Train and save best model on all data
    best_classifier = train_and_save_best_model(best_model_type, all_grids)

    # Generate final report
    generate_final_report(results, best_model_type, all_grids)

    print("\n" + "="*70)
    print("✅ CROSS-VALIDATION COMPLETE!")
    print("="*70)
    print(f"\n🏆 Winner: {best_model_type.upper()}")
    print(f"📊 Visualization: cross_validation_results.png")
    print(f"💾 Saved Model: models/best_model.pkl")
    print(f"📄 Report: FINAL_CV_RESULTS.md")
    print("="*70)


if __name__ == "__main__":
    main()
