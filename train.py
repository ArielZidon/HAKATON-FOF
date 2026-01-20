"""
Training script for Grid2D Classifier

This script trains a classifier on multiple grids.
"""

import argparse
from pathlib import Path
from model import Grid2DClassifier
from utils import load_csv_grids, visualize_training_grids, generate_synthetic_grids
import pickle


def train_classifier(grid_files=None, model_type='random_forest',
                     n_estimators=150, save_path=None, visualize=True,
                     use_synthetic=False, n_synthetic_grids=5):
    """
    Train a Grid2D classifier.

    Args:
        grid_files: List of CSV file paths to use as training grids
        model_type: Type of model ('random_forest', 'gradient_boosting', 'neural_network')
        n_estimators: Number of estimators (for RF and GB)
        save_path: Path to save the trained model
        visualize: Whether to visualize training grids
        use_synthetic: Whether to generate synthetic data
        n_synthetic_grids: Number of synthetic grids to generate

    Returns:
        Trained classifier
    """
    print("="*60)
    print("GRID 2D CLASSIFIER - TRAINING")
    print("="*60)

    # Load or generate training data
    if use_synthetic:
        print(f"\nGenerating {n_synthetic_grids} synthetic training grids...")
        training_grids = generate_synthetic_grids(
            n_grids=n_synthetic_grids,
            n_points_per_grid=300,
            random_state=42
        )
    else:
        if not grid_files:
            raise ValueError("Must provide grid_files or set use_synthetic=True")
        print(f"\nLoading training grids from {len(grid_files)} files...")
        training_grids = load_csv_grids(grid_files)

    # Visualize if requested
    if visualize:
        print("\nVisualizing training grids...")
        visualize_training_grids(training_grids)

    # Create and train classifier
    print(f"\nCreating {model_type} classifier...")
    classifier = Grid2DClassifier(model_type=model_type)
    classifier.add_training_grids(training_grids)

    print(f"\nTraining model...")
    training_results = classifier.train(
        n_estimators=n_estimators,
        random_state=42
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total training samples: {training_results['n_samples']}")
    print(f"Class distribution: {training_results['class_distribution']}")

    # Display model info
    info = classifier.get_model_info()
    print(f"\nModel Information:")
    print(f"  Model type: {info['model_type']}")
    print(f"  Training grids: {info['n_training_grids']}")
    print(f"  Total training points: {info['total_training_points']}")

    # Save model if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(classifier, f)
        print(f"\nModel saved to: {save_path}")

    return classifier


def main():
    parser = argparse.ArgumentParser(description='Train a Grid2D Classifier')

    parser.add_argument('--grid-files', nargs='+', type=str,
                       help='CSV files to use as training grids')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'neural_network'],
                       help='Type of model to train')
    parser.add_argument('--n-estimators', type=int, default=150,
                       help='Number of estimators (for RF and GB)')
    parser.add_argument('--save-path', type=str, default='trained_model.pkl',
                       help='Path to save the trained model')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic data for training')
    parser.add_argument('--n-synthetic-grids', type=int, default=5,
                       help='Number of synthetic grids to generate')

    args = parser.parse_args()

    # Train classifier
    classifier = train_classifier(
        grid_files=args.grid_files,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        save_path=args.save_path,
        visualize=not args.no_visualize,
        use_synthetic=args.use_synthetic,
        n_synthetic_grids=args.n_synthetic_grids
    )


if __name__ == "__main__":
    # Example: train on synthetic data
    print("Example training run with synthetic data...\n")

    classifier = train_classifier(
        use_synthetic=True,
        n_synthetic_grids=5,
        model_type='random_forest',
        n_estimators=150,
        save_path='models/trained_model.pkl',
        visualize=True
    )
