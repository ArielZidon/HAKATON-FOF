"""
Utility functions for data loading and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from pathlib import Path


def load_csv_grids(csv_files):
    """
    Load multiple CSV files as training grids.

    Args:
        csv_files: List of CSV file paths. Each CSV should have columns:
                  'latitude' (or 'x'), 'longitude' (or 'y'), 'class' (or 'label')

    Returns:
        List of tuples (grid_name, points) where points is array of (x, y, label)
    """
    grids = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Determine column names (flexible)
        x_col = 'longitude' if 'longitude' in df.columns else 'x'
        y_col = 'latitude' if 'latitude' in df.columns else 'y'
        label_col = 'class' if 'class' in df.columns else 'label'

        if x_col not in df.columns or y_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"CSV file {csv_file} must contain coordinate and class columns")

        # Extract data
        x = df[x_col].values
        y = df[y_col].values
        labels = df[label_col].values

        # Combine into (x, y, label) format
        points = np.column_stack([x, y, labels])

        # Use filename as grid name
        grid_name = Path(csv_file).stem

        grids.append((grid_name, points))
        print(f"Loaded grid '{grid_name}': {len(points)} points from {csv_file}")

    return grids


def split_labeled_unlabeled(points, labeled_fraction=0.2, random_state=42):
    """
    Split a grid into labeled and unlabeled points.

    Args:
        points: Array of (x, y, label)
        labeled_fraction: Fraction of points to keep as labeled
        random_state: Random seed

    Returns:
        Tuple of (labeled_points, unlabeled_points, true_labels_for_unlabeled)
    """
    np.random.seed(random_state)

    n_points = len(points)
    n_labeled = int(labeled_fraction * n_points)

    # Randomly select labeled indices
    indices = np.arange(n_points)
    np.random.shuffle(indices)

    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]

    labeled_points = points[labeled_indices]
    unlabeled_coords = points[unlabeled_indices, :2]
    true_labels_for_unlabeled = points[unlabeled_indices, 2].astype(int)

    return labeled_points, unlabeled_coords, true_labels_for_unlabeled


def visualize_training_grids(grids, max_grids=8):
    """
    Visualize multiple training grids in a grid layout.

    Args:
        grids: List of tuples (grid_name, points) where points is (x, y, label)
        max_grids: Maximum number of grids to visualize
    """
    n_grids = min(len(grids), max_grids)
    n_cols = min(4, n_grids)
    n_rows = (n_grids + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_grids == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cmap = ListedColormap(colors)

    for idx, (grid_name, points) in enumerate(grids[:max_grids]):
        ax = axes[idx]
        points = np.array(points)
        coords = points[:, :2]
        labels = points[:, 2].astype(int)

        # Remap labels to 0, 1, 2 for coloring
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
        mapped_labels = np.array([label_map[l] for l in labels])

        ax.scatter(coords[:, 0], coords[:, 1], c=mapped_labels,
                   cmap=cmap, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax.set_title(f"{grid_name}\n({len(labels)} points)", fontsize=10, fontweight='bold')
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_grids, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle('Training Grids', fontsize=14, fontweight='bold', y=1.00)
    plt.show()


def visualize_prediction(unlabeled_points, predictions, labeled_points=None,
                        true_labels=None, title="Grid Predictions"):
    """
    Visualize predictions on a new grid.

    Args:
        unlabeled_points: Array of (x, y) coordinates that were predicted
        predictions: Predicted labels for unlabeled_points
        labeled_points: Optional array of (x, y, label) for known points in the grid
        true_labels: Optional true labels for unlabeled points (for evaluation)
        title: Plot title
    """
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Remap predictions for consistent coloring
    unique_preds = np.unique(predictions)
    pred_map = {old: new for new, old in enumerate(sorted(unique_preds))}
    mapped_predictions = np.array([pred_map[p] for p in predictions])

    # Plot predicted points
    scatter1 = ax.scatter(unlabeled_points[:, 0], unlabeled_points[:, 1],
                         c=mapped_predictions, cmap=cmap, s=100, alpha=0.6,
                         edgecolors='black', linewidth=1, label='Predicted', marker='o')

    # Plot labeled points if provided
    if labeled_points is not None and len(labeled_points) > 0:
        labeled_points = np.array(labeled_points)
        labeled_coords = labeled_points[:, :2]
        labeled_classes = labeled_points[:, 2].astype(int)

        # Remap labeled classes
        unique_labeled = np.unique(labeled_classes)
        label_map = {old: new for new, old in enumerate(sorted(unique_labeled))}
        mapped_labeled = np.array([label_map[l] for l in labeled_classes])

        ax.scatter(labeled_coords[:, 0], labeled_coords[:, 1],
                  c=mapped_labeled, cmap=cmap, s=150, alpha=0.9,
                  edgecolors='gold', linewidth=2, label='Known (Labeled)', marker='s')

    # If true labels provided, highlight errors
    if true_labels is not None:
        errors = true_labels != predictions
        if np.any(errors):
            ax.scatter(unlabeled_points[errors, 0], unlabeled_points[errors, 1],
                      facecolors='none', edgecolors='red', s=200, linewidths=3,
                      label='Errors', marker='o')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(scatter1, ax=ax)
    cbar.set_label('Class', fontsize=12)

    plt.tight_layout()
    plt.show()


def visualize_comparison(coords, true_labels, predictions, title="True vs Predicted"):
    """
    Visualize true labels vs predictions side by side.

    Args:
        coords: Array of (x, y) coordinates
        true_labels: True labels
        predictions: Predicted labels
        title: Plot title
    """
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Remap labels for consistent coloring
    all_labels = np.unique(np.concatenate([true_labels, predictions]))
    label_map = {old: new for new, old in enumerate(sorted(all_labels))}

    mapped_true = np.array([label_map[l] for l in true_labels])
    mapped_pred = np.array([label_map[p] for p in predictions])

    # True labels
    axes[0].scatter(coords[:, 0], coords[:, 1], c=mapped_true,
                   cmap=cmap, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0].set_title('True Labels', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)

    # Predictions
    axes[1].scatter(coords[:, 0], coords[:, 1], c=mapped_pred,
                   cmap=cmap, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1].set_title('Predictions', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)

    # Errors
    errors = true_labels != predictions
    accuracy = np.mean(true_labels == predictions)

    axes[2].scatter(coords[~errors, 0], coords[~errors, 1],
                   c='lightgreen', s=50, alpha=0.5, label='Correct')
    axes[2].scatter(coords[errors, 0], coords[errors, 1],
                   c='red', s=100, alpha=0.8, marker='X', label='Error')
    axes[2].set_title(f'Errors (Accuracy: {accuracy:.4f})', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Synthetic data generation removed - using real field data only
