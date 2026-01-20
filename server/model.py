"""
Grid 2D Classifier Model

This module contains the Grid2DClassifier class for classifying 2D coordinate points.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class Grid2DClassifier:
    """
    A classifier for 2D grids with coordinate-based classification.

    Training workflow:
    1. Train on multiple grids, each with fully labeled points (X, Y, class)
    2. When given a new grid with some labeled points, predict the unlabeled ones

    Supports 3 classes: 0, 1, and 2.

    Example:
        # Training grids (fully labeled)
        grid1 = [(3.23, 7.45, 1), (5.23, 5.0, 0), (2.11, -0.45, 0), (4.23, 0.0, 2)]
        grid2 = [(1.0, 2.0, 0), (3.0, 4.0, 1), (5.0, 6.0, 2)]

        classifier = Grid2DClassifier()
        classifier.add_training_grids([
            ('grid1', grid1),
            ('grid2', grid2)
        ])
        classifier.train()

        # New grid with some labeled and unlabeled points
        new_grid_labeled = [(1.5, 2.5, 0), (4.5, 5.5, 1)]  # Known points
        new_grid_unlabeled = [(2.0, 3.0), (3.5, 4.0)]      # Unknown points

        predictions = classifier.predict_grid(new_grid_unlabeled, new_grid_labeled)
    """

    def __init__(self, model_type='random_forest'):
        """
        Initialize the classifier.

        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'neural_network')
        """
        self.model_type = model_type
        self.scaler = None
        self.model = None
        self.training_grids = []
        self.is_trained = False

    def add_training_grids(self, grids_list):
        """
        Add multiple training grids.

        Args:
            grids_list: List of tuples (grid_name, points)
                       where points is a list of tuples (x, y, label)
                       Example: [('grid1', [(3.23, 7.45, 1), (5.23, 5.0, 0), ...])]
        """
        for grid_name, points in grids_list:
            points = np.array(points)

            # Validate dimensions
            if points.shape[1] != 3:
                raise ValueError(f"Grid '{grid_name}' points must have format (x, y, label). Got {points.shape[1]} columns")

            coordinates = points[:, :2]  # X, Y
            labels = points[:, 2].astype(int)  # Class labels

            # Validate labels are in {0, 1, 2}
            unique_labels = np.unique(labels)
            valid_labels = set(range(len(unique_labels)))

            # Remap labels if needed (e.g., if they're not 0, 1, 2)
            label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
            labels = np.array([label_map[label] for label in labels])

            self.training_grids.append({
                'name': grid_name,
                'coordinates': coordinates,
                'labels': labels,
                'label_map': label_map
            })

            print(f"Added training grid '{grid_name}': {len(labels)} labeled points")

        print(f"\nTotal training grids: {len(self.training_grids)}")

    def _create_model(self, **kwargs):
        """Create a model instance based on model_type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.model_type == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100, 50)),
                max_iter=kwargs.get('max_iter', 500),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, **kwargs):
        """
        Train the model on all training grids combined.

        Args:
            **kwargs: Additional parameters for the model

        Returns:
            Dictionary with training results
        """
        if not self.training_grids:
            raise ValueError("No training grids added. Use add_training_grids() first.")

        # Combine all training grids
        all_coordinates = []
        all_labels = []

        for grid in self.training_grids:
            all_coordinates.append(grid['coordinates'])
            all_labels.append(grid['labels'])

        X = np.vstack(all_coordinates)
        y = np.concatenate(all_labels)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = self._create_model(**kwargs)
        self.model.fit(X_scaled, y)

        self.is_trained = True

        return {
            'n_samples': len(y),
            'class_distribution': {i: int(np.sum(y == i)) for i in np.unique(y)}
        }

    def predict_grid(self, unlabeled_points, labeled_points=None, refine_with_labeled=True):
        """
        Predict labels for unlabeled points in a new grid.

        Optionally uses labeled points from the same grid to refine predictions.

        Args:
            unlabeled_points: List or array of (x, y) coordinates to predict
                             Example: [(2.0, 3.0), (3.5, 4.0), (1.0, 5.0)]
            labeled_points: Optional list of (x, y, label) from the same grid
                           Example: [(1.5, 2.5, 0), (4.5, 5.5, 1)]
            refine_with_labeled: If True and labeled_points provided, fine-tune predictions
                                using the labeled points from this grid

        Returns:
            Array of predicted labels for unlabeled_points
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        unlabeled_points = np.array(unlabeled_points)
        if unlabeled_points.ndim == 1:
            unlabeled_points = unlabeled_points.reshape(1, -1)

        if unlabeled_points.shape[1] != 2:
            raise ValueError(f"Unlabeled points must be 2D (x, y). Got shape {unlabeled_points.shape}")

        # Scale and predict
        unlabeled_scaled = self.scaler.transform(unlabeled_points)
        predictions = self.model.predict(unlabeled_scaled)

        # If labeled points provided and refinement requested, use them to adjust predictions
        if labeled_points is not None and refine_with_labeled and len(labeled_points) > 0:
            predictions = self._refine_predictions_with_context(
                unlabeled_points, predictions, labeled_points
            )

        return predictions

    def _refine_predictions_with_context(self, unlabeled_points, initial_predictions, labeled_points):
        """
        Refine predictions using labeled points from the same grid.
        Uses a simple distance-weighted voting approach.

        Args:
            unlabeled_points: (N, 2) array of points to predict
            initial_predictions: (N,) array of initial predictions
            labeled_points: List of (x, y, label) tuples from the same grid

        Returns:
            Refined predictions
        """
        labeled_points = np.array(labeled_points)
        labeled_coords = labeled_points[:, :2]
        labeled_classes = labeled_points[:, 2].astype(int)

        refined_predictions = initial_predictions.copy()

        # Get unique classes from initial predictions and labeled points
        all_classes = np.unique(np.concatenate([initial_predictions, labeled_classes]))

        # For each unlabeled point, check if nearby labeled points strongly suggest a different class
        for i, (point, pred) in enumerate(zip(unlabeled_points, initial_predictions)):
            # Calculate distances to all labeled points
            distances = np.linalg.norm(labeled_coords - point, axis=1)

            # Use inverse distance weighting (avoid division by zero)
            weights = 1.0 / (distances + 1e-6)

            # Weight by class
            class_weights = {int(c): 0.0 for c in all_classes}
            for j, label in enumerate(labeled_classes):
                class_weights[int(label)] += weights[j]

            # If a different class has significantly more weight, switch to it
            max_weight_class = max(class_weights, key=class_weights.get)
            if max_weight_class != pred and class_weights[max_weight_class] > 1.5 * class_weights.get(int(pred), 0):
                refined_predictions[i] = max_weight_class

        return refined_predictions

    def predict_entire_grid(self, all_points):
        """
        Predict labels for all points in a grid (convenience method).

        Args:
            all_points: List or array of (x, y) coordinates
                       Example: [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]

        Returns:
            Array of predicted labels
        """
        return self.predict_grid(all_points, labeled_points=None, refine_with_labeled=False)

    def get_model_info(self):
        """Get information about the trained model."""
        info = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'n_training_grids': len(self.training_grids),
            'training_grids': []
        }

        if self.training_grids:
            total_points = 0
            for grid in self.training_grids:
                labels = grid['labels']
                total_points += len(labels)
                info['training_grids'].append({
                    'name': grid['name'],
                    'n_points': len(labels),
                    'class_distribution': {int(i): int(np.sum(labels == i)) for i in np.unique(labels)}
                })
            info['total_training_points'] = total_points

        return info
