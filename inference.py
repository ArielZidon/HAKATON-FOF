"""
Inference script for Grid2DClassifier.

Usage:
  python inference.py --model-path models/trained_model.pkl \
    --state-files coordinates/field1.csv coordinates/field2.csv \
    --longitude 35.55 --latitude 33.28

The script loads a trained model, merges labeled points from the provided
state files, and predicts the class for a single coordinate.
"""

import argparse
from pathlib import Path
import pickle
import numpy as np

from utils import load_csv_grids


def _parse_coordinate(args):
    if args.coordinate:
        parts = [p.strip() for p in args.coordinate.split(",")]
        if len(parts) != 2:
            raise ValueError("--coordinate must be in 'longitude,latitude' format")
        longitude = float(parts[0])
        latitude = float(parts[1])
        return longitude, latitude

    if args.longitude is None or args.latitude is None:
        raise ValueError("Provide --longitude and --latitude, or use --coordinate")

    return float(args.longitude), float(args.latitude)


def load_labeled_points(state_files):
    """
    Load labeled points from a list of state CSV files.

    Returns:
        labeled_points: np.ndarray of shape (N, 3) with (x, y, label)
    """
    grids = load_csv_grids(state_files)
    all_points = []

    for grid_name, points in grids:
        if points.shape[1] != 3:
            raise ValueError(f"Grid '{grid_name}' must have 3 columns (x, y, label)")
        all_points.append(points)

    if not all_points:
        raise ValueError("No labeled points loaded from state files")

    return np.vstack(all_points)


def run_inference(model_path, state_files, longitude, latitude, refine_with_labeled=True):
    """
    Run inference for a single coordinate.

    Args:
        model_path: Path to trained model pickle
        state_files: List of CSV files containing labeled points (states)
        longitude: X coordinate
        latitude: Y coordinate
        refine_with_labeled: Whether to use labeled points for refinement

    Returns:
        predicted_label: int
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        classifier = pickle.load(f)

    labeled_points = load_labeled_points(state_files)

    unlabeled_point = np.array([[longitude, latitude]])

    prediction = classifier.predict_grid(
        unlabeled_point,
        labeled_points=labeled_points,
        refine_with_labeled=refine_with_labeled
    )

    return int(prediction[0])


def main():
    parser = argparse.ArgumentParser(description="Run inference for a single coordinate")

    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model pickle (.pkl)")
    parser.add_argument("--state-files", nargs="+", type=str, required=True,
                        help="List of CSV files with labeled coordinates (states)")
    parser.add_argument("--longitude", type=float, help="Longitude (x)")
    parser.add_argument("--latitude", type=float, help="Latitude (y)")
    parser.add_argument("--coordinate", type=str,
                        help="Single coordinate in 'longitude,latitude' format")
    parser.add_argument("--no-refine", action="store_true",
                        help="Disable refinement with labeled points")

    args = parser.parse_args()

    longitude, latitude = _parse_coordinate(args)

    prediction = run_inference(
        model_path=args.model_path,
        state_files=args.state_files,
        longitude=longitude,
        latitude=latitude,
        refine_with_labeled=not args.no_refine
    )

    print("=" * 60)
    print("INFERENCE RESULT")
    print("=" * 60)
    print(f"Input coordinate: (longitude={longitude}, latitude={latitude})")
    print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()
