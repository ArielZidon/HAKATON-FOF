"""
Test script for inference workflow.

This script:
1) Ensures a trained model exists (trains one if missing).
2) Loads labeled state files.
3) Picks a sample coordinate and runs inference.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from inference import run_inference
from utils import load_csv_grids
from model import Grid2DClassifier


def _pick_sample_coordinate(csv_file):
    df = pd.read_csv(csv_file)
    x_col = "longitude" if "longitude" in df.columns else "x"
    y_col = "latitude" if "latitude" in df.columns else "y"
    label_col = "class" if "class" in df.columns else "label"

    if x_col not in df.columns or y_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV {csv_file} missing coordinate/label columns")

    row = df.iloc[0]
    return float(row[x_col]), float(row[y_col]), int(row[label_col])


def main():
    root = Path(__file__).parent
    model_path = root / "models" / "trained_model.pkl"

    # State files to use for inference
    state_files = [
        str(root / "coordinates" / "field1.csv"),
        str(root / "coordinates" / "field2.csv"),
        str(root / "coordinates" / "field3.csv"),
    ]

    # Ensure model exists
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print("Model not found. Training a small model for testing...")
        training_grids = load_csv_grids(state_files)
        classifier = Grid2DClassifier(model_type="random_forest")
        classifier.add_training_grids(training_grids)
        classifier.train(n_estimators=100, random_state=42)

        with open(model_path, "wb") as f:
            import pickle
            pickle.dump(classifier, f)

    # Pick a sample coordinate from the first state file
    longitude, latitude, true_label = _pick_sample_coordinate(state_files[0])

    # Run inference
    predicted_label = run_inference(
        model_path=str(model_path),
        state_files=state_files,
        longitude=longitude,
        latitude=latitude,
        refine_with_labeled=True,
    )

    print("=" * 60)
    print("TEST INFERENCE RESULT")
    print("=" * 60)
    print(f"Sample coordinate: (longitude={longitude}, latitude={latitude})")
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_label}")

    if predicted_label == true_label:
        print("Result: OK")
    else:
        print("Result: MISMATCH")


if __name__ == "__main__":
    main()
