"""
Grid 2D Classifier - Test File

Quick verification test using real field data.

Main testing is done via:
- final_cross_validation.py: Complete CV with model comparison
- test_crossval.py: Leave-one-out CV (legacy)
- test_single_grid.py: K-fold CV for single grid
"""

from model import Grid2DClassifier
from utils import load_csv_grids
from sklearn.metrics import accuracy_score
from pathlib import Path


def quick_test():
    """
    Quick test to verify everything works with real data.
    """
    print("="*60)
    print("QUICK TEST - Grid 2D Classifier")
    print("="*60)

    # Load first 3 field files
    coords_dir = Path('coordinates')
    field_files = sorted(coords_dir.glob('field*.csv'), key=lambda x: int(x.stem[5:]))[:3]
    field_files = [str(f) for f in field_files]

    print(f"\nLoading {len(field_files)} field files for testing...")
    grids = load_csv_grids(field_files)

    # Train on first 2, test on 3rd
    print("\nTraining on field1 and field2, testing on field3...")
    classifier = Grid2DClassifier(model_type='random_forest')
    classifier.add_training_grids(grids[:2])
    classifier.train(n_estimators=100, random_state=42)

    # Test
    test_grid_name, test_grid_points = grids[2]
    test_coords = test_grid_points[:, :2]
    test_labels = test_grid_points[:, 2].astype(int)

    predictions = classifier.predict_entire_grid(test_coords)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"\nTest Grid: {test_grid_name}")
    print(f"Test Points: {len(test_labels)}")
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\n" + "="*60)
    print("QUICK TEST COMPLETE!")
    print("="*60)
    print("\nFor complete testing, run:")
    print("  python final_cross_validation.py")

    return accuracy


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*60)
    print("Running quick verification test...")
    print("="*60 + "\n")

    accuracy = quick_test()

    if accuracy > 0.5:
        print("\n✓ Test passed! Classifier is working correctly.")
    else:
        print("\n⚠ Test completed with low accuracy. Check data or model.")
