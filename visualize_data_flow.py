"""
Visualize the data transformation from CSV to model input
"""

import numpy as np
import pandas as pd
from utils import load_csv_grids
from model import Grid2DClassifier


def demonstrate_data_flow():
    """Show exactly what happens to the data at each step."""

    print("="*70)
    print("DATA TRANSFORMATION: CSV → MODEL")
    print("="*70)

    # STEP 1: Load CSV
    print("\n📄 STEP 1: Load CSV File")
    print("-"*70)
    df = pd.read_csv('coordinates/field1.csv')
    print(f"Pandas DataFrame loaded:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")

    # STEP 2: load_csv_grids
    print("\n\n🔄 STEP 2: load_csv_grids()")
    print("-"*70)
    grids_list = load_csv_grids(['coordinates/field1.csv'])
    grid_name, points = grids_list[0]

    print(f"\nOutput: List of tuples")
    print(f"  Type: {type(grids_list)}")
    print(f"  Length: {len(grids_list)}")
    print(f"\nFirst element:")
    print(f"  grid_name: '{grid_name}' (type: {type(grid_name)})")
    print(f"  points: (type: {type(points)})")
    print(f"  points.shape: {points.shape}")
    print(f"  points.dtype: {points.dtype}")
    print(f"\nFirst 3 rows of points array:")
    print(points[:3])
    print(f"\n  Column 0 (longitude): {points[:3, 0]}")
    print(f"  Column 1 (latitude):  {points[:3, 1]}")
    print(f"  Column 2 (class):     {points[:3, 2]}")

    # STEP 3: add_training_grids
    print("\n\n🏗️  STEP 3: add_training_grids()")
    print("-"*70)
    classifier = Grid2DClassifier(model_type='random_forest')
    classifier.add_training_grids(grids_list)

    training_grid = classifier.training_grids[0]
    print(f"\nStored in self.training_grids[0]:")
    print(f"  Type: {type(training_grid)}")
    print(f"  Keys: {list(training_grid.keys())}")
    print(f"\n  'name': '{training_grid['name']}'")
    print(f"  'coordinates': shape={training_grid['coordinates'].shape}, dtype={training_grid['coordinates'].dtype}")
    print(f"  'labels': shape={training_grid['labels'].shape}, dtype={training_grid['labels'].dtype}")
    print(f"  'label_map': {training_grid['label_map']}")

    print(f"\nFirst 3 coordinate pairs:")
    print(training_grid['coordinates'][:3])
    print(f"\nFirst 10 labels:")
    print(training_grid['labels'][:10])

    # STEP 4: train() - combine grids
    print("\n\n🔗 STEP 4: train() - Combine Grids")
    print("-"*70)

    # Simulate what train() does
    all_coordinates = []
    all_labels = []
    for grid in classifier.training_grids:
        all_coordinates.append(grid['coordinates'])
        all_labels.append(grid['labels'])

    X = np.vstack(all_coordinates)
    y = np.concatenate(all_labels)

    print(f"X (feature matrix):")
    print(f"  Type: {type(X)}")
    print(f"  Shape: {X.shape}")
    print(f"  Dtype: {X.dtype}")
    print(f"\nFirst 3 rows:")
    print(X[:3])
    print(f"\ny (label vector):")
    print(f"  Type: {type(y)}")
    print(f"  Shape: {y.shape}")
    print(f"  Dtype: {y.dtype}")
    print(f"\nFirst 10 labels:")
    print(y[:10])

    # STEP 5: Scale features
    print("\n\n📏 STEP 5: Scale Features (Standardization)")
    print("-"*70)

    classifier.train(n_estimators=50, random_state=42)

    # Access the scaler and show transformation
    X_scaled = classifier.scaler.transform(X)

    print(f"StandardScaler parameters:")
    print(f"  Mean: {classifier.scaler.mean_}")
    print(f"  Std:  {classifier.scaler.scale_}")

    print(f"\nX_scaled (normalized features):")
    print(f"  Type: {type(X_scaled)}")
    print(f"  Shape: {X_scaled.shape}")
    print(f"  Dtype: {X_scaled.dtype}")
    print(f"\nFirst 3 rows (normalized):")
    print(X_scaled[:3])

    print(f"\nComparison for first point:")
    print(f"  Original:   longitude={X[0, 0]:.5f}, latitude={X[0, 1]:.5f}")
    print(f"  Normalized: longitude={X_scaled[0, 0]:.5f}, latitude={X_scaled[0, 1]:.5f}")

    # STEP 6: Model input
    print("\n\n🤖 STEP 6: Model Input (sklearn RandomForest)")
    print("-"*70)

    print(f"model.fit() receives:")
    print(f"  X: np.ndarray of shape {X_scaled.shape}")
    print(f"  y: np.ndarray of shape {y.shape}")
    print(f"\nModel type: {type(classifier.model)}")
    print(f"Model parameters:")
    print(f"  n_estimators: {classifier.model.n_estimators}")
    print(f"  n_features_in_: {classifier.model.n_features_in_}")
    print(f"  n_classes_: {classifier.model.n_classes_}")
    print(f"  classes_: {classifier.model.classes_}")

    # STEP 7: Prediction
    print("\n\n🔮 STEP 7: Prediction on New Points")
    print("-"*70)

    new_points = [[35.5500, 33.2700], [35.5485, 33.2865]]
    print(f"New points to predict:")
    print(f"  Type: {type(new_points)}")
    print(f"  Points: {new_points}")

    predictions = classifier.predict_grid(new_points)

    new_points_scaled = classifier.scaler.transform(new_points)

    print(f"\nAfter scaling:")
    print(new_points_scaled)

    print(f"\nPredictions:")
    print(f"  Type: {type(predictions)}")
    print(f"  Shape: {predictions.shape}")
    print(f"  Values: {predictions}")

    print(f"\nInterpretation:")
    for i, (point, pred) in enumerate(zip(new_points, predictions)):
        print(f"  Point {i+1}: ({point[0]:.4f}, {point[1]:.4f}) → Class {pred}")

    # Summary diagram
    print("\n\n📊 COMPLETE DATA FLOW SUMMARY")
    print("="*70)
    print("""
CSV File (field1.csv)
    ↓ pd.read_csv()
DataFrame (15 rows × 3 cols: latitude, longitude, class)
    ↓ load_csv_grids() → extract columns, combine
List[Tuple[str, ndarray]]  →  [('field1', array(15×3))]
    ↓ add_training_grids() → split into coords & labels
Dict {'coordinates': array(15×2), 'labels': array(15)}
    ↓ train() → combine all grids
X: ndarray(15×2), y: ndarray(15)
    ↓ StandardScaler.fit_transform()
X_scaled: ndarray(15×2) [normalized], y: ndarray(15)
    ↓ model.fit(X_scaled, y)
Trained RandomForestClassifier
    ↓ model.predict(new_points_scaled)
predictions: ndarray [class labels]
    """)

    print("="*70)
    print("✅ DATA FLOW COMPLETE")
    print("="*70)

    # Print key object fields
    print("\n\n📋 KEY OBJECT FIELDS REFERENCE")
    print("="*70)

    print("\n1. grids_list (from load_csv_grids):")
    print("   Type: List[Tuple[str, np.ndarray]]")
    print("   Structure: [('grid_name', points_array), ...]")
    print("   points_array shape: (n_points, 3)")
    print("   points_array columns: [longitude, latitude, class]")

    print("\n2. self.training_grids (in Grid2DClassifier):")
    print("   Type: List[Dict[str, Any]]")
    print("   Structure: [{")
    print("       'name': str,")
    print("       'coordinates': np.ndarray(shape=(n, 2)),  # [lon, lat]")
    print("       'labels': np.ndarray(shape=(n,)),         # [class values]")
    print("       'label_map': dict")
    print("   }]")

    print("\n3. X, y (training data):")
    print("   X: np.ndarray(shape=(n_samples, 2))  # [longitude, latitude]")
    print("   y: np.ndarray(shape=(n_samples,))    # class labels")

    print("\n4. X_scaled (final model input):")
    print("   Type: np.ndarray(shape=(n_samples, 2))")
    print("   Values: Z-score normalized features")
    print("   This + y goes to model.fit()")

    print("="*70)


if __name__ == "__main__":
    demonstrate_data_flow()
