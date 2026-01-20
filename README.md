# Grid 2D Classifier

A machine learning classifier for 2D coordinate-based classification with 3 classes (0, 1, 2).

## Project Structure

```
.
├── model.py           # Grid2DClassifier model definition
├── utils.py           # Utility functions (data loading, visualization)
├── train.py           # Training script
├── test_crossval.py   # Cross-validation testing script
├── demo.py            # Simple demonstration
├── coordinates/       # Directory containing coordinate CSV files
│   ├── friend.csv
│   ├── foe.csv
│   └── ...
└── README.md
```

## Usage

### 1. Basic Demo

Run a simple demonstration with synthetic data:

```bash
python demo.py
```

### 2. Training

Train a model on your coordinate CSV files:

```bash
# Train on CSV files in coordinates directory
python train.py --grid-files coordinates/friend.csv coordinates/foe.csv

# Or use synthetic data
python train.py --use-synthetic --n-synthetic-grids 5

# Save the trained model
python train.py --grid-files coordinates/*.csv --save-path models/my_model.pkl
```

Training options:
- `--grid-files`: List of CSV files to use as training grids
- `--model-type`: Model type (`random_forest`, `gradient_boosting`, `neural_network`)
- `--n-estimators`: Number of estimators (default: 150)
- `--save-path`: Path to save trained model
- `--use-synthetic`: Generate synthetic training data
- `--no-visualize`: Disable visualization

### 3. Cross-Validation Testing

Run leave-one-out cross-validation on your coordinate files:

```bash
# Test on all CSV files in coordinates directory
python test_crossval.py

# Test on specific files
python test_crossval.py --grid-files coordinates/friend.csv coordinates/foe.csv

# Adjust labeled fraction
python test_crossval.py --labeled-fraction 0.3
```

Testing options:
- `--grid-files`: List of CSV files for cross-validation
- `--coordinates-dir`: Directory containing CSV files (default: `coordinates`)
- `--model-type`: Model type to use
- `--n-estimators`: Number of estimators
- `--labeled-fraction`: Fraction of test points to use as "known" labels (default: 0.2)
- `--no-visualize`: Disable visualizations

## CSV File Format

CSV files should have the following columns:

```csv
latitude,longitude,class
33.28661,35.54842,1
33.28673,35.54871,1
...
```

Or alternatively:

```csv
x,y,label
3.23,7.45,0
5.23,5.00,1
...
```

## Workflow

### Training Workflow

1. Load multiple fully labeled grids (training data)
2. Train a classifier on all grids combined
3. Save the trained model (optional)

### Prediction Workflow

1. Load a trained classifier
2. For a new grid with some labeled and unlabeled points:
   - Provide labeled points: `[(x, y, label), ...]`
   - Provide unlabeled points: `[(x, y), ...]`
3. Predict labels for unlabeled points
4. Optionally refine predictions using labeled points from the same grid

### Cross-Validation Workflow

1. For each grid in the dataset:
   - Train on all other grids
   - Simulate partial labeling on test grid (e.g., 20% labeled)
   - Predict the remaining unlabeled points
   - Evaluate accuracy
2. Report average accuracy across all test grids

## Example Code

```python
from model import Grid2DClassifier
from utils import load_csv_grids, split_labeled_unlabeled

# Load training grids
training_grids = load_csv_grids([
    'coordinates/friend.csv',
    'coordinates/foe.csv'
])

# Create and train classifier
classifier = Grid2DClassifier(model_type='random_forest')
classifier.add_training_grids(training_grids)
classifier.train(n_estimators=150)

# Load a new grid and split into labeled/unlabeled
new_grid = load_csv_grids(['coordinates/new_grid.csv'])[0]
labeled_points, unlabeled_coords, true_labels = split_labeled_unlabeled(
    new_grid[1],
    labeled_fraction=0.2
)

# Predict unlabeled points
predictions = classifier.predict_grid(
    unlabeled_coords,
    labeled_points=labeled_points,
    refine_with_labeled=True
)

# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

## Features

- **Multiple Model Types**: Random Forest, Gradient Boosting, Neural Network
- **Context-Aware Refinement**: Uses labeled points from the same grid to refine predictions
- **Cross-Validation**: Leave-one-out CV to evaluate generalization
- **Visualization**: Built-in plotting for training data and predictions
- **Flexible Input**: Supports various coordinate column names

## Requirements

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
