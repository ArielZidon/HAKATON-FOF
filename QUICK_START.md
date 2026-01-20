# Quick Start Guide

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Available Scripts

### 1. Quick Test (Verify Installation)
```bash
python test.py
```
Runs a quick test to verify all modules are working correctly.

### 2. Basic Demo
```bash
python demo.py
```
Demonstrates basic usage with synthetic data.

### 3. Example Usage
```bash
python example_usage.py
```
Shows how to use the classifier with your coordinate CSV files.

### 4. Cross-Validation Testing
```bash
# Test on all CSV files in coordinates directory
python test_crossval.py

# Test with custom settings
python test_crossval.py --labeled-fraction 0.3 --n-estimators 200

# Test on specific files
python test_crossval.py --grid-files coordinates/friend.csv coordinates/foe.csv

# Disable visualizations (faster)
python test_crossval.py --no-visualize
```

### 5. Training
```bash
# Train on CSV files
python train.py --grid-files coordinates/friend.csv coordinates/foe.csv

# Train on synthetic data
python train.py --use-synthetic --n-synthetic-grids 5

# Save trained model
python train.py --grid-files coordinates/*.csv --save-path models/my_model.pkl

# Choose different model type
python train.py --model-type gradient_boosting --n-estimators 200
```

## Python API Usage

### Basic Example
```python
from model import Grid2DClassifier
from utils import load_csv_grids

# Load training data
grids = load_csv_grids(['coordinates/friend.csv', 'coordinates/foe.csv'])

# Create and train classifier
classifier = Grid2DClassifier(model_type='random_forest')
classifier.add_training_grids(grids)
classifier.train(n_estimators=150)

# Predict new points
unlabeled = [(33.2860, 35.5520), (33.2870, 35.5515)]
predictions = classifier.predict_grid(unlabeled)

print(f"Predictions: {predictions}")
```

### With Context Refinement
```python
# Provide some known labels from the same grid
labeled = [(33.2684, 35.5523, 0), (33.2866, 35.5521, 0)]
unlabeled = [(33.2860, 35.5520), (33.2870, 35.5515)]

predictions = classifier.predict_grid(
    unlabeled,
    labeled_points=labeled,
    refine_with_labeled=True
)
```

### Manual Data Entry
```python
import numpy as np

# Define grids manually
grid1 = [
    (3.23, 7.45, 1),
    (5.23, 5.00, 0),
    (2.11, -0.45, 0),
    (4.23, 0.00, 2)
]

training_grids = [('grid1', np.array(grid1))]

classifier = Grid2DClassifier()
classifier.add_training_grids(training_grids)
classifier.train()

# Predict
predictions = classifier.predict_grid([(3.0, 5.0), (4.0, 2.0)])
```

## CSV File Format

Your CSV files should have these columns:

```csv
latitude,longitude,class
33.28661,35.54842,1
33.28673,35.54871,1
```

Or alternatively:

```csv
x,y,label
3.23,7.45,0
5.23,5.00,1
```

## Model Types

Choose from:
- `random_forest` (default) - Best for most cases
- `gradient_boosting` - Good for complex patterns
- `neural_network` - For non-linear patterns

## Key Parameters

### Training
- `n_estimators`: Number of trees (RF/GB) or epochs (NN). Default: 100
- `random_state`: Random seed for reproducibility. Default: 42

### Prediction
- `refine_with_labeled`: Use known labels to refine predictions. Default: True

### Cross-Validation
- `labeled_fraction`: Fraction of points to use as "known". Default: 0.2 (20%)

## File Structure

```
project/
├── model.py              # Grid2DClassifier class
├── utils.py              # Helper functions
├── train.py              # Training script
├── test_crossval.py      # Cross-validation testing
├── demo.py               # Simple demonstration
├── example_usage.py      # Detailed examples
├── test.py               # Quick test
├── README.md             # Full documentation
├── QUICK_START.md        # This file
└── coordinates/          # Your CSV files
    ├── friend.csv
    └── foe.csv
```

## Common Workflows

### Workflow 1: Train and Predict
1. Prepare CSV files with labeled coordinates
2. Run: `python train.py --grid-files coordinates/*.csv --save-path model.pkl`
3. Load model and predict on new data

### Workflow 2: Cross-Validation
1. Prepare multiple CSV files (one per grid)
2. Run: `python test_crossval.py`
3. Review accuracy results

### Workflow 3: Interactive Development
```python
# In Python/IPython/Jupyter
from model import Grid2DClassifier
from utils import *

# Load your data
grids = load_csv_grids(['coordinates/friend.csv'])

# Train
clf = Grid2DClassifier()
clf.add_training_grids(grids)
clf.train()

# Experiment with predictions
predictions = clf.predict_grid([(33.27, 35.55)])
```

## Troubleshooting

### "No module named 'model'"
Make sure you're in the project directory where model.py is located.

### "No CSV files found"
Check that your CSV files are in the `coordinates/` directory or specify the full path.

### Low accuracy
- Increase `n_estimators` (e.g., 200-300)
- Use more training grids
- Increase `labeled_fraction` in testing
- Try different `model_type`

## Next Steps

- Read [README.md](README.md) for detailed documentation
- See [example_usage.py](example_usage.py) for more examples
- Run [test_crossval.py](test_crossval.py) to evaluate on your data
