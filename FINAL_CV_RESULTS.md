# Cross-Validation Results - 22 Field Files

## Executive Summary

**Date**: 2026-01-20 12:55:31
**Total Grids**: 22
**Total Data Points**: 341
**Cross-Validation Method**: Leave-One-Out (LOOCV)
**Best Model**: **GRADIENT_BOOSTING**

## Model Comparison

| Model | Mean Accuracy | Std Dev | Mean F1 | Min Acc | Max Acc | Time (s) |
|-------|---------------|---------|---------|---------|---------|----------|
| RANDOM_FOREST | 0.9856 | 0.0658 | 0.9876 | 0.6842 | 1.0000 | 5.7 |
| GRADIENT_BOOSTING | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9 |
| NEURAL_NETWORK | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.6 |

## Winner: GRADIENT_BOOSTING

- **Mean Accuracy**: 1.0000 ± 0.0000
- **Mean F1 Score**: 1.0000 ± 0.0000
- **Accuracy Range**: [1.0000, 1.0000]
- **Training Time**: 0.94 seconds

## Per-Fold Results (Best Model)

| Fold | Test Grid | Points | Accuracy | F1 Score |
|------|-----------|--------|----------|----------|
|  1 | field1       |  15 | 1.0000 | 1.0000 |
|  2 | field2       |  11 | 1.0000 | 1.0000 |
|  3 | field3       |  18 | 1.0000 | 1.0000 |
|  4 | field4       |  19 | 1.0000 | 1.0000 |
|  5 | field5       |  14 | 1.0000 | 1.0000 |
|  6 | field6       |  19 | 1.0000 | 1.0000 |
|  7 | field7       |  16 | 1.0000 | 1.0000 |
|  8 | field8       |  13 | 1.0000 | 1.0000 |
|  9 | field9       |  18 | 1.0000 | 1.0000 |
| 10 | field10      |  12 | 1.0000 | 1.0000 |
| 11 | field11      |  22 | 1.0000 | 1.0000 |
| 12 | field12      |  16 | 1.0000 | 1.0000 |
| 13 | field13      |  21 | 1.0000 | 1.0000 |
| 14 | field14      |  27 | 1.0000 | 1.0000 |
| 15 | field15      |  16 | 1.0000 | 1.0000 |
| 16 | field16      |  18 | 1.0000 | 1.0000 |
| 17 | field17      |   5 | 1.0000 | 1.0000 |
| 18 | field18      |  16 | 1.0000 | 1.0000 |
| 19 | field19      |  17 | 1.0000 | 1.0000 |
| 20 | field20      |  15 | 1.0000 | 1.0000 |
| 21 | field21      |   9 | 1.0000 | 1.0000 |
| 22 | field22      |   4 | 1.0000 | 1.0000 |

## Saved Model

- **Location**: `models/best_model.pkl`
- **Model Type**: gradient_boosting
- **Trained On**: All 22 grids (341 points)

## Usage

```python
import pickle

# Load model
with open('models/best_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Predict new points
new_points = [[35.550, 33.270], [35.548, 33.286]]
predictions = classifier.predict_grid(new_points)
print(predictions)  # [0, 1] for example
```
