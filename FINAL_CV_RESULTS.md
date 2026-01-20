# Cross-Validation Results - 22 Field Files

## Executive Summary

**Date**: 2026-01-20 16:11:19
**Total Grids**: 51
**Total Data Points**: 805
**Cross-Validation Method**: Leave-One-Out (LOOCV)
**Best Model**: **GRADIENT_BOOSTING**

## Model Comparison

| Model | Mean Accuracy | Std Dev | Mean F1 | Min Acc | Max Acc | Time (s) |
|-------|---------------|---------|---------|---------|---------|----------|
| RANDOM_FOREST | 0.9517 | 0.1124 | 0.9482 | 0.4615 | 1.0000 | 15.0 |
| GRADIENT_BOOSTING | 0.9540 | 0.1066 | 0.9501 | 0.4615 | 1.0000 | 5.2 |
| NEURAL_NETWORK | 0.9489 | 0.1090 | 0.9454 | 0.5385 | 1.0000 | 35.0 |

## Winner: GRADIENT_BOOSTING

- **Mean Accuracy**: 0.9540 ± 0.1066
- **Mean F1 Score**: 0.9501 ± 0.1192
- **Accuracy Range**: [0.4615, 1.0000]
- **Training Time**: 5.19 seconds

## Per-Fold Results (Best Model)

| Fold | Test Grid | Points | Accuracy | F1 Score |
|------|-----------|--------|----------|----------|
|  1 | field1       |  15 | 1.0000 | 1.0000 |
|  2 | field2       |  20 | 0.8500 | 0.8590 |
|  3 | field3       |  13 | 1.0000 | 1.0000 |
|  4 | field4       |  18 | 1.0000 | 1.0000 |
|  5 | field5       |  15 | 1.0000 | 1.0000 |
|  6 | field6       |  20 | 0.9000 | 0.9000 |
|  7 | field7       |  20 | 1.0000 | 1.0000 |
|  8 | field8       |  28 | 1.0000 | 1.0000 |
|  9 | field9       |  15 | 1.0000 | 1.0000 |
| 10 | field10      |  16 | 1.0000 | 1.0000 |
| 11 | field11      |  14 | 1.0000 | 1.0000 |
| 12 | field12      |  16 | 1.0000 | 1.0000 |
| 13 | field13      |  19 | 0.8421 | 0.8509 |
| 14 | field14      |   6 | 1.0000 | 1.0000 |
| 15 | field15      |  16 | 1.0000 | 1.0000 |
| 16 | field16      |  15 | 1.0000 | 1.0000 |
| 17 | field17      |  18 | 1.0000 | 1.0000 |
| 18 | field18      |  13 | 1.0000 | 1.0000 |
| 19 | field19      |  24 | 1.0000 | 1.0000 |
| 20 | field20      |  10 | 1.0000 | 1.0000 |
| 21 | field21      |  23 | 1.0000 | 1.0000 |
| 22 | field22      |  18 | 1.0000 | 1.0000 |
| 23 | field23      |  15 | 1.0000 | 1.0000 |
| 24 | field24      |  24 | 1.0000 | 1.0000 |
| 25 | field25      |  12 | 1.0000 | 1.0000 |
| 26 | field26      |  24 | 1.0000 | 1.0000 |
| 27 | field27      |  14 | 1.0000 | 1.0000 |
| 28 | field28      |  10 | 0.8000 | 0.7810 |
| 29 | field29      |  21 | 0.8571 | 0.8552 |
| 30 | field30      |  10 | 1.0000 | 1.0000 |
| 31 | field31      |  13 | 1.0000 | 1.0000 |
| 32 | field32      |  10 | 1.0000 | 1.0000 |
| 33 | field33      |  18 | 1.0000 | 1.0000 |
| 34 | field34      |  14 | 1.0000 | 1.0000 |
| 35 | field35      |   5 | 1.0000 | 1.0000 |
| 36 | field36      |   6 | 1.0000 | 1.0000 |
| 37 | field37      |  11 | 0.8182 | 0.8084 |
| 38 | field38      |  27 | 0.9259 | 0.9265 |
| 39 | field39      |  19 | 1.0000 | 1.0000 |
| 40 | field40      |  11 | 1.0000 | 1.0000 |
| 41 | field41      |  15 | 1.0000 | 1.0000 |
| 42 | field42      |  19 | 1.0000 | 1.0000 |
| 43 | field43      |  26 | 0.8077 | 0.7931 |
| 44 | field44      |   8 | 1.0000 | 1.0000 |
| 45 | field45      |  19 | 0.7895 | 0.7706 |
| 46 | field46      |  13 | 1.0000 | 1.0000 |
| 47 | field47      |   7 | 1.0000 | 1.0000 |
| 48 | field48      |  13 | 0.4615 | 0.4615 |
| 49 | field49      |  14 | 1.0000 | 1.0000 |
| 50 | field50      |  15 | 0.6000 | 0.4500 |
| 51 | field51      |  20 | 1.0000 | 1.0000 |

## Saved Model

- **Location**: `models/best_model.pkl`
- **Model Type**: gradient_boosting
- **Trained On**: All 51 grids (805 points)

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
