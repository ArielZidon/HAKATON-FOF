# Data Structures and Input Objects Documentation

## Overview: Data Flow from CSV to Model

```
CSV File → load_csv_grids() → grids_list → add_training_grids() →
training_grids → train() → X, y → model.fit(X_scaled, y)
```

---

## 1. CSV File Format

**Location**: `coordinates/field1.csv`

**Structure**:
```csv
latitude,longitude,class
33.28661,35.54842,1
33.26877,35.55219,0
...
```

**Type**: Plain text CSV file
**Columns**: 3 (latitude, longitude, class)

---

## 2. Output of `load_csv_grids()`

**Location**: [utils.py:13-51](utils.py#L13-L51)

**Function signature**:
```python
def load_csv_grids(csv_files):
    # Returns: List of tuples
```

### Return Type: `List[Tuple[str, np.ndarray]]`

**Structure**:
```python
grids = [
    (
        'field1',  # str: grid name (from filename)
        np.array([  # np.ndarray with shape (n_points, 3)
            [35.54842, 33.28661, 1.0],  # [longitude, latitude, class]
            [35.55219, 33.26877, 0.0],
            [35.54854, 33.28619, 1.0],
            # ... more points
        ])
    )
]
```

**Key Details**:
- **Type**: `list` of `tuple`
- **Tuple structure**: `(grid_name: str, points: np.ndarray)`
- **Points array**:
  - **Type**: `numpy.ndarray`
  - **Shape**: `(n_points, 3)` where n_points = 15 for field1.csv
  - **dtype**: `float64`
  - **Columns**: `[longitude, latitude, class]`

**Code Location**:
```python
# utils.py:38-43
x = df[x_col].values        # Extract longitude
y = df[y_col].values        # Extract latitude
labels = df[label_col].values  # Extract class

# Combine into (x, y, label) format
points = np.column_stack([x, y, labels])  # ← THIS IS THE OUTPUT
```

**Example for field1.csv**:
```python
grids_list = [
    ('field1', array([[35.54842, 33.28661, 1.],
                      [35.54871, 33.28673, 1.],
                      [35.55219, 33.26877, 0.],
                      # ... 12 more rows
                     ]))
]
```

---

## 3. Input to `add_training_grids()`

**Location**: [model.py:56-92](model.py#L56-L92)

**Function signature**:
```python
def add_training_grids(self, grids_list):
    # Parameter: grids_list
```

### Parameter Type: `List[Tuple[str, np.ndarray]]`

**Expected format** (same as output from `load_csv_grids()`):
```python
grids_list = [
    ('grid_name', np.array([[x1, y1, label1],
                            [x2, y2, label2],
                            ...])),
    # ... more grids
]
```

**Processing** (model.py:65-88):
```python
for grid_name, points in grids_list:
    points = np.array(points)  # Ensure numpy array

    # SPLIT INTO COORDINATES AND LABELS
    coordinates = points[:, :2]   # Extract first 2 columns ← KEY STEP
    labels = points[:, 2].astype(int)  # Extract 3rd column ← KEY STEP

    # Store in internal structure
    self.training_grids.append({
        'name': grid_name,
        'coordinates': coordinates,  # ← THIS IS THE MODEL INPUT
        'labels': labels,
        'label_map': label_map
    })
```

---

## 4. Internal Storage: `self.training_grids`

**Location**: [model.py:83-88](model.py#L83-L88)

### Type: `List[Dict[str, Any]]`

**Structure**:
```python
self.training_grids = [
    {
        'name': 'field1',  # str
        'coordinates': np.ndarray(shape=(15, 2), dtype=float64),  # ← MODEL INPUT
        'labels': np.ndarray(shape=(15,), dtype=int64),
        'label_map': {0: 0, 1: 1}  # dict mapping original to remapped labels
    }
]
```

**Field Details**:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `name` | `str` | - | Grid identifier (e.g., 'field1') |
| `coordinates` | `np.ndarray` | `(n, 2)` | **[longitude, latitude]** pairs |
| `labels` | `np.ndarray` | `(n,)` | Integer class labels (0, 1, 2, ...) |
| `label_map` | `dict` | - | Maps original labels to 0-indexed |

**Example for field1.csv**:
```python
{
    'name': 'field1',
    'coordinates': array([[35.54842, 33.28661],  # Point 1: [lon, lat]
                          [35.54871, 33.28673],  # Point 2: [lon, lat]
                          [35.55219, 33.26877],  # Point 3: [lon, lat]
                          # ... 12 more rows
                         ]),  # shape: (15, 2)
    'labels': array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]),  # shape: (15,)
    'label_map': {0: 0, 1: 1}  # No remapping needed
}
```

---

## 5. Model Training Input: `X` and `y`

**Location**: [model.py:132-149](model.py#L132-L149)

**Function**: `train()`

### Creating Training Data:

```python
# STEP 1: Combine all grids
all_coordinates = []
all_labels = []

for grid in self.training_grids:
    all_coordinates.append(grid['coordinates'])  # Each is (n, 2)
    all_labels.append(grid['labels'])            # Each is (n,)

# STEP 2: Stack into single arrays
X = np.vstack(all_coordinates)  # ← FEATURE MATRIX
y = np.concatenate(all_labels)  # ← LABEL VECTOR
```

### Type: Feature Matrix `X` and Labels `y`

**X (Features)**:
- **Type**: `numpy.ndarray`
- **Shape**: `(n_total_points, 2)`
- **dtype**: `float64`
- **Columns**: `[longitude, latitude]`

**y (Labels)**:
- **Type**: `numpy.ndarray`
- **Shape**: `(n_total_points,)`
- **dtype**: `int64`
- **Values**: Integer class labels (0, 1, 2, ...)

**For field1.csv** (15 points):
```python
X = array([[35.54842, 33.28661],  # 15 rows
           [35.54871, 33.28673],  # 2 columns (longitude, latitude)
           [35.55219, 33.26877],
           ...
          ])  # shape: (15, 2)

y = array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0])  # shape: (15,)
```

---

## 6. Scaled Features: `X_scaled`

**Location**: [model.py:143-145](model.py#L143-L145)

```python
# STEP 3: Feature scaling (standardization)
self.scaler = StandardScaler()
X_scaled = self.scaler.fit_transform(X)  # ← FINAL INPUT TO MODEL
```

### Type: Scaled Feature Matrix

**X_scaled**:
- **Type**: `numpy.ndarray`
- **Shape**: `(n_total_points, 2)`
- **dtype**: `float64`
- **Values**: Z-score normalized (mean=0, std=1)
- **Columns**: `[normalized_longitude, normalized_latitude]`

**Transformation**:
```
X_scaled = (X - mean) / std

For each feature:
  mean = average of all values in that column
  std = standard deviation of that column
```

**Example**:
```python
# Original X
X = array([[35.54842, 33.28661],
           [35.55219, 33.26877],
           ...])

# After scaling
X_scaled = array([[ 0.523, -1.234],  # Normalized values
                  [ 1.234,  0.891],
                  ...])  # shape: (15, 2)
```

---

## 7. Final Model Input

**Location**: [model.py:148-149](model.py#L148-L149)

```python
# STEP 4: Train the model
self.model = self._create_model(**kwargs)  # Creates RandomForest, etc.
self.model.fit(X_scaled, y)  # ← SKLEARN MODEL RECEIVES THESE
```

### Sklearn Model Input

**`.fit(X, y)` receives**:
- **X**: `X_scaled` - numpy array of shape `(n_samples, n_features)`
  - n_samples = 15 (for field1.csv)
  - n_features = 2 (longitude, latitude)
- **y**: Label vector - numpy array of shape `(n_samples,)`
  - Integer class labels

**This is the standard sklearn format!**

---

## 8. Prediction Input

**Location**: [model.py:158-217](model.py#L158-L217)

**Function**: `predict_grid()`

### Input Type for Prediction:

```python
def predict_grid(self, unlabeled_points, labeled_points=None, ...):
    # unlabeled_points: array-like of shape (n, 2)
```

**unlabeled_points**:
- **Type**: `list` or `numpy.ndarray`
- **Shape**: `(n_predict, 2)`
- **Format**: `[[longitude1, latitude1], [longitude2, latitude2], ...]`

**Example**:
```python
# Predict for new points
new_points = [
    [35.5500, 33.2700],  # [longitude, latitude]
    [35.5485, 33.2865]
]

predictions = classifier.predict_grid(new_points)
# predictions = array([0, 1])  # Predicted classes
```

**Processing**:
```python
# model.py:199-208
unlabeled_points = np.array(unlabeled_points)  # Convert to numpy

# Scale using the same scaler from training
unlabeled_scaled = self.scaler.transform(unlabeled_points)

# Predict using trained model
predictions = self.model.predict(unlabeled_scaled)
```

---

## Summary Table: Object Types at Each Stage

| Stage | Variable | Type | Shape | Description |
|-------|----------|------|-------|-------------|
| **1. Load CSV** | `points` | `np.ndarray` | `(15, 3)` | [lon, lat, class] |
| **2. Grid List** | `grids_list` | `List[Tuple]` | - | [(name, points)] |
| **3. Coordinates** | `coordinates` | `np.ndarray` | `(15, 2)` | [lon, lat] |
| **3. Labels** | `labels` | `np.ndarray` | `(15,)` | [0, 1, 1, ...] |
| **4. Training X** | `X` | `np.ndarray` | `(15, 2)` | Combined coords |
| **4. Training y** | `y` | `np.ndarray` | `(15,)` | Combined labels |
| **5. Scaled X** | `X_scaled` | `np.ndarray` | `(15, 2)` | Normalized X |
| **6. Model Input** | `X_scaled, y` | `np.ndarray` | `(15, 2), (15,)` | Final sklearn input |

---

## Code Locations Quick Reference

| Component | File | Lines | Function |
|-----------|------|-------|----------|
| Load CSV | utils.py | 13-51 | `load_csv_grids()` |
| Add grids | model.py | 56-92 | `add_training_grids()` |
| Training | model.py | 119-156 | `train()` |
| Combine grids | model.py | 132-141 | (in `train()`) |
| Scale features | model.py | 143-145 | (in `train()`) |
| Fit model | model.py | 148-149 | (in `train()`) |
| Prediction | model.py | 158-217 | `predict_grid()` |

---

## Key Insight: The Core Data Structure

**The main input object that goes to the sklearn model is**:

```python
X_scaled: np.ndarray of shape (n_samples, 2)
y: np.ndarray of shape (n_samples,)

Where:
  X_scaled[:, 0] = normalized longitude values
  X_scaled[:, 1] = normalized latitude values
  y = class labels (0, 1, 2, ...)
```

**This is created from**:
- CSV file → pandas DataFrame → numpy array → split into coordinates/labels →
  combine all grids → scale features → **final input to model.fit()**

The entire pipeline transforms:
```
CSV rows → (longitude, latitude, class) → X, y → X_scaled, y → sklearn model
```
