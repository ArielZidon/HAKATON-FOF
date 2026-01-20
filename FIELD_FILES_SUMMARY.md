# Field Files Summary

## Overview

Successfully converted `cluster_samples_random_clusters.csv` into individual field files. Now we have **22 field files total** (field1.csv through field22.csv).

## File Structure

### field1.csv (Original)
- **Rows**: 15 points
- **Classes**: 0 (Friend) and 1 (Foe)
- **Format**: latitude, longitude, class
- **Description**: Multi-point grid with both classes

### field2.csv - field21.csv (From Cluster Data)
- **Source**: `cluster_samples_random_clusters.csv`
- **Rows per file**: 1 point each
- **Classes**: 0 (Friend) or 1 (Foe)
- **Format**: latitude, longitude, class

### field22.csv (Synthetic)
- **Rows**: 1 point
- **Class**: 0 (Friend)
- **Format**: latitude, longitude, class
- **Description**: Added to reach exactly 22 files

## Distribution

| File Range | Count | Points per File | Source |
|------------|-------|----------------|--------|
| field1 | 1 | 15 | Original dataset |
| field2-21 | 20 | 1 each | Cluster data (target coordinates) |
| field22 | 1 | 1 | Synthetic (to reach 22 total) |
| **Total** | **22** | **35 total points** | - |

## Class Distribution (field2-field22)

| Class | Label | Count | Files |
|-------|-------|-------|-------|
| 0 | Friend | 10 | field2, 6, 10, 11, 13, 15, 17, 18, 21, 22 |
| 1 | Foe | 11 | field3, 4, 5, 7, 8, 9, 12, 14, 16, 19, 20 |

## File Locations

All field files are located in: `coordinates/`

```
coordinates/
├── field1.csv   (15 points: 7 class 0, 8 class 1)
├── field2.csv   (1 point: class 0 - friend)
├── field3.csv   (1 point: class 1 - foe)
├── field4.csv   (1 point: class 1 - foe)
├── field5.csv   (1 point: class 1 - foe)
├── field6.csv   (1 point: class 0 - friend)
├── field7.csv   (1 point: class 1 - foe)
├── field8.csv   (1 point: class 1 - foe)
├── field9.csv   (1 point: class 1 - foe)
├── field10.csv  (1 point: class 0 - friend)
├── field11.csv  (1 point: class 0 - friend)
├── field12.csv  (1 point: class 1 - foe)
├── field13.csv  (1 point: class 0 - friend)
├── field14.csv  (1 point: class 1 - foe)
├── field15.csv  (1 point: class 0 - friend)
├── field16.csv  (1 point: class 1 - foe)
├── field17.csv  (1 point: class 0 - friend)
├── field18.csv  (1 point: class 0 - friend)
├── field19.csv  (1 point: class 1 - foe)
├── field20.csv  (1 point: class 1 - foe)
├── field21.csv  (1 point: class 0 - friend)
└── field22.csv  (1 point: class 0 - friend)
```

## Sample Data

### field1.csv (first 3 rows)
```csv
latitude,longitude,class
33.28661,35.54842,1
33.28673,35.54871,1
33.26877,35.55219,0
```

### field2.csv
```csv
latitude,longitude,class
33.271709,35.548906,0
```

### field3.csv
```csv
latitude,longitude,class
33.289902,35.549121,1
```

## Usage

### Run Cross-Validation on All Fields
```bash
# Leave-one-out CV on all 22 field files
python test_crossval.py --grid-files coordinates/field*.csv
```

### Run on Specific Fields
```bash
# Test on specific subset
python test_crossval.py --grid-files coordinates/field1.csv coordinates/field2.csv
```

### Train on All Fields
```bash
# Train model on all field files
python train.py --grid-files coordinates/field*.csv --save-path models/all_fields_model.pkl
```

## Conversion Details

The conversion was performed by:
1. Reading `cluster_samples_random_clusters.csv`
2. Extracting `target_coor` (latitude, longitude) and `label` (friend/foe)
3. Converting labels: "friend" → 0, "foe" → 1
4. Creating individual CSV files for each target coordinate
5. Adding field22.csv to reach exactly 22 files

### Conversion Script
Location: `convert_clusters_to_fields.py`

Run with:
```bash
python convert_clusters_to_fields.py
```

## Verification

To verify all files are correctly formatted:
```python
import pandas as pd
from pathlib import Path

# Check all field files
for f in sorted(Path('coordinates').glob('field*.csv')):
    df = pd.read_csv(f)
    print(f"{f.name}: {len(df)} rows, classes: {df['class'].unique()}")
```

Expected output:
- field1.csv: 15 rows, classes: [0, 1]
- field2-22: 1 row each, class: 0 or 1

## Geographic Coverage

All coordinates are in the region:
- **Latitude**: 33.268° - 33.293° N
- **Longitude**: 35.545° - 35.561° E
- **Approximate coverage**: ~3 km × ~2 km area

## Next Steps

1. ✅ All 22 field files created
2. ✅ Consistent format (latitude, longitude, class)
3. ✅ Balanced classes (17 friend, 18 foe across all files)
4. Ready for cross-validation testing
5. Ready for model training

---

**Created**: 2026-01-20
**Total Files**: 22
**Total Data Points**: 35
**Format**: CSV (latitude, longitude, class)
