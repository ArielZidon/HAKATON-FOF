# Field Files Conversion - Final Summary

## ✅ Completed Successfully

**Total Files Created**: 22 field files (field1.csv through field22.csv)
**Total Data Points**: 341 coordinates across all files

## Conversion Details

### Source Data
- **Input File**: `coordinates/cluster_samples_random_clusters.csv`
- **Columns Used**:
  - `friend_pos_list` → Extracted ALL friend coordinates → **class 0**
  - `foe_pos_list` → Extracted ALL foe coordinates → **class 1**
- **Columns Ignored**: `target_coor`, `label`

### Output Format
Each field file has the format:
```csv
latitude,longitude,class
33.27109,35.55233,0
33.283497,35.549366,1
...
```

## File Statistics

| File | Points | Friends (0) | Foes (1) | Source |
|------|--------|-------------|----------|--------|
| field1.csv | 15 | 7 | 8 | Original dataset |
| field2.csv | 11 | 2 | 9 | Cluster row 2 |
| field3.csv | 18 | 11 | 7 | Cluster row 3 |
| field4.csv | 19 | 16 | 3 | Cluster row 4 |
| field5.csv | 14 | 4 | 10 | Cluster row 5 |
| field6.csv | 19 | 6 | 13 | Cluster row 6 |
| field7.csv | 16 | 4 | 12 | Cluster row 7 |
| field8.csv | 13 | 4 | 9 | Cluster row 8 |
| field9.csv | 18 | 3 | 15 | Cluster row 9 |
| field10.csv | 12 | 6 | 6 | Cluster row 10 |
| field11.csv | 22 | 12 | 10 | Cluster row 11 |
| field12.csv | 16 | 4 | 12 | Cluster row 12 |
| field13.csv | 21 | 13 | 8 | Cluster row 13 |
| field14.csv | 27 | 17 | 10 | Cluster row 14 |
| field15.csv | 16 | 10 | 6 | Cluster row 15 |
| field16.csv | 18 | 14 | 4 | Cluster row 16 |
| field17.csv | 5 | 3 | 2 | Cluster row 17 |
| field18.csv | 16 | 10 | 6 | Cluster row 18 |
| field19.csv | 17 | 5 | 12 | Cluster row 19 |
| field20.csv | 15 | 2 | 13 | Cluster row 20 |
| field21.csv | 9 | 3 | 6 | Cluster row 21 |
| field22.csv | 4 | 2 | 2 | Synthetic |
| **TOTAL** | **341** | **158** | **183** | - |

## Overall Statistics

- **Total Fields**: 22
- **Total Points**: 341
- **Friends (Class 0)**: 158 points (46.3%)
- **Foes (Class 1)**: 183 points (53.7%)
- **Average Points per Field**: 15.5

## File Size Distribution

- **Smallest**: field17.csv (5 points)
- **Largest**: field14.csv (27 points)
- **Median**: 16 points

## Geographic Coverage

All coordinates are in the region:
- **Latitude**: 33.268° - 33.293° N
- **Longitude**: 35.545° - 35.561° E
- **Approximate Area**: ~3 km × ~2 km

## Usage Examples

### 1. Leave-One-Out Cross-Validation
Train on 21 fields, test on 1 field, repeat for all fields:
```bash
python test_crossval.py --grid-files coordinates/field*.csv
```

### 2. Train on All Fields
```bash
python train.py --grid-files coordinates/field*.csv --save-path models/all_fields.pkl
```

### 3. Load and Inspect Data
```python
from utils import load_csv_grids

# Load all field files
grids = load_csv_grids([f'coordinates/field{i}.csv' for i in range(1, 23)])

# Inspect
for name, points in grids:
    print(f"{name}: {len(points)} points")
```

## Conversion Scripts

### Primary Script
**File**: `convert_clusters_to_fields_v2.py`

**What it does**:
1. Reads `cluster_samples_random_clusters.csv`
2. Parses `friend_pos_list` and `foe_pos_list` from each row
3. Creates one field file per row combining all coordinates
4. Assigns class 0 to friends, class 1 to foes

**Run it**:
```bash
python convert_clusters_to_fields_v2.py
```

### Old Script
**File**: `convert_clusters_to_fields.py` (deprecated)
- Only extracted target_coor (single point per row)
- Not recommended for use

## Verification

To verify all files:
```bash
# Count files
ls -1 coordinates/field*.csv | wc -l
# Should output: 22

# Check total points
python -c "
import pandas as pd
from pathlib import Path
total = sum(len(pd.read_csv(f)) for f in Path('coordinates').glob('field*.csv'))
print(f'Total points: {total}')
"
# Should output: Total points: 341
```

## Data Quality

✅ All files validated:
- Correct CSV format
- Three columns: latitude, longitude, class
- Classes are 0 or 1
- No missing values
- Coordinates in valid geographic range

## Next Steps

1. ✅ All 22 field files created and verified
2. ✅ Consistent format across all files
3. ✅ Balanced class distribution (46.3% / 53.7%)
4. Ready for machine learning experiments
5. Ready for cross-validation testing

## Class Mapping

- **Class 0** = Friend (from `friend_pos_list`)
- **Class 1** = Foe (from `foe_pos_list`)

This mapping is consistent with the original field1.csv where:
- Friend coordinates were assigned class 0
- Foe coordinates were assigned class 1

---

**Created**: 2026-01-20
**Conversion Method**: Extract all coordinates from friend_pos_list and foe_pos_list
**Status**: ✅ Complete and Ready for Use
