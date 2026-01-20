# Inference Script User Guide

This guide explains how to set up dependencies, provide inputs to the inference script, and read the output.

## 1. Setup for Dependencies

1. Create and activate a Python environment (recommended):
   - Using venv:
     - `python -m venv .venv`
     - `source .venv/bin/activate`

2. Install required packages:
   - `pip install numpy pandas scikit-learn matplotlib seaborn`

> Note: If your environment is already set up, you can skip these steps.

## 2. How to Give Input to the Script

The inference script is [inference.py](inference.py). It requires:
- A trained model file (`.pkl`) created by the training script.
- N state files (CSV) containing labeled coordinates. Each file must include:
  - `latitude`, `longitude`, and `class` columns (or `x`, `y`, `label`).
- A single coordinate to classify (unknown label).

### Input Options

**Option A — Provide longitude/latitude separately**

- Required arguments:
  - `--model-path` (path to trained model)
  - `--state-files` (one or more CSV files)
  - `--longitude` and `--latitude`

**Option B — Provide coordinate as a single string**

- Use:
  - `--coordinate "<longitude>,<latitude>"`

### Example Command

```
python inference.py \
  --model-path models/trained_model.pkl \
  --state-files coordinates/field1.csv coordinates/field2.csv coordinates/field3.csv \
  --longitude 35.55 --latitude 33.28
```

Or:

```
python inference.py \
  --model-path models/trained_model.pkl \
  --state-files coordinates/field1.csv coordinates/field2.csv \
  --coordinate "35.55,33.28"
```

### Optional Flag

- `--no-refine` disables refinement using labeled points from the same state files.

## 3. Where to See the Output

The script prints the result to standard output (terminal). Example:

```
============================================================
INFERENCE RESULT
============================================================
Input coordinate: (longitude=35.55, latitude=33.28)
Predicted class: 1
```

- **Predicted class** is the classification label for the input coordinate.
- You can redirect output to a file if needed:
  - `python inference.py ... > inference_output.txt`
