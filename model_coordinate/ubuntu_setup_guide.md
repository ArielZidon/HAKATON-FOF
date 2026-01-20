# Ubuntu Setup Guide (Python Dependencies)

This guide installs everything needed to run training/inference for the coordinate-classification project on Ubuntu Linux.

## 1. System Packages
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential git
```

## 2. Create and Activate a Virtual Environment
From the repository root:
```bash
cd /home/noamhev/git/HAKATON-FOF/model_coordinate
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Upgrade Pip Tools
```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install Core ML Dependencies
```bash
python -m pip install pandas numpy scikit-learn joblib pyyaml
```

## 5. Install Visualization Dependencies
Pick one or both stacks:

**Option A (interactive maps):**
```bash
python -m pip install folium
```

**Option B (static plots):**
```bash
python -m pip install matplotlib
```

**Optional (advanced geospatial plotting):**
```bash
python -m pip install cartopy
```
Note: `cartopy` may require extra system libs on Ubuntu. If needed:
```bash
sudo apt install -y libgeos-dev libproj-dev proj-data proj-bin
```

## 6. Neural Network Dependencies (Optional)
Choose one:

**PyTorch (recommended):**
```bash
python -m pip install torch torchvision torchaudio
```

**TensorFlow/Keras:**
```bash
python -m pip install tensorflow
```

## 7. Developer Tools (Optional)
```bash
python -m pip install pytest black ruff
```

## 8. Verify Installation
```bash
python - <<'PY'
import numpy, pandas, sklearn, joblib
print('OK')
PY
```

If you see `OK`, dependencies are ready.
