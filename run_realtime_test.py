import os
import re
import time
import ast
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# For sklearn>=1.6, cv='prefit' is deprecated; FrozenEstimator is the recommended approach.
try:
    from sklearn.frozen import FrozenEstimator  # type: ignore
except Exception:  # pragma: no cover
    FrozenEstimator = None  # type: ignore

from convert_clusters_to_fields_v2 import convert_clusters_to_fields

DATA_DIR = "download_senrio/battlefields/out"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
POLL_INTERVAL = 2  # seconds

# Where we store converted single-field CSVs
CONVERT_DIR = "convert_csv"

os.makedirs(MODEL_DIR, exist_ok=True)

ROW_RE = re.compile(r"row_(\d+)\.csv")


def load_or_create_model():
    # Only reuse an existing model if it's sklearn-like (has fit).
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            if hasattr(model, "fit"):
                print("Loading existing model")
                return model
            print("Existing model is not sklearn-compatible; creating a new model")
        except Exception as e:
            print(f"Failed to load existing model; creating a new model. Reason: {e}")

    print("Creating new model")
    return GradientBoostingClassifier()


def get_row_index(filename):
    m = ROW_RE.fullmatch(filename)
    return int(m.group(1)) if m else None


def wait_until_file_stable(path, retries=5, delay=0.5):
    last_size = -1
    for _ in range(retries):
        size = os.path.getsize(path)
        if size == last_size:
            return
        last_size = size
        time.sleep(delay)
    raise RuntimeError("File never stabilized")


def _safe_literal_eval(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple, dict)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def featurize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the coordinate/list columns into numeric features."""
    out = pd.DataFrame(index=df.index)

    if "target_coor" in df.columns:
        parsed = df["target_coor"].map(_safe_literal_eval)
        out["target_lat"] = parsed.map(lambda v: float(v[0]) if isinstance(v, (list, tuple)) and len(v) >= 2 else 0.0)
        out["target_lon"] = parsed.map(lambda v: float(v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else 0.0)

    def _list_features(series: pd.Series, prefix: str):
        parsed = series.map(_safe_literal_eval)

        def _coords(v):
            if not isinstance(v, (list, tuple)):
                return []
            coords = []
            for item in v:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    coords.append((float(item[0]), float(item[1])))
            return coords

        coords = parsed.map(_coords)
        out[f"{prefix}_count"] = coords.map(len).astype(float)
        out[f"{prefix}_lat_mean"] = coords.map(lambda xs: sum(x[0] for x in xs) / len(xs) if xs else 0.0)
        out[f"{prefix}_lon_mean"] = coords.map(lambda xs: sum(x[1] for x in xs) / len(xs) if xs else 0.0)
        out[f"{prefix}_lat_min"] = coords.map(lambda xs: min(x[0] for x in xs) if xs else 0.0)
        out[f"{prefix}_lat_max"] = coords.map(lambda xs: max(x[0] for x in xs) if xs else 0.0)
        out[f"{prefix}_lon_min"] = coords.map(lambda xs: min(x[1] for x in xs) if xs else 0.0)
        out[f"{prefix}_lon_max"] = coords.map(lambda xs: max(x[1] for x in xs) if xs else 0.0)

    if "friend_pos_list" in df.columns:
        _list_features(df["friend_pos_list"], "friend")

    if "foe_pos_list" in df.columns:
        _list_features(df["foe_pos_list"], "foe")

    if {"friend_lat_mean", "friend_lon_mean", "target_lat", "target_lon"}.issubset(out.columns):
        out["friend_to_target_dlat"] = out["friend_lat_mean"] - out["target_lat"]
        out["friend_to_target_dlon"] = out["friend_lon_mean"] - out["target_lon"]

    if {"foe_lat_mean", "foe_lon_mean", "target_lat", "target_lon"}.issubset(out.columns):
        out["foe_to_target_dlat"] = out["foe_lat_mean"] - out["target_lat"]
        out["foe_to_target_dlon"] = out["foe_lon_mean"] - out["target_lon"]

    return out


def _load_training_from_csv(csv_path: str):
    """Load a converted field CSV and return numeric X and y.

    Expected converted schema: latitude, longitude, class (0/1).
    """
    wait_until_file_stable(csv_path)
    df = pd.read_csv(csv_path)

    if "class" in df.columns:
        y = df["class"].astype(int)
    elif "label" in df.columns:
        y = df["label"].map(lambda v: 0 if str(v).lower() == "friend" else 1).astype(int)
    else:
        raise RuntimeError("Converted CSV must include a 'class' (0/1) or 'label' column")

    required = {"latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Converted CSV missing required columns: {sorted(missing)}")

    X = df[["latitude", "longitude"]].astype(float)
    return X, y


def _fit_if_possible(model, X_all: pd.DataFrame, y_all: pd.Series):
    classes = sorted(set(y_all.tolist()))
    if len(classes) < 2:
        return None, classes

    # Fit base model
    model.fit(X_all, y_all)

    # Calibrate probabilities (sigmoid) to avoid overly-confident 0/100 outputs.
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
        calibrated.fit(X_all, y_all)
    else:
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        calibrated.fit(X_all, y_all)

    return calibrated, classes


def _atomic_joblib_dump(obj, path: str):
    """Atomic write to avoid corrupted/partial pickle reads."""
    tmp_path = f"{path}.tmp"
    joblib.dump(obj, tmp_path)
    os.replace(tmp_path, path)


def main():
    model = load_or_create_model()
    processed_idx = -1

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, DATA_DIR)

    convert_dir = os.path.join(base_dir, CONVERT_DIR)
    os.makedirs(convert_dir, exist_ok=True)

    print(f"Watching directory for new CSV files: {data_dir}")

    while not os.path.isdir(data_dir):
        time.sleep(POLL_INTERVAL)

    X_parts = []
    y_parts = []

    while True:
        files = []
        for f in os.listdir(data_dir):
            idx = get_row_index(f)
            if idx is not None and idx > processed_idx:
                files.append((idx, f))

        if not files:
            time.sleep(POLL_INTERVAL)
            continue

        files.sort()

        for idx, fname in files:
            csv_path = os.path.join(data_dir, fname)
            print(f"Processing {fname}")

            try:
                convert_clusters_to_fields(csv_path, output_dir=convert_dir, start_field_num=idx)
            except Exception as e:
                print(f"⚠️ convert_clusters_to_fields() failed for {fname}: {e}")
                processed_idx = idx
                continue

            converted_path = os.path.join(convert_dir, f"field{idx}.csv")
            if not os.path.exists(converted_path):
                print(f"⚠️ Expected converted file not found: {converted_path}")
                processed_idx = idx
                continue

            try:
                X_new, y_new = _load_training_from_csv(converted_path)
                X_parts.append(X_new)
                y_parts.append(y_new)

                X_all = pd.concat(X_parts, ignore_index=True)
                y_all = pd.concat(y_parts, ignore_index=True)

                fitted, classes = _fit_if_possible(model, X_all, y_all)
                if fitted is None:
                    print(f"Not enough classes yet ({classes}). Waiting for more data...")
                else:
                    model = fitted
                    _atomic_joblib_dump(model, MODEL_PATH)
                    print(f"Model updated using {fname} (classes={classes})")

            except Exception as e:
                print(f"Failed on {fname}: {e}")

            processed_idx = idx

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
