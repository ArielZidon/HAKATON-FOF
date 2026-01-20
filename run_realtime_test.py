import os
import re
import time
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

DATA_DIR = "download_senrio/battlefields/out"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
POLL_INTERVAL = 2  # seconds

os.makedirs(MODEL_DIR, exist_ok=True)

ROW_RE = re.compile(r"row_(\d+)\.csv")


def load_or_create_model():
    if os.path.exists(MODEL_PATH):
        print("Loading existing model")
        return joblib.load(MODEL_PATH)

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


def train_on_csv(model, csv_path):
    wait_until_file_stable(csv_path)
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["label"])
    y = df["label"]

    # Grid2DClassifier: no args
    if model.__class__.__name__ == "Grid2DClassifier":
        model.train()

    # sklearn-like models
    elif hasattr(model, "fit"):
        model.fit(X, y)

    else:
        raise RuntimeError(
            f"Unsupported model type: {type(model)}"
        )

    return model


def main():
    model = load_or_create_model()
    processed_idx = -1

    print("Watching directory for new CSV files...")

    while True:
        files = []

        for f in os.listdir(DATA_DIR):
            idx = get_row_index(f)
            if idx is not None and idx > processed_idx:
                files.append((idx, f))

        if not files:
            time.sleep(POLL_INTERVAL)
            continue

        files.sort()

        for idx, fname in files:
            csv_path = os.path.join(DATA_DIR, fname)
            print(f"Training on {fname}")

            try:
                model = train_on_csv(model, csv_path)
            except Exception as e:
                print(f"Failed on {fname}: {e}")
                break

            joblib.dump(model, MODEL_PATH)
            processed_idx = idx

            print(f"Model updated using {fname}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
