import joblib
import numpy as np
import os
import pandas as pd

MODEL_PATH = "models/best_model.pkl"


def _as_percent(x: float) -> float:
    return float(x) * 100.0


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}. Run training first.")

    model = joblib.load(MODEL_PATH)

    # points to infer on: shape (N, 2) => [latitude, longitude]
    raw_points = np.array(
        [
            [33.27920,35.550131],
        ],
        dtype=float,
    )

    # Use a DataFrame to match training feature names and avoid sklearn warnings.
    points = pd.DataFrame(raw_points, columns=["latitude", "longitude"])

    # Prefer probabilities so we can print percent confidence.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(points)
        classes = getattr(model, "classes_", np.array([0, 1]))

        # Map class -> column index
        idx_by_class = {int(c): i for i, c in enumerate(classes)}
        foe_idx = idx_by_class.get(1)
        friend_idx = idx_by_class.get(0)

        if foe_idx is None or friend_idx is None:
            # Fallback: just print raw probabilities
            print("Model classes:", classes)
            for i, row in enumerate(proba, start=1):
                perc = [f"{_as_percent(p):.2f}%" for p in row]
                print(f"Point {i}: {raw_points[i-1].tolist()} => proba={perc}")
            return

        print("Inference (percent):")
        for i, row in enumerate(proba, start=1):
            friend_p = row[friend_idx]
            foe_p = row[foe_idx]
            pred_class = 1 if foe_p >= friend_p else 0
            pred_label = "foe" if pred_class == 1 else "friend"
            print(
                f"Point {i}: {raw_points[i-1].tolist()} => "
                f"friend={_as_percent(friend_p):.2f}% , "
                f"foe={_as_percent(foe_p):.2f}% => predicted={pred_label}"
            )
        return

    # Fallback for models without predict_proba
    if hasattr(model, "predict"):
        preds = model.predict(points)
        print("Predictions (no probabilities available):")
        for i, pred in enumerate(preds, start=1):
            label = "foe" if int(pred) == 1 else "friend"
            print(f"Point {i}: {raw_points[i-1].tolist()} => predicted={label}")
        return

    raise TypeError(
        f"Loaded model type {type(model)} does not support predict_proba() or predict()."
    )


if __name__ == "__main__":
    main()