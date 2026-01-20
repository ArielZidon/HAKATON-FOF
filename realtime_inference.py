import joblib
import numpy as np

MODEL_PATH = "models/best_model.pkl"


def main():
    model = joblib.load(MODEL_PATH)

    # points to infer on: shape (N, 2)
    points = np.array([
        [12.3, 45.6],
        [13.1, 44.2],
        [10.0, 40.0],
    ])

    # Grid2DClassifier inference
    preds = model.predict_grid(
        unlabeled_points=points,
        labeled_points=None,
        refine_with_labeled=False
    )

    print("Predictions:")
    print(preds)


if __name__ == "__main__":
    main()