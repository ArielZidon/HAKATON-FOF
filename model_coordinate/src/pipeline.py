from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from .features.feature_engineering import build_feature_matrix


def build_pipeline(model_type: str, params: Dict[str, Any]) -> Pipeline:
    transformer = FunctionTransformer(build_feature_matrix, validate=False)

    if model_type == "knn":
        model = KNeighborsClassifier(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    elif model_type == "mlp":
        model = MLPClassifier(max_iter=500, **params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline([
        ("features", transformer),
        ("scaler", StandardScaler()),
        ("model", model),
    ])
