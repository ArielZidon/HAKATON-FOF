import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from .data.io import load_csv
from .pipeline import build_pipeline
from .features.feature_engineering import feature_names


@dataclass
class TrainOutputs:
    model_path: str
    metrics_path: str
    meta_path: str
    log_path: str


def _load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {
            "split": {"seed": 42, "train_ratio": 0.7, "val_ratio": 0.15},
            "model": {"type": "random_forest", "params": {"n_estimators": 200}},
            "reports": {"output_dir": "reports"},
        }
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _compute_metrics(y_true, y_pred, y_prob, positive_label: str) -> Dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=positive_label
    )
    roc_auc = None
    brier = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score((y_true == positive_label).astype(int), y_prob)
            brier = brier_score_loss((y_true == positive_label).astype(int), y_prob)
        except ValueError:
            pass

    cm = confusion_matrix(y_true, y_pred, labels=["friend", "foe"]).tolist()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "brier": brier,
        "confusion_matrix": cm,
    }


def _init_logger(reports_dir: str, timestamp: str) -> str:
    os.makedirs(reports_dir, exist_ok=True)
    log_path = os.path.join(reports_dir, f"train_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return log_path


def train_model(data_path: str, output_model_path: str, config_path: str | None = None) -> TrainOutputs:
    config = _load_config(config_path)
    reports_dir = config.get("reports", {}).get("output_dir", "reports")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = _init_logger(reports_dir, timestamp)

    logging.info("Starting training run")
    logging.info("Data path: %s", data_path)
    logging.info("Model output: %s", output_model_path)
    logging.info("Config: %s", config)

    df = load_csv(data_path, require_label=True)

    seed = int(config["split"].get("seed", 42))
    train_ratio = float(config["split"].get("train_ratio", 0.7))
    val_ratio = float(config["split"].get("val_ratio", 0.15))

    logging.info("Split ratios: train=%s, val=%s", train_ratio, val_ratio)

    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, random_state=seed, stratify=df["label"]
    )
    val_size = val_ratio / (1.0 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_size, random_state=seed, stratify=temp_df["label"]
    )

    model_type = config["model"]["type"]
    model_params = config["model"].get("params", {})
    logging.info("Model type: %s", model_type)
    logging.info("Model params: %s", model_params)

    pipeline = build_pipeline(model_type, model_params)

    logging.info("Fitting model on training split")
    pipeline.fit(train_df, train_df["label"])

    val_pred = pipeline.predict(val_df)
    val_prob = None
    if hasattr(pipeline, "predict_proba"):
        val_prob = pipeline.predict_proba(val_df)
        class_index = list(pipeline.named_steps["model"].classes_).index("friend")
        val_prob = val_prob[:, class_index]

    val_metrics = _compute_metrics(val_df["label"], val_pred, val_prob, "friend")
    logging.info("Validation metrics: %s", val_metrics)

    test_pred = pipeline.predict(test_df)
    test_prob = None
    if hasattr(pipeline, "predict_proba"):
        test_prob = pipeline.predict_proba(test_df)
        class_index = list(pipeline.named_steps["model"].classes_).index("friend")
        test_prob = test_prob[:, class_index]

    test_metrics = _compute_metrics(test_df["label"], test_pred, test_prob, "friend")
    logging.info("Test metrics: %s", test_metrics)

    train_val_df = pd.concat([train_df, val_df], axis=0)

    logging.info("Refitting model on train+val")
    pipeline.fit(train_val_df, train_val_df["label"])

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(pipeline, output_model_path)
    logging.info("Model artifact written")

    metrics_path = os.path.join(reports_dir, f"metrics_{timestamp}.json")

    metrics_payload = {
        "config": config,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_names": feature_names(),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logging.info("Metrics written: %s", metrics_path)

    meta_path = os.path.join(os.path.dirname(output_model_path), "model_meta.json")
    meta = {
        "model_type": model_type,
        "model_params": model_params,
        "feature_names": feature_names(),
        "metrics_path": metrics_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Metadata written: %s", meta_path)
    logging.info("Training complete")

    return TrainOutputs(
        model_path=output_model_path,
        metrics_path=metrics_path,
        meta_path=meta_path,
        log_path=log_path,
    )
