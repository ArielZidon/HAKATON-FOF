import json
from typing import List, Tuple, Dict, Any

import joblib
import pandas as pd

from .data.io import parse_coord_list, parse_target_coord

Coord = Tuple[float, float]


def classify_coordinate(
    model_path: str,
    friend_pos_list: List[Coord],
    foe_pos_list: List[Coord],
    target_coord: Coord,
) -> Dict[str, Any]:
    pipeline = joblib.load(model_path)
    df = pd.DataFrame([
        {
            "friend_pos_list": friend_pos_list,
            "foe_pos_list": foe_pos_list,
            "target_coord": target_coord,
        }
    ])

    prediction = pipeline.predict(df)[0]
    probabilities = {}
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(df)[0]
        for cls, score in zip(pipeline.named_steps["model"].classes_, proba):
            probabilities[str(cls)] = float(score)

    return {
        "prediction": str(prediction),
        "probabilities": probabilities,
    }


def parse_json_or_file(value: str) -> Any:
    if value.endswith(".json"):
        with open(value, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def parse_inputs(friend_value: str, foe_value: str, target_value: str):
    friend = parse_coord_list(parse_json_or_file(friend_value))
    foe = parse_coord_list(parse_json_or_file(foe_value))
    target = parse_target_coord(parse_json_or_file(target_value))
    return friend, foe, target
