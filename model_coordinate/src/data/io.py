import json
from typing import Any, List, Tuple

import pandas as pd


Coord = Tuple[float, float]


def _parse_json(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise ValueError(f"Unsupported value type: {type(value)}")


def _validate_coord(coord: Any) -> Coord:
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        raise ValueError(f"Invalid coordinate: {coord}")
    lon = float(coord[0])
    lat = float(coord[1])
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude out of range: {lon}")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude out of range: {lat}")
    return lon, lat


def parse_coord_list(value: Any) -> List[Coord]:
    coords = _parse_json(value)
    if not isinstance(coords, (list, tuple)):
        raise ValueError("Coordinate list must be a list")
    parsed = [_validate_coord(c) for c in coords]
    if len(parsed) == 0:
        raise ValueError("Coordinate list cannot be empty")
    return parsed


def parse_target_coord(value: Any) -> Coord:
    return _validate_coord(_parse_json(value))


def load_csv(path: str, require_label: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()

    if "target_coord" not in df.columns and "target_coor" in df.columns:
        df = df.rename(columns={"target_coor": "target_coord"})

    required_columns = {"friend_pos_list", "foe_pos_list", "target_coord"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if require_label and "label" not in df.columns:
        raise ValueError("Missing label column for training data")

    df["friend_pos_list"] = df["friend_pos_list"].apply(parse_coord_list)
    df["foe_pos_list"] = df["foe_pos_list"].apply(parse_coord_list)
    df["target_coord"] = df["target_coord"].apply(parse_target_coord)
    return df
