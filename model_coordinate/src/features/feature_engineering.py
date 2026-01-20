from typing import List, Tuple
import math

import numpy as np
import pandas as pd

Coord = Tuple[float, float]


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _centroid(coords: List[Coord]) -> Coord:
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return float(np.mean(lons)), float(np.mean(lats))


def _mean_distance_to_point(coords: List[Coord], point: Coord) -> float:
    return float(np.mean([haversine_km(c[0], c[1], point[0], point[1]) for c in coords]))


def _min_distance_to_point(coords: List[Coord], point: Coord) -> float:
    return float(np.min([haversine_km(c[0], c[1], point[0], point[1]) for c in coords]))


def _mean_pairwise_distance(coords: List[Coord]) -> float:
    if len(coords) < 2:
        return 0.0
    dists = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dists.append(haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1]))
    return float(np.mean(dists)) if dists else 0.0


def feature_names() -> List[str]:
    return [
        "friend_count",
        "foe_count",
        "friend_centroid_lon",
        "friend_centroid_lat",
        "foe_centroid_lon",
        "foe_centroid_lat",
        "target_to_friend_centroid_km",
        "target_to_foe_centroid_km",
        "friend_to_foe_centroid_km",
        "nearest_friend_km",
        "nearest_foe_km",
        "friend_mean_spread_km",
        "foe_mean_spread_km",
        "friend_mean_distance_to_centroid_km",
        "foe_mean_distance_to_centroid_km",
        "target_nearest_delta_km",
    ]


def build_feature_vector(friend_pos_list: List[Coord], foe_pos_list: List[Coord], target_coord: Coord) -> List[float]:
    friend_centroid = _centroid(friend_pos_list)
    foe_centroid = _centroid(foe_pos_list)

    target_to_friend = haversine_km(target_coord[0], target_coord[1], friend_centroid[0], friend_centroid[1])
    target_to_foe = haversine_km(target_coord[0], target_coord[1], foe_centroid[0], foe_centroid[1])
    friend_to_foe = haversine_km(friend_centroid[0], friend_centroid[1], foe_centroid[0], foe_centroid[1])

    nearest_friend = _min_distance_to_point(friend_pos_list, target_coord)
    nearest_foe = _min_distance_to_point(foe_pos_list, target_coord)

    friend_spread = _mean_pairwise_distance(friend_pos_list)
    foe_spread = _mean_pairwise_distance(foe_pos_list)

    friend_mean_to_centroid = _mean_distance_to_point(friend_pos_list, friend_centroid)
    foe_mean_to_centroid = _mean_distance_to_point(foe_pos_list, foe_centroid)

    return [
        float(len(friend_pos_list)),
        float(len(foe_pos_list)),
        friend_centroid[0],
        friend_centroid[1],
        foe_centroid[0],
        foe_centroid[1],
        target_to_friend,
        target_to_foe,
        friend_to_foe,
        nearest_friend,
        nearest_foe,
        friend_spread,
        foe_spread,
        friend_mean_to_centroid,
        foe_mean_to_centroid,
        nearest_friend - nearest_foe,
    ]


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    features = []
    for _, row in df.iterrows():
        features.append(
            build_feature_vector(
                row["friend_pos_list"],
                row["foe_pos_list"],
                row["target_coord"],
            )
        )
    return np.array(features, dtype=float)
