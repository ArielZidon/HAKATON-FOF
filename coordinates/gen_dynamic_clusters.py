import csv
import random
import math

FRIEND_ADVANTAGE_REGION = {
    "lat_min": 33.283,
    "lat_max": 33.290,
    "lon_min": 35.546,
    "lon_max": 35.552
}



def in_region(center, region):
    lat, lon = center
    return (
        region["lat_min"] <= lat <= region["lat_max"] and
        region["lon_min"] <= lon <= region["lon_max"]
    )

def generate_cluster(center, num_points=5, spread=0.0005):
    """Generate points around a cluster center"""
    cluster = []
    for _ in range(num_points):
        lat = center[0] + random.uniform(-spread, spread)
        lon = center[1] + random.uniform(-spread, spread)
        cluster.append([round(lat,6), round(lon,6)])
    return cluster

def distance(coord1, coord2):
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

def closest_cluster(target, friend_list, foe_list):
    """Label target based on closest cluster centers"""
    ret = "friend"
    if len(foe_list) > 0:
        friend_center = [sum(c[0] for c in friend_list)/len(friend_list),
                        sum(c[1] for c in friend_list)/len(friend_list)]
        foe_center = [sum(c[0] for c in foe_list)/len(foe_list),
                    sum(c[1] for c in foe_list)/len(foe_list)]
        ret = "friend" if distance(target, friend_center) < distance(target, foe_center) else "foe"
    return ret if random.random() > 0.1 else ("foe" if ret == "friend" else "friend")  # 10% noise

def generate_dynamic_clusters(num_samples=20,
                              cluster_prob_change=0.3,
                              num_clusters_range=(3,5),
                              min_points=6,
                              max_points=12,
                              spread=0.0005,
                              grid_bounds=[[33.26074, 35.53925],[33.29121, 35.56258]],
                              friend_bounds=[[33.268, 35.550],[33.274, 35.556]],
                              foe_bounds=[[33.283, 35.546],[33.290, 35.552]],
                              output_file="dynamic_clusters.csv"):

    with open(output_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["friend_pos_list","foe_pos_list","target_coor","label"])
        writer.writeheader()

        # First row: completely random clusters
        friend_clusters = []
        foe_clusters = []

        num_friend_clusters = random.randint(*num_clusters_range) + random.randint(*num_clusters_range)
        num_foe_clusters = random.randint(*num_clusters_range)

        for _ in range(num_friend_clusters):
            center_lat = random.uniform(friend_bounds[0][0], friend_bounds[1][0])
            center_lon = random.uniform(friend_bounds[0][1], friend_bounds[1][1])
            friend_clusters.append([center_lat, center_lon])

        for _ in range(num_foe_clusters):
            center_lat = random.uniform(foe_bounds[0][0], foe_bounds[1][0])
            center_lon = random.uniform(foe_bounds[0][1], foe_bounds[1][1])
            foe_clusters.append([center_lat, center_lon])
        foe_removal_prob = 0
        has_advanced = False
        advanced_id = 0
        fwd_prob = 0.2
        for sample_idx in range(num_samples):
            # Move clusters with some probability
            for i in range(len(friend_clusters)):
                if random.random() < cluster_prob_change:
                    friend_clusters[i][0] += random.uniform(-0.0005,0.0005)
                    friend_clusters[i][1] += random.uniform(-0.001,0.0002)
                if random.random() < fwd_prob and friend_clusters[i][0] < 33.285:
                    friend_clusters[i][0] += random.uniform(-0.0005,0.0005) + 0.001
                if not has_advanced and random.random() < 0.05 and friend_clusters[i][0] < 33.285:
                    has_advanced = True
                    friend_clusters[i][0] = 33.285 + random.uniform(-0.002, 0.0002)
                    foe_removal_prob = 0.02
                    fwd_prob = 0.4
                    advanced_id = i
                if i != advanced_id and in_region(friend_clusters[i], FRIEND_ADVANTAGE_REGION):
                    foe_removal_prob = 0.1
            remaining_foe_clusters = []
            for cluster in foe_clusters:
                if random.random() >= foe_removal_prob:
                    remaining_foe_clusters.append(cluster)

            foe_clusters = remaining_foe_clusters
                
            for i in range(len(foe_clusters)):
                if random.random() < cluster_prob_change:
                    foe_clusters[i][0] += random.uniform(-0.0005,0.0005)
                    foe_clusters[i][1] += random.uniform(-0.0005,0.0005)

            # Generate points for friend and foe clusters
            friend_pos_list = []
            foe_pos_list = []

            for c in friend_clusters:
                num_points = random.randint(min_points, max_points)
                friend_pos_list.extend(generate_cluster(c, num_points, spread))
            for c in foe_clusters:
                num_points = random.randint(min_points, max_points)
                foe_pos_list.extend(generate_cluster(c, num_points, spread))

            # Place target anywhere in the grid
            target_lat = random.uniform(grid_bounds[0][0], grid_bounds[1][0])
            target_lon = random.uniform(grid_bounds[0][1], grid_bounds[1][1])
            target_coor = [round(target_lat,6), round(target_lon,6)]

            # Determine label
            label = closest_cluster(target_coor, friend_pos_list, foe_pos_list)

            # Write CSV row
            writer.writerow({
                "friend_pos_list": str(friend_pos_list),
                "foe_pos_list": str(foe_pos_list),
                "target_coor": str(target_coor),
                "label": label
            })

    print(f"CSV generated: {output_file}")


if __name__ == "__main__":
    generate_dynamic_clusters(num_samples=50)
