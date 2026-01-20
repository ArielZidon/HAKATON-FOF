import csv
import random
import math

def generate_cluster(center, num_points=5, spread=0.0001):
    """Generate a small cluster of coordinates around a center"""
    cluster = []
    for _ in range(num_points):
        lat = center[0] + random.uniform(-spread, spread)
        lon = center[1] + random.uniform(-spread, spread)
        cluster.append([round(lat, 6), round(lon, 6)])
    return cluster

def generate_random_clusters(num_clusters_range, center_range, min_points=2, max_points=6):
    """Generate a random number of clusters, each with random points"""
    num_clusters = random.randint(*num_clusters_range)
    pos_list = []
    cluster_centers = []
    for _ in range(num_clusters):
        # Random cluster center within the given bounding box
        lat = random.uniform(center_range[0][0], center_range[1][0])
        lon = random.uniform(center_range[0][1], center_range[1][1])
        center = [lat, lon]
        cluster_centers.append(center)
        num_points = random.randint(min_points, max_points)
        pos_list.extend(generate_cluster(center, num_points))
    return pos_list, cluster_centers

def distance(coord1, coord2):
    """Euclidean distance"""
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2)

def closest_cluster(target, friend_list, foe_list):
    """Determine label based on distance to cluster centers"""
    friend_center = [sum(c[0] for c in friend_list)/len(friend_list),
                     sum(c[1] for c in friend_list)/len(friend_list)]
    foe_center = [sum(c[0] for c in foe_list)/len(foe_list),
                  sum(c[1] for c in foe_list)/len(foe_list)]
    return "friend" if distance(target, friend_center) < distance(target, foe_center) else "foe"

def generate_samples(num_samples=50, output_file="cluster_samples_random_clusters.csv"):
    with open(output_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["friend_pos_list","foe_pos_list","target_coor","label"])
        writer.writeheader()
        
        # Bounding boxes for friends and foes
        friend_bounds = [[33.268, 35.550], [33.278, 35.556]]  # min lat/lon, max lat/lon
        foe_bounds = [[33.283, 35.546], [33.290, 35.556]]

        for _ in range(num_samples):
            # Random number of clusters (1-3) for each list
            friend_pos_list, friend_centers = generate_random_clusters((1, 3), friend_bounds)
            foe_pos_list, foe_centers = generate_random_clusters((1, 3), foe_bounds)

            # Place target near a random cluster (friend or foe)
            if random.random() < 0.5:
                # near friend cluster
                target_center = random.choice(friend_centers)
                target_coor = generate_cluster(target_center, num_points=1, spread=0.005)[0]
            else:
                target_center = random.choice(foe_centers)
                target_coor = generate_cluster(target_center, num_points=1, spread=0.005)[0]

            label = closest_cluster(target_coor, friend_pos_list, foe_pos_list)

            writer.writerow({
                "friend_pos_list": str(friend_pos_list),
                "foe_pos_list": str(foe_pos_list),
                "target_coor": str(target_coor),
                "label": label
            })

    print(f"CSV generated: {output_file}")

if __name__ == "__main__":
    generate_samples(num_samples=20)
