import csv
import ast  # to safely convert strings like "[[33.285, 35.548], ...]" back to Python lists
import matplotlib.pyplot as plt

def plot_cluster_csv(csv_file, title="Cluster Grid"):
    plt.figure(figsize=(8, 8))

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert strings to lists
            friend_list = ast.literal_eval(row["friend_pos_list"])
            foe_list = ast.literal_eval(row["foe_pos_list"])
            target = ast.literal_eval(row["target_coor"])
            label = row["label"]

            # Plot friend cluster
            if friend_list:
                lats, lons = zip(*friend_list)
                plt.scatter(lons, lats, c="blue", marker="o", s=50, edgecolors="black", alpha=0.7)

            # Plot foe cluster
            if foe_list:
                lats, lons = zip(*foe_list)
                plt.scatter(lons, lats, c="red", marker="s", s=50, edgecolors="black", alpha=0.7)

            # Plot target coordinate
            plt.scatter(target[1], target[0], c="green", marker="*", s=100, edgecolors="black")

            # Optional: annotate with label
            plt.text(target[1], target[0]+0.0003, label, fontsize=9, ha="center")

    # Labels, grid, aspect ratio
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


if __name__ == "__main__":
    plot_cluster_csv("cluster_samples_random_clusters.csv", title="Friend vs Foe Clusters")
