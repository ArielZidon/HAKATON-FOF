import csv
import ast
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_cluster_csv_shade(csv_file, title="Cluster Grid with Shade"):
    # First, count total rows for shading
    with open(csv_file, "r") as f:
        total_rows = sum(1 for _ in f) - 1  # minus header

    plt.figure(figsize=(8, 8))

    # Use a colormap (Blues for friend, Reds for foe)
    friend_cmap = cm.Blues
    foe_cmap = cm.Reds
    # Normalize so 0 = first row, 1 = last row
    norm = mcolors.Normalize(vmin=0, vmax=total_rows-1)

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Convert strings to lists
            friend_list = ast.literal_eval(row["friend_pos_list"])
            foe_list = ast.literal_eval(row["foe_pos_list"])
            target = ast.literal_eval(row["target_coor"])
            label = row["label"]

            shade = norm(i)  # value between 0 and 1 for color gradient

            # Plot friend cluster
            if friend_list:
                lats, lons = zip(*friend_list)
                plt.scatter(lons, lats, c=[friend_cmap(0.3 + 0.7*shade)]*len(lats),
                            marker="o", s=50, edgecolors="black", alpha=0.8)

            # Plot foe cluster
            if foe_list:
                lats, lons = zip(*foe_list)
                plt.scatter(lons, lats, c=[foe_cmap(0.3 + 0.7*shade)]*len(lats),
                            marker="s", s=50, edgecolors="black", alpha=0.8)

            # Plot target coordinate
            plt.scatter(target[1], target[0], c='green', marker="*", s=100, edgecolors="black")
            plt.text(target[1], target[0]+0.0002, label, fontsize=8, ha="center")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

if __name__ == "__main__":
    plot_cluster_csv_shade("cluster_samples_random_clusters.csv", title="Clusters with Shading by Order")
