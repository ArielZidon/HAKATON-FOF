import csv
import ast
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio
import os

def generate_gif_constant_bounds(csv_file, gif_file="clusters_constant_bounds.gif",
                                 dpi=100, duration=0.5):
    # Read CSV into list
    with open(csv_file, "r") as f:
        reader = list(csv.DictReader(f))
        total_rows = len(reader)

    # Colormaps
    friend_cmap = cm.Blues
    foe_cmap = cm.Reds
    norm = mcolors.Normalize(vmin=0, vmax=total_rows-1)

    # Set fixed bounding box (from your original coordinates)
    lat_min = 33.26074
    lat_max = 33.29121
    lon_min = 35.53925
    lon_max = 35.56258

    temp_frames = []

    for i, row in enumerate(reader):
        fig, ax = plt.subplots(figsize=(8,8))
        shade = norm(i)

        # Extract data
        friend_list = ast.literal_eval(row["friend_pos_list"])
        foe_list = ast.literal_eval(row["foe_pos_list"])
        target = ast.literal_eval(row["target_coor"])
        label = row["label"]

        # Friend cluster
        if friend_list:
            lats, lons = zip(*friend_list)
            ax.scatter(lons, lats, c=[friend_cmap(0.3 + 0.7*shade)]*len(lats),
                       marker="o", s=50, edgecolors="black", alpha=0.8)

        # Foe cluster
        if foe_list:
            lats, lons = zip(*foe_list)
            ax.scatter(lons, lats, c=[foe_cmap(0.3 + 0.7*shade)]*len(lats),
                       marker="s", s=50, edgecolors="black", alpha=0.8)

        # Target coordinate
        ax.scatter(target[1], target[0], c='green', marker="*", s=100, edgecolors="black")
        ax.text(target[1], target[0]+0.0002, label, fontsize=8, ha="center")

        # Formatting
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Row {i+1}/{total_rows}")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

        # Set constant bounds
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # Save temporary frame
        frame_filename = f"_temp_frame_{i}.png"
        plt.savefig(frame_filename, dpi=dpi)
        plt.close(fig)
        temp_frames.append(frame_filename)

    # Create GIF
    images = [imageio.imread(frame) for frame in temp_frames]
    imageio.mimsave(gif_file, images, duration=duration)
    print(f"GIF saved: {gif_file}")

    # Clean up temporary frames
    for frame in temp_frames:
        os.remove(frame)


if __name__ == "__main__":
    generate_gif_constant_bounds("cluster_samples_random_clusters.csv",
                                 gif_file="clusters_constant_bounds.gif",
                                 duration=0.5)


# import csv
# import ast
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# import imageio
# import os

# def generate_gif_one_frame_per_row(csv_file, gif_file="clusters_one_row.gif", dpi=100, duration=0.5):
#     # Read CSV into list
#     with open(csv_file, "r") as f:
#         reader = list(csv.DictReader(f))
#         total_rows = len(reader)

#     # Colormaps
#     friend_cmap = cm.Blues
#     foe_cmap = cm.Reds
#     norm = mcolors.Normalize(vmin=0, vmax=total_rows-1)

#     temp_frames = []

#     for i, row in enumerate(reader):
#         fig, ax = plt.subplots(figsize=(8,8))
#         shade = norm(i)

#         # Extract data
#         friend_list = ast.literal_eval(row["friend_pos_list"])
#         foe_list = ast.literal_eval(row["foe_pos_list"])
#         target = ast.literal_eval(row["target_coor"])
#         label = row["label"]

#         # Friend cluster
#         if friend_list:
#             lats, lons = zip(*friend_list)
#             ax.scatter(lons, lats, c=[friend_cmap(0.3 + 0.7*shade)]*len(lats),
#                        marker="o", s=50, edgecolors="black", alpha=0.8)

#         # Foe cluster
#         if foe_list:
#             lats, lons = zip(*foe_list)
#             ax.scatter(lons, lats, c=[foe_cmap(0.3 + 0.7*shade)]*len(lats),
#                        marker="s", s=50, edgecolors="black", alpha=0.8)

#         # Target coordinate
#         ax.scatter(target[1], target[0], c='green', marker="*", s=100, edgecolors="black")
#         ax.text(target[1], target[0]+0.0002, label, fontsize=8, ha="center")

#         # Formatting
#         ax.set_xlabel("Longitude")
#         ax.set_ylabel("Latitude")
#         ax.set_title(f"Row {i+1}/{total_rows}")
#         ax.grid(True)
#         ax.set_aspect("equal", adjustable="box")

#         # Save temporary frame
#         frame_filename = f"_temp_frame_{i}.png"
#         plt.savefig(frame_filename, dpi=dpi)
#         plt.close(fig)
#         temp_frames.append(frame_filename)

#     # Create GIF
#     images = [imageio.imread(frame) for frame in temp_frames]
#     imageio.mimsave(gif_file, images, duration=duration)
#     print(f"GIF saved: {gif_file}")

#     # Clean up temporary frames
#     for frame in temp_frames:
#         os.remove(frame)


# if __name__ == "__main__":
#     generate_gif_one_frame_per_row("cluster_samples_random_clusters.csv",
#                                    gif_file="clusters_one_row.gif",
#                                    duration=0.5)


# import csv
# import ast
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# import imageio
# import os

# def generate_gif_from_csv(csv_file, gif_file="clusters_animation.gif", dpi=100):
#     # First, count total rows for shading
#     with open(csv_file, "r") as f:
#         total_rows = sum(1 for _ in f) - 1  # minus header

#     # Use colormaps
#     friend_cmap = cm.Blues
#     foe_cmap = cm.Reds
#     norm = mcolors.Normalize(vmin=0, vmax=total_rows-1)

#     # Read CSV into list
#     with open(csv_file, "r") as f:
#         reader = list(csv.DictReader(f))

#     temp_frames = []

#     # Create each frame
#     for i, row in enumerate(reader):
#         fig, ax = plt.subplots(figsize=(8,8))
#         shade = norm(i)

#         # Plot all rows up to current i
#         for j in range(i+1):
#             r = reader[j]
#             friend_list = ast.literal_eval(r["friend_pos_list"])
#             foe_list = ast.literal_eval(r["foe_pos_list"])
#             target = ast.literal_eval(r["target_coor"])
#             label = r["label"]

#             # Friend cluster
#             if friend_list:
#                 lats, lons = zip(*friend_list)
#                 ax.scatter(lons, lats, c=[friend_cmap(0.3 + 0.7*norm(j))]*len(lats),
#                            marker="o", s=50, edgecolors="black", alpha=0.8)

#             # Foe cluster
#             if foe_list:
#                 lats, lons = zip(*foe_list)
#                 ax.scatter(lons, lats, c=[foe_cmap(0.3 + 0.7*norm(j))]*len(lats),
#                            marker="s", s=50, edgecolors="black", alpha=0.8)

#             # Target
#             ax.scatter(target[1], target[0], c='green', marker="*", s=100, edgecolors="black")
#             ax.text(target[1], target[0]+0.0002, label, fontsize=8, ha="center")

#         # Formatting
#         ax.set_xlabel("Longitude")
#         ax.set_ylabel("Latitude")
#         ax.set_title("Cluster Animation")
#         ax.grid(True)
#         ax.set_aspect("equal", adjustable="box")

#         # Save temporary frame
#         frame_filename = f"_temp_frame_{i}.png"
#         plt.savefig(frame_filename, dpi=dpi)
#         plt.close(fig)
#         temp_frames.append(frame_filename)

#     # Create GIF
#     images = [imageio.imread(frame) for frame in temp_frames]
#     imageio.mimsave(gif_file, images, duration=0.5)  # duration in seconds per frame
#     print(f"GIF saved: {gif_file}")

#     # Clean up temporary frames
#     for frame in temp_frames:
#         os.remove(frame)

# if __name__ == "__main__":
#     generate_gif_from_csv("cluster_samples_random_clusters.csv", gif_file="clusters_animation.gif")
