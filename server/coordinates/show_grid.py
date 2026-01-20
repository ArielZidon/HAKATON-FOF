import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os

def plot_multiple_csvs(csv_files, title="Coordinate Grid"):
    # Styles to cycle through
    colors = itertools.cycle(["blue", "red", "green", "purple", "orange", "brown"])
    markers = itertools.cycle(["o", "s", "^", "D", "x", "*"])

    plt.figure(figsize=(8, 8))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if not {"latitude", "longitude"}.issubset(df.columns):
            raise ValueError(f"{csv_file} must contain 'latitude' and 'longitude' columns")

        color = next(colors)
        marker = next(markers)
        label = os.path.splitext(os.path.basename(csv_file))[0]

        plt.scatter(
            df["longitude"],
            df["latitude"],
            c=color,
            marker=marker,
            s=60,
            edgecolors="black",
            label=label
        )

    # Labels and title
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)

    # Grid and aspect ratio
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")

    # Legend
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # Example usage
    plot_multiple_csvs(
        ["friend.csv", "foe.csv"],
        title="Clusters 1 and 3"
    )
