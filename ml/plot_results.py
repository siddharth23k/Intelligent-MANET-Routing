import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data_utils import DATA_PATH, FIGURES_DIR, load_raw_dataset


def plot_distance_distribution(df: pd.DataFrame, output_path: str | None = None, show: bool = True) -> str:
    """
    Plot the distribution of link distances and save to disk.

    Returns the path of the saved figure.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "distance_distribution.png")

    plt.figure(figsize=(8, 6))
    sns.histplot(df["distance"], bins=50)

    plt.title("Distribution of Link Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")

    plt.savefig(output_path)
    if show:
        plt.show()
    else:
        plt.close()

    return output_path


def plot_link_status(df: pd.DataFrame, output_path: str | None = None, show: bool = True) -> str:
    """
    Plot the distribution of link_status labels and save to disk.

    Returns the path of the saved figure.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "link_stability.png")

    plt.figure()
    sns.countplot(x="link_status", data=df)
    plt.title("Link Stability Distribution")
    plt.xlabel("Link Status (0 = Broken, 1 = Stable)")
    plt.ylabel("Count")

    plt.savefig(output_path)
    if show:
        plt.show()
    else:
        plt.close()

    return output_path


def main() -> None:
    """Entry point for quick CLI-style usage."""
    df = load_raw_dataset(DATA_PATH)
    print("Dataset shape:", df.shape)

    dist_path = plot_distance_distribution(df)
    status_path = plot_link_status(df)

    print("Graphs saved to:")
    print(" -", dist_path)
    print(" -", status_path)


if __name__ == "__main__":
    main()
