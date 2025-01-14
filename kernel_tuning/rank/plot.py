import pandas as pd
import os
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import argparse


def plot(df, save_path):
    # add column with product of two other columns
    df["tot_threads"] = df["num_threads"] * df["num_blocks"]

    # group by the product of the two columns
    grouped = df.groupby("tot_threads")

    nrows = round(len(grouped) / 2)
    fig, axs = plt.subplots(nrows, 2, figsize=(20, 10 * nrows))

    # Get one color for each alphabet_size
    num_alphabet_sizes = df["alphabet_size"].nunique()
    colors = plt.cm.rainbow([i / num_alphabet_sizes for i in range(num_alphabet_sizes)])
    for i, (name, group) in enumerate(grouped):
        ax = axs[i // 2, i % 2]
        subgroup = group.groupby("alphabet_size")
        for j, (name2, group2) in enumerate(subgroup):
            ax.plot(
                group2["num_threads"],
                group2["duration"],
                label=f"alphabet_size={name2}",
                color=colors[j],
            )
        ax.set_title(f"tot_threads={name}")
        ax.set_xlabel("num_threads")
        ax.set_ylabel("duration")
        if i == 0:
            ax.legend()

    # Use same y axis for all plots
    for ax in axs.flat:
        ax.label_outer()
        ax.set_ylim(0, df["duration"].max())
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "plot.png"))


if __name__ == "__main__":
    df = pd.read_csv("rank.csv")
    plot(df, ".")
