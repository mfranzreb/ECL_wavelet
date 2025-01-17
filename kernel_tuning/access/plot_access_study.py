import pandas as pd
import os
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import argparse


def plot(df, save_path):
    # add column with product of two other columns
    df["tot_threads"] = df["num_threads"] * df["num_blocks"]

    nrows = df["group_size"].nunique()
    ncols = df["alphabet_size"].nunique()
    fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows))

    # Get one color for each tot_threads
    num_tot_threads = df["tot_threads"].nunique()
    colors = plt.cm.rainbow([i / num_tot_threads for i in range(num_tot_threads)])

    grouped = df.groupby("alphabet_size")
    for i, (name, group) in enumerate(grouped):
        subgroups = group.groupby("group_size")
        for j, (name2, subgroup) in enumerate(subgroups):
            ax = axs[j, i]
            for k, (name3, subsubgroup) in enumerate(subgroup.groupby("tot_threads")):
                ax.plot(
                    subsubgroup["num_threads"],
                    subsubgroup["duration"],
                    label=f"tot_threads={name3}",
                    color=colors[k],
                )
            ax.set_title(f"group_size={name2}, alphabet_size={name}")
            ax.set_xlabel("num_threads")
            ax.set_ylabel("time")
            if j == 0 and i == 0:
                ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "plot.png"))


if __name__ == "__main__":
    df = pd.read_csv("accessKernel.csv")
    plot(df, ".")
