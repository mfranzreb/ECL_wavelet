import pandas as pd
import os
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import argparse


def get_marker(use_shmem, use_small_grid):
    if use_shmem == 1 and use_small_grid == 1:
        return "x"
    elif use_shmem == 1 and use_small_grid == 0:
        return "^"
    elif use_shmem == 0 and use_small_grid == 1:
        return "_"
    else:
        return "|"


def plot_hist_bm(df, save_path):
    # remove the first two rows
    df = df.iloc[2:]
    df = df.reset_index(drop=True)
    # group by rows which have "CUB" in the name
    df_cub = df[df["name"].str.contains("CUB")]
    df_own = df[~df["name"].str.contains("CUB")]

    # group own and cub dfs by "param.data_size"
    df_cub = df_cub.groupby("param.data_size")
    df_own = df_own.groupby("param.data_size")

    # Create a figure with two columns, and n rows so that all the plots fit
    fig, axs = plt.subplots(len(df_cub), 2, figsize=(15, 5 * len(df_cub)))

    for i, (name, group) in enumerate(df_own):
        # group rows whose name has "uint8_t" in it
        byte_group = group[group["name"].str.contains("uint8_t")]
        short_group = group[group["name"].str.contains("uint16_t")]

        cub_group = df_cub.get_group(name)
        byte_cub_group = cub_group[cub_group["name"].str.contains("uint8_t")]
        short_cub_group = cub_group[cub_group["name"].str.contains("uint16_t")]
        ax = axs[i, 0]

        # group by the different values of use_shmem and use_small_grid
        subgroups = byte_group.groupby(["param.use_shmem", "param.use_small_grid"])
        for j, (subname, subgroup) in enumerate(subgroups):
            ax.plot(
                subgroup["param.alphabet_size"],
                subgroup["real_time"],
                color=("green" if subname[0] == 1 else "red"),
                linestyle="-" if subname[1] == 1 else "--",
            )

        ax.plot(
            byte_cub_group["param.alphabet_size"],
            byte_cub_group["real_time"],
            color="blue",
            linestyle="-",
            label="CUB",
        )
        ax.set_title("Data size: " + str(name) + " uint8_t")
        ax.set_xlabel("Alphabet size")
        ax.set_ylabel("Time (ms)")
        ax.set_yscale("log")

        ax = axs[i, 1]
        subgroups = short_group.groupby(["param.use_shmem", "param.use_small_grid"])
        for j, (subname, subgroup) in enumerate(subgroups):
            ax.plot(
                subgroup["param.alphabet_size"],
                subgroup["real_time"],
                color=("green" if subname[0] == 1 else "red"),
                linestyle="-" if subname[1] == 1 else "--",
            )

        ax.plot(
            short_cub_group["param.alphabet_size"],
            short_cub_group["real_time"],
            color="blue",
            linestyle="-",
            label="CUB",
        )

        # Find smallest alphabet size where "could_use_shmem" is 0
        smallest = short_group[short_group["could_use_shmem"] == 0].iloc[0]
        # plot a vertical line at that point
        ax.axvline(
            x=smallest["param.alphabet_size"],
            color="black",
            linestyle="--",
            label="Could not use shared memory",
        )
        ax.set_title("Data size: " + str(name) + " uint16_t")
        ax.set_xlabel("Alphabet size")
        ax.set_ylabel("Time (ms)")
        ax.set_yscale("log")

    # Add a global legend
    fig.legend(
        [
            plt.Line2D([0], [0], color="green"),
            plt.Line2D([0], [0], color="red"),
            plt.Line2D([0], [0], color="blue"),
            plt.Line2D([0], [0], color="black"),
            plt.Line2D([0], [0], linestyle="-"),
            plt.Line2D([0], [0], linestyle="--"),
        ],
        [
            "Use shmem",
            "No shmem",
            "CUB",
            "No shmem possible",
            "Use small grid",
            "No small grid",
        ],
        # Attach the legend to the figure above the first row
        loc="upper right",
    )

    fig.suptitle("Benchmark Histogram construction", fontsize=24)
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(save_path, "bm_hist.png"), dpi=300)


if __name__ == "__main__":
    # get path to json file as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str, help="path to json file")
    parser.add_argument(
        "s", type=str, help="path to directory where pictures should be saved"
    )

    args = parser.parse_args()
    path = args.p
    save_path = args.s

    if not os.path.exists(save_path):
        # Create the directory and any intermediate directories
        os.makedirs(save_path)

    # load json file

    data = json.load(open(path))
    df = pd.DataFrame(data["benchmarks"])
    plot_hist_bm(df, save_path)
