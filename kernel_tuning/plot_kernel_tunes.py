import pandas as pd
import os
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import argparse


def plot_FLK_tune(df, save_path):
    num_sizes = len(df["data_size"].unique())
    df = df.groupby("num_threads")

    fig, axs = plt.subplots(len(df), 2, figsize=(15, 5 * len(df)))
    # One color for each "data_size" value
    colors_data = plt.cm.jet([i / num_sizes for i in range(num_sizes)])
    colors_threads = plt.cm.rainbow([i / len(df) for i in range(len(df))])
    for i, (name, group) in enumerate(df):
        ax = axs[i, 0]
        for j, (data_size, data_group) in enumerate(group.groupby("data_size")):
            ax.plot(
                data_group["num_blocks"],
                data_group["duration"],
                color=colors_data[j],
                label="Data size: " + str(data_size),
            )
        ax.set_title("Block size: " + str(name))
        ax.set_xlabel("Grid size")
        ax.set_ylabel("Time (mus)")
        ax.set_xscale("log")
        ax.set_yscale("log")

        for j, (data_size, data_group) in enumerate(group.groupby("data_size")):
            ax = axs[j, 1]
            ax.plot(
                data_group["num_blocks"],
                data_group["duration"],
                color=colors_threads[i],
                label="Block size: " + str(name),
            )
            ax.set_title("Data size: " + str(data_size))
            ax.set_xlabel("Grid size")
            ax.set_ylabel("Time (mus)")
            ax.set_xscale("log")
            ax.set_yscale("log")

    # All axes should have the same y limits
    min_y_left = min(ax.get_ylim()[0] for ax in axs[:, 0])
    max_y_left = max(ax.get_ylim()[1] for ax in axs[:, 0])
    min_y_right = min(ax.get_ylim()[0] for ax in axs[:, 1])
    max_y_right = max(ax.get_ylim()[1] for ax in axs[:, 1])
    for ax in axs[:, 0]:
        ax.set_ylim(min_y_left, max_y_left)
    for ax in axs[:, 1]:
        ax.set_ylim(min_y_right, max_y_right)

    # Add a legend at the top of the figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left")
    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "FLK_tune.png"))


def plot_hist_kernel_tune(df, save_path):
    occ_limit = 68 * 1024
    num_data_sizes = len(df["data_size"].unique())
    num_block_sizes = len(df["num_threads"].unique())

    fig, axs = plt.subplots(
        num_block_sizes,
        num_data_sizes,
        figsize=(5 * num_data_sizes, 5 * num_block_sizes),
    )
    # One color for each "num_blocks" value
    colors_alphabet = plt.cm.rainbow(
        [
            i / len(df["alphabet_size"].unique())
            for i in range(len(df["alphabet_size"].unique()))
        ]
    )
    df = df.groupby("data_size")
    for i, (name, group) in enumerate(df):
        for j, (block_size, block_group) in enumerate(group.groupby("num_threads")):
            ax = axs[j, i]
            for k, (alphabet_size, num_blocks_group) in enumerate(
                block_group.groupby("alphabet_size")
            ):
                # separate by "used_shmem" = 0 or 1
                used_shmem_group = num_blocks_group.groupby("used_shmem")
                for l, (used_shmem, used_shmem_group) in enumerate(used_shmem_group):
                    ax.plot(
                        used_shmem_group["num_blocks"],
                        used_shmem_group["duration"],
                        color=colors_alphabet[k],
                        label="Alphabet size: " + str(alphabet_size),
                        linestyle="--" if used_shmem == 0 else "-",
                    )
            ax.set_title("Block size: " + str(block_size))
            ax.set_ylabel("Time (mus)")
            ax.set_xlabel("Grid size")
            ax.set_xscale("log")
            ax.set_yscale("log")
            # Set a vertical line where the occupancy is 1
            occ_limit_grid_size = occ_limit / block_size
            ax.axvline(occ_limit_grid_size, color="black", linestyle="--")

    # Set title of each column
    # for i, ax in enumerate(axs[0]):
    #    ax.set_title("Data size: " + str(df.get_group(i)["data_size"].unique()[0]))

    # All axes inside of each column should have the same y limits
    for i in range(num_data_sizes):
        min_y = min(ax.get_ylim()[0] for ax in axs[:, i])
        max_y = max(ax.get_ylim()[1] for ax in axs[:, i])
        for ax in axs[:, i]:
            ax.set_ylim(min_y, max_y)

    # Add a legend at the top of the figure
    handles, labels = axs[0, 0].get_legend_handles_labels()
    # Add line style to labels
    handles.extend(
        [
            plt.Line2D([0], [0], color="black", linestyle="--"),
            plt.Line2D([0], [0], color="black", linestyle="-"),
        ]
    )
    labels.extend(["No shmem", "Used shmem"])
    fig.legend(handles, labels, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "hist_tune.png"))


if __name__ == "__main__":
    # get path to json file as command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=str, help="path to input file")
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

    df = pd.read_csv(path)
    if "FLK" in path:
        plot_FLK_tune(df, save_path)
    else:
        plot_hist_kernel_tune(df, save_path)
