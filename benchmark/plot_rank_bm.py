import pandas as pd
import os
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import argparse


def plot_access_bm(bm, save_path):
    byte_bm = bm[bm["param.alphabet_size"] <= 256]
    short_bm = bm[bm["param.alphabet_size"] > 256]
    for bm, bm_name in [(byte_bm, "byte"), (short_bm, "short")]:
        # group by "SDSL" in name
        sdsl_bm = bm[bm["name"].str.contains("SDSL")]
        own_bm = bm[~bm["name"].str.contains("SDSL")]

        # 2 cols with enough rows for all data sizes
        n_data_sizes = len(bm["param.data_size"].unique())
        n_rows = n_data_sizes // 2 + (n_data_sizes % 2 > 0)
        fig, axs = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows))

        # One color for each num_queries
        n_query_sizes = len(bm["param.num_queries"].unique())
        colors = plt.cm.rainbow([i / n_query_sizes for i in range(n_query_sizes)])
        for i, (name, group) in enumerate(sdsl_bm.groupby("param.data_size")):
            if n_rows == 1:
                ax = axs[i % 2]
            else:
                ax = axs[i // 2, i % 2]
            # group by "param.num_queries"
            for j, (name2, group2) in enumerate(group.groupby("param.num_queries")):
                ax.plot(
                    group2["param.alphabet_size"],
                    group2["real_time"],
                    linestyle="--",
                    marker="o",
                    color=colors[j],
                )

            ax.set_title(f"Data size: {name}")
            ax.set_xlabel("Alphabet size")
            ax.set_ylabel("Time (ms)")

        for i, (name, group) in enumerate(own_bm.groupby("param.data_size")):
            if n_rows == 1:
                ax = axs[i % 2]
            else:
                ax = axs[i // 2, i % 2]
            # group by "param-num_queries"
            for j, (name2, group2) in enumerate(group.groupby("param.num_queries")):
                for k, (name3, group3) in enumerate(group2.groupby("param.pin_memory")):
                    if int(name3) == 0:
                        continue
                    ax.plot(
                        group3["param.alphabet_size"],
                        group3["real_time"],
                        linestyle="-" if int(name3) == 1 else ":",
                        marker="o",
                        color=colors[j],
                    )

            ax.set_title(f"Data size: {name}")
            ax.set_xlabel("Alphabet size")
            ax.set_ylabel("Time (ms)")
            ax.set_yscale("log")

        fig.legend(
            handles=[
                plt.Line2D([0], [0], color=colors[i], label=f"num_queries: {num}")
                for i, num in enumerate(bm["param.num_queries"].unique())
            ],
        )

        fig.suptitle("Benchmark Rank")
        plt.savefig(
            os.path.join(save_path, f"bm_rank_{bm_name}.png"),
            dpi=300,
        )


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
    plot_access_bm(df, save_path)
