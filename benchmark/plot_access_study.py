import pandas as pd
import os
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import argparse


def plot_access_bm(bm, save_path):
    n_occs = len(bm["param.occupancy"].unique())
    n_query_sizes = len(bm["param.num_queries"].unique())
    small_alphabet_bm = bm[bm["param.alphabet_size"] < 256]
    large_alphabet_bm = bm[bm["param.alphabet_size"] >= 256]
    for bm_name, bm in [
        ("small_alphabet", small_alphabet_bm),
        ("large_alphabet", large_alphabet_bm),
    ]:
        fig, axs = plt.subplots(
            n_query_sizes, n_occs, figsize=(10 * n_occs, 5 * n_query_sizes)
        )

        n_chunks = len(bm["param.num_chunks"].unique())
        colors = plt.cm.rainbow([i / n_chunks for i in range(n_chunks)])
        for i, (name, group) in enumerate(bm.groupby("param.num_queries")):
            for j, (name2, group2) in enumerate(group.groupby("param.occupancy")):
                ax = axs[i, j]
                for k, (name3, group3) in enumerate(group2.groupby("param.num_chunks")):
                    ax.plot(
                        group3["param.alphabet_size"],
                        group3["real_time"],
                        label="num_chunks " + str(name3),
                        linestyle="--",
                        marker="o",
                        color=colors[k],
                    )

                ax.set_title(f"Num queries: {name}, Occupancy: {name2}")
                ax.set_xlabel("Alphabet size")
                ax.set_ylabel("Time (ms)")

            # Lock the y axis to the row's max value
            row_max = group["real_time"].max()
            for j in range(n_occs):
                axs[i, j].set_ylim(0, row_max)

        # add legend to the first plot
        axs[0, 0].legend(loc="upper left", bbox_to_anchor=(-0.3, 1))
        fig.suptitle("Access Study")
        plt.savefig(
            os.path.join(save_path, "bm_access_study_" + bm_name + ".png"),
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
