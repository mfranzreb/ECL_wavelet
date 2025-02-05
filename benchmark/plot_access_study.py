import pandas as pd
import os
import json
import matplotlib
import numpy as np

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
            2 * n_query_sizes,
            n_occs + 1,
            figsize=(10 * (n_occs + 1), 5 * 2 * n_query_sizes),
        )

        n_chunks = len(bm["param.num_chunks"].unique())
        colors = plt.cm.rainbow([i / n_chunks for i in range(n_chunks)])
        for i, (name, group) in enumerate(bm.groupby("param.num_queries")):
            for j, (name2, group2) in enumerate(group.groupby("param.occupancy")):
                avg_distances = np.array([])
                ax_1 = axs[i * 2, j]
                ax_2 = axs[i * 2 + 1, j]
                for k, (name3, group3) in enumerate(group2.groupby("param.num_chunks")):
                    ax_1.plot(
                        group3["param.alphabet_size"],
                        group3["real_time"],
                        label="num_chunks " + str(name3),
                        linestyle="--",
                        marker="o",
                        color=colors[k],
                    )
                    ax_2.plot(
                        group3["param.alphabet_size"],
                        group3["kernel_time"],
                        linestyle="--",
                        marker="o",
                        color=colors[k],
                    )
                    ax_2.plot(
                        group3["param.alphabet_size"],
                        group3["copy_time"],
                        linestyle=":",
                        marker="o",
                        color=colors[k],
                    )
                    tot_time = np.array([])
                    num_chunks = group3["param.num_chunks"].unique()
                    for l in range(len(group3)):
                        copy_time = group3["copy_time"].iloc[l]
                        kernel_time = group3["kernel_time"].iloc[l]
                        if copy_time > kernel_time:
                            tot_time = np.append(
                                tot_time,
                                copy_time * num_chunks + kernel_time,
                            )
                        else:
                            tot_time = np.append(
                                tot_time,
                                kernel_time * num_chunks + copy_time,
                            )
                    ax_2.plot(
                        group3["param.alphabet_size"],
                        tot_time.reshape(-1, 1),
                        linestyle="-",
                        marker="o",
                        color=colors[k],
                    )
                    # get average distance between kernel and copy time
                    avg_distances = np.append(
                        avg_distances,
                        np.mean(np.abs(group3["kernel_time"] - group3["copy_time"])),
                    )

                ax_1.set_title(f"Num queries: {name}, Occupancy: {name2}")
                ax_1.set_xlabel("Alphabet size")
                ax_1.set_ylabel("Time (ms)")
                ax_2.set_yscale("log")
                print(
                    f"Num queries: {name}, Occupancy: {name2}, Best avg distance: {np.min(avg_distances)} for chunk_size: {group['param.num_chunks'].unique()[np.argmin(avg_distances)]}"
                )
            if int(name) == 5000000:
                # print occupancy that achieves best kernel time for ach alphabet size
                for j, (name2, group2) in enumerate(
                    group.groupby("param.alphabet_size")
                ):
                    best_kernel_time = group2["kernel_time"].min()
                    best_occ = group2.loc[group2["kernel_time"] == best_kernel_time][
                        "param.occupancy"
                    ].values[0]
                    min_time_of_full_occ = group2.loc[group2["param.occupancy"] == 100][
                        "kernel_time"
                    ].min()
                    diff_percent = (
                        (min_time_of_full_occ - best_kernel_time) / min_time_of_full_occ
                    ) * 100
                    print(
                        f"Num queries: {name}, alphabet_size {name2}, Best kernel time: {best_kernel_time} for occupancy: {best_occ}, {diff_percent}% better than full occ."
                    )
            # Lock the y axis to the row's max value
            row_max = group["real_time"].max()
            row_min = group["real_time"].min()
            kernel_max = (group["kernel_time"] * group["param.num_chunks"]).max()
            copy_max = (group["copy_time"] * group["param.num_chunks"]).max()
            kernel_min = (group["kernel_time"] * group["param.num_chunks"]).min()
            copy_min = (group["copy_time"] * group["param.num_chunks"]).min()
            for j in range(n_occs):
                axs[i * 2, j].set_ylim(row_min, row_max)
                # axs[2 * i + 1, j].set_ylim(
                #    min(kernel_min, copy_min), max(kernel_max, copy_max)
                # )

            occ_ax = axs[i * 2, -1]
            for j, (name2, group2) in enumerate(group.groupby("param.occupancy")):
                occ_ax.plot(
                    group2["param.num_chunks"],
                    group2["achieved_occ"],
                    label="Occupancy " + str(name2),
                    linestyle="--",
                    marker="o",
                )
            time_vs_occ_ax = axs[i * 2 + 1, -1]
            for j, (name2, group2) in enumerate(group.groupby("param.num_chunks")):
                for k, (name3, group3) in enumerate(
                    group2.groupby("param.alphabet_size")
                ):
                    time_vs_occ_ax.plot(
                        group3["achieved_occ"],
                        group3["real_time"],
                        label="Num chunks " + str(name2),
                        linestyle="--",
                        marker="o",
                        color=colors[j],
                    )
                    time_vs_occ_ax.set_title(f"Alphabet size: {name3}")
                    break
            time_vs_occ_ax.set_xlabel("Occupancy")
            time_vs_occ_ax.set_ylabel("Kernel Time (ms)")

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
