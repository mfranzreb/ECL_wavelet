import pandas as pd
import os
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import argparse


def plot_RS_bm(bm, save_path):
    # plot the benchmarks
    plt.scatter(bm["param.size"], bm["real_time"], label="GPU time")
    # plot 1.3*10^(-11)x + 0.001 as best CPU time for all sizes
    x = bm["param.size"]
    y = 1.3 * 10 ** (-11) * x + 0.001
    # convert y to milliseconds
    y = [i * 10**3 for i in y]

    plt.plot(x, y, color="red", label="Best CPU time")

    plt.xlabel("Size")
    plt.ylabel("Time (" + time_unit + ")")
    # plt.xscale("log")

    plt.legend()

    plt.suptitle("Benchmark RankSelect construction")
    plt.savefig(
        os.path.join(save_path, "bm_RS.png"),
        dpi=300,
    )
    plt.close()


def plot_WT_bm(bm, save_path):
    # plot the benchmarks
    plt.scatter(bm["param.size"], bm["real_time"], label="GPU time")
    # plot 1.3*10^(-11)x + 0.001 as best CPU time for all sizes
    x = bm["param.size"]
    y = 1.3 * 10 ** (-11) * x + 0.001
    # convert y to milliseconds
    y = [i * 10**3 for i in y]

    plt.plot(x, y, color="red", label="Best CPU time")

    plt.xlabel("Size")
    plt.ylabel("Time (" + time_unit + ")")
    # plt.xscale("log")

    plt.legend()

    plt.suptitle("Benchmark WaveletTree construction")
    plt.savefig(
        os.path.join(save_path, "bm_WT.png"),
        dpi=300,
    )
    plt.close()


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
    # Subtract 1 from the odd family indeices, since each benchmark has two families
    df["family_index"] = df["family_index"].apply(lambda x: x - 1 if x % 2 == 1 else x)

    # Group the rows into different categories by "family_index"
    benchmarks = df.groupby("family_index")
    time_unit = df["time_unit"].unique()[0]
    for bm in benchmarks:
        if "rankselectconstruction" in bm[1]["name"].values[0].lower():
            plot_RS_bm(bm[1], save_path)
        elif "treeconstruction" in bm[1]["name"].values[0].lower():
            plot_WT_bm(bm[1], save_path)
