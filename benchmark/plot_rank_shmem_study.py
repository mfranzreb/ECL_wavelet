import pandas as pd
import os
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import argparse


def plot_rank(bm, save_path):
    for i, (name, group) in enumerate(bm.groupby("param.use_shmem")):
        plt.plot(
            group["param.alphabet_size"],
            group["real_time"],
            label="shmem " + str(name),
            linestyle="--" if int(name) == 1 else "-",
            marker="o",
            color="green" if int(name) == 1 else "red",
        )
    plt.title("Rank shmem study")
    plt.xlabel("Alphabet size")
    plt.ylabel("Time (ms)")
    plt.xscale("log")
    plt.legend(loc="upper left", bbox_to_anchor=(-0.3, 1.4))
    plt.savefig(os.path.join(save_path, "rank_shmem_study.png"))


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
    plot_rank(df, save_path)
