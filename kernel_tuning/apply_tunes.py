from collections import OrderedDict
import csv
import os
import argparse
import pandas as pd
import numpy as np


def apply_access_tunes(tune_files):
    for tune_file in tune_files:
        with open(tune_file, "r") as f:
            df = pd.read_csv(f)
        if "chunks" in tune_file:
            # Group by "num_queries"
            grouped_df = df.groupby("num_queries")
            # find "chunk_size" that minimizes "time"
            best_chunk_sizes = grouped_df["time"].idxmin()
            # remove all other chunk sizes
            trimmed_df = df.loc[best_chunk_sizes]
            slope, intercept = np.polyfit(
                trimmed_df["num_queries"], trimmed_df["num_chunks"], 1
            )
        elif "warps" in tune_file:
            # Make polynomial line of best fit for "num_warps" vs "time"
            square, slope, intercept = np.polyfit(df["num_warps"], df["time"], 2)
            # Find minimum of polynomial
            best_num_warps = -slope / (2 * square)
        elif "alphabet" in tune_file:
            # Find "alphabet_size" from which "time" becomes larger than "time_no_shmem" for all alphabet sizes larger than it
            for i, row in df.iterrows():
                if row["time"] > row["time_no_shmem"]:
                    alphabet_size = row["alphabet_size"]
                else:
                    alphabet_size = 2 * df["alphabet_size"].max()

    tune_string = f".ideal_tot_threads_accessKernel = {int(best_num_warps*32)}, .accessKernel_linrel = {{{slope}, {intercept}}}, .access_counts_shmem_limit = {alphabet_size}"
    return tune_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tune_file", type=str)
    args = parser.parse_args()

    tunes = OrderedDict()
    # Get all csv files in directory of this script
    file_dir = os.path.dirname(os.path.realpath(__file__))
    tune_csvs = [file_dir + "/" + f for f in os.listdir(file_dir) if f.endswith(".csv")]
    # get access csv files
    access_tunes = [f for f in tune_csvs if f.split("/")[-1].startswith("access")]
    GPU_name = pd.read_csv(access_tunes[0])["GPU_name"][0]
    tune_string = '{"' + GPU_name + '"' + ", IdealConfigs {"
    tune_string += apply_access_tunes(access_tunes)
    tune_string += "}}"

    with open(args.tune_file, "r") as f:
        file_string = f.read()
    # Decompose string
    start = file_string.split("configs{")[0]
    entries = file_string.split("configs{")[1].split("};")[0].split("}},")
    # Remove empty entries, or ones with only whitespace or newlines
    entries = [entry for entry in entries if entry.strip() != ""]
    # Add "}}" to entries
    entries = [entry + "}}" for entry in entries]
    end = file_string.split("configs{")[1].split("};")[1]
    if GPU_name in file_string:
        for i, entry in enumerate(entries):
            if GPU_name in entry:
                entries[i] = tune_string
    else:
        entries.append(tune_string)
    new_file_string = start + "configs{" + ",".join(entries) + ",};" + end
    with open(args.tune_file, "w") as f:
        f.write(new_file_string)
