from collections import OrderedDict
import csv
import os
import argparse
import pandas as pd
import numpy as np


def apply_queries_tunes(tune_files, query_type):
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
            # remove the rows with "chunk_size" == max_chunk_size except the first occurrence
            max_chunk_size = trimmed_df["num_chunks"].max()
            trimmed_df = trimmed_df.drop(
                trimmed_df[trimmed_df["num_chunks"] == max_chunk_size].index[1:]
            )
            # Find logaritmic fit
            mult, intercept = np.polyfit(
                np.log(trimmed_df["num_queries"]), trimmed_df["num_chunks"], 1
            )

    tune_string = f".{query_type}Kernel_logrel = {{{mult}, {intercept}}}"
    return tune_string


def apply_l2kernel_tunes(tune_files):
    tune_file = tune_files[0]
    with open(tune_file, "r") as f:
        df = pd.read_csv(f)

    # Get "tpb" that minimizes "time"
    best_tpb = df.loc[df["time"].idxmin()]["tpb"]
    return ".ideal_TPB_calculateL2EntriesKernel = " + str(int(best_tpb))


def apply_samples_kernel_tunes(tune_files):
    for file in tune_files:
        with open(file, "r") as f:
            df = pd.read_csv(f)
        if "tpb" in file.lower():
            # Get "tpb" that minimizes "time"
            best_tpb = df.loc[df["time"].idxmin()]["tpb"]
        elif "tot_threads" in file.lower():
            # Get "tot_threads" that minimizes "time"
            best_tot_threads = df.loc[df["time"].idxmin()]["tot_threads"]
    tune_string = f".ideal_TPB_calculateSelectSamplesKernel = {int(best_tpb)}, .ideal_tot_threads_calculateSelectSamplesKernel = {int(best_tot_threads)}"
    return tune_string


def apply_FLK_tune(tune_file):
    with open(tune_file, "r") as f:
        df = pd.read_csv(f)
    # Get "tpb" that minimizes "time"
    best_tpb = df.loc[df["time"].idxmin()]["tpb"]
    return ".ideal_TPB_fillLevelKernel = " + str(int(best_tpb))


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
    rank_tunes = [f for f in tune_csvs if f.split("/")[-1].startswith("rank")]
    select_tunes = [f for f in tune_csvs if f.split("/")[-1].startswith("select")]
    l2kernel_tunes = [
        f for f in tune_csvs if f.split("/")[-1].startswith("calculateL2EntriesKernel")
    ]
    samples_tunes = [
        f
        for f in tune_csvs
        if f.split("/")[-1].startswith("calculateSelectSamplesKernel")
    ]
    FLK_tunes = [f for f in tune_csvs if f.split("/")[-1].startswith("fillLevelKernel")]
    GPU_name = pd.read_csv(access_tunes[0])["GPU_name"][0]
    tune_string = '{"' + GPU_name + '"' + ", IdealConfigs {"
    tune_string += apply_queries_tunes(access_tunes, "access") + ","
    tune_string += apply_queries_tunes(rank_tunes, "rank") + ","
    tune_string += apply_queries_tunes(select_tunes, "select") + ","
    tune_string += apply_l2kernel_tunes(l2kernel_tunes) + ","
    tune_string += apply_samples_kernel_tunes(samples_tunes) + ","
    tune_string += apply_FLK_tune(FLK_tunes[0])
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
