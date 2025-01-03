import pandas as pd
import os
import argparse


def get_best_tune(file):
    df = pd.read_csv(file)
    # Find row of minimum time
    best_row = df.loc[df["duration"].idxmin()]
    best_grid = best_row["num_blocks"]
    best_block = best_row["num_threads"]

    return (best_block, best_grid)


def apply_tunes(tune_file, tunes, GPU_name):
    tune_string = '{"' + GPU_name + '"' + ", IdealConfigs {"
    for kernel_name, (block, grid) in tunes.items():
        tune_string += f".ideal_TPB_{kernel_name} = {block}, .ideal_tot_threads_{kernel_name} = {grid*block}, "

    tune_string = tune_string[:-2] + "}}"

    with open(tune_file, "r") as f:
        file_string = f.read()
    # Decompose string
    start = file_string.split("configs{")[0]
    entries = file_string.split("configs{")[1].split("};")[0].split(",")
    # Remove empty entries, or ones with only whitespace or newlines
    entries = [entry for entry in entries if entry.strip() != ""]
    end = file_string.split("configs{")[1].split("};")[1]
    if GPU_name in file_string:
        for i, entry in enumerate(entries):
            if GPU_name in entry:
                entries[i] = tune_string
    else:
        entries.append(tune_string)
    new_file_string = start + "configs{" + ",".join(entries) + ",};" + end
    with open(tune_file, "w") as f:
        f.write(new_file_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tune_file", type=str)
    args = parser.parse_args()

    tunes = dict()
    # Get all csv files in directory of this script
    file_dir = os.path.dirname(os.path.realpath(__file__))
    tune_csvs = [f for f in os.listdir(file_dir) if f.endswith(".csv")]
    for tune in tune_csvs:
        best_vals = get_best_tune(file_dir + "/" + tune)
        kernel_name = tune.replace(".csv", "")
        tunes[kernel_name] = best_vals

    GPU_name = pd.read_csv(file_dir + "/" + tune_csvs[0]).iloc[0]["GPU_name"]
    apply_tunes(args.tune_file, tunes, GPU_name)
