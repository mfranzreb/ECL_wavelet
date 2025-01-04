from collections import OrderedDict
import csv
import os
import argparse


def get_best_tune(file):
    best_duration = float("inf")
    best_block = None
    best_grid = None

    with open(file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            duration = float(row["duration"])
            if duration < best_duration:
                best_duration = duration
                best_block = int(row["num_threads"])
                if "num_blocks" in row:
                    best_grid = int(row["num_blocks"])

    return (best_block, best_grid)


def apply_tunes(tune_file, tunes, GPU_name):
    tune_string = '{"' + GPU_name + '"' + ", IdealConfigs {"
    for kernel_name, (block, grid) in tunes.items():
        if grid == None:
            tune_string += f".ideal_TPB_{kernel_name} = {block}, "
        else:
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

    tunes = OrderedDict()
    # Get all csv files in directory of this script
    file_dir = os.path.dirname(os.path.realpath(__file__))
    tune_csvs = [f for f in os.listdir(file_dir) if f.endswith(".csv")]
    # order files alphabetically
    tune_csvs.sort()
    for tune in tune_csvs:
        best_vals = get_best_tune(file_dir + "/" + tune)
        kernel_name = tune.replace(".csv", "")
        tunes[kernel_name] = best_vals

    with open(file_dir + "/" + tune_csvs[0], "r") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)  # Get the first row
        gpu_name = first_row["GPU_name"]
    apply_tunes(args.tune_file, tunes, gpu_name)
