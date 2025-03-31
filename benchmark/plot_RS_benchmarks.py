import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os

basedir = os.path.dirname(os.path.abspath(__file__))

cpu_bms_files = {"CPU": basedir + "/results/benchmarks_RS_pasta.json"}
gpu_bms_files = {
    "A100": basedir + "/results/benchmark_RS_512_16384_A100.json",
    "3090": basedir + "/results/benchmark_RS_512_16384_3090.json",
}


cpu_dfs = {}
for name, file in cpu_bms_files.items():
    with open(file, "r") as f:
        data = json.load(f)
    cpu_dfs[name] = pd.DataFrame(data["benchmarks"])

gpu_dfs = {}
for name, file in gpu_bms_files.items():
    with open(file, "r") as f:
        data = json.load(f)
    gpu_dfs[name] = pd.DataFrame(data["benchmarks"])

# Filter benchmarks containing "Construction"
cpu_construct_dfs = {}
for name, df in cpu_dfs.items():
    cpu_construct_dfs[name] = df[df["name"].str.contains("Construction")]
    fill_rate = cpu_construct_dfs[name]["param.fill_rate"].unique()

gpu_construct_dfs = {}
for name, df in gpu_dfs.items():
    gpu_construct_dfs[name] = df[df["name"].str.contains("Construction")]
    fill_rates = gpu_construct_dfs[name]["param.fill_rate"].unique()

plt.rcParams["font.family"] = "cmr10"

# Plot configuration
fig, axes = plt.subplots(2, len(fill_rates), figsize=(5 * len(fill_rates), 10))

for row, is_adversarial in enumerate([0, 1]):
    for col, fill_rate in enumerate(fill_rates):
        ax = axes[row, col]
        for name, df in cpu_construct_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            ax.plot(
                group["param.size"],
                group["median_real_time"],
                label=name,
                marker="o",
                color="blue",
            )
        for name, df in gpu_construct_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            ax.plot(
                group["param.size"],
                group["median_real_time"],
                label=name,
                marker="o",
                color="red" if name == "A100" else "green",
            )
        ax.set_title("Fill count: " + str(fill_rate) + " %")
        ax.set_xlabel("bits")
        if col == 0:
            ax.set_ylabel(
                f"Time (ms) - {'Adversarial' if is_adversarial else 'Uniform'} distribution"
            )
            if row == 0:
                ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid()

fig.tight_layout()
plt.savefig(basedir + "/results/RS_construction.png", dpi=300)

cpu_rank_0_dfs = {}
for name, df in cpu_dfs.items():
    cpu_rank_0_dfs[name] = df[df["name"].str.lower().str.contains("binaryrank<0>")]

gpu_rank_0_dfs = {}
for name, df in gpu_dfs.items():
    gpu_rank_0_dfs[name] = df[df["name"].str.lower().str.contains("binaryrank<0>")]

# Plot configuration
fig, axes = plt.subplots(2, len(fill_rates), figsize=(5 * len(fill_rates), 10))

for row, is_adversarial in enumerate([0, 1]):
    for col, fill_rate in enumerate(fill_rates):
        ax = axes[row, col]
        for name, df in cpu_rank_0_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            ax.plot(
                group["param.size"],
                # convert to throughput
                1000 * (group["param.num_queries"] / group["median_real_time"]),
                label=name,
                marker="o",
                color="blue",
            )
        for name, df in gpu_rank_0_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            ax.plot(
                group["param.size"],
                1000 * (group["param.num_queries"] / group["median_real_time"]),
                label=name,
                marker="o",
                color="red" if name == "A100" else "green",
            )
        ax.set_title("Fill count: " + str(fill_rate) + " %")
        ax.set_xlabel("bits")
        if col == 0:
            ax.set_ylabel(
                f"Throughput (queries/s) - {'Adversarial' if is_adversarial else 'Uniform'} distribution"
            )
            if row == 0:
                ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid()

fig.tight_layout()
plt.savefig(basedir + "/results/RS_rank_0.png", dpi=300)

cpu_rank_1_dfs = {}
for name, df in cpu_dfs.items():
    cpu_rank_1_dfs[name] = df[df["name"].str.lower().str.contains("binaryrank<1>")]

gpu_rank_1_dfs = {}
for name, df in gpu_dfs.items():
    gpu_rank_1_dfs[name] = df[df["name"].str.lower().str.contains("binaryrank<1>")]

# Plot configuration
fig, axes = plt.subplots(2, len(fill_rates), figsize=(5 * len(fill_rates), 10))

for row, is_adversarial in enumerate([0, 1]):
    for col, fill_rate in enumerate(fill_rates):
        ax = axes[row, col]
        for name, df in cpu_rank_1_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            ax.plot(
                group["param.size"],
                1000 * (group["param.num_queries"] / group["median_real_time"]),
                label=name,
                marker="o",
                color="blue",
            )
        for name, df in gpu_rank_1_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            ax.plot(
                group["param.size"],
                1000 * (group["param.num_queries"] / group["median_real_time"]),
                label=name,
                marker="o",
                color="red" if name == "A100" else "green",
            )
        ax.set_title("Fill count: " + str(fill_rate) + " %")
        ax.set_xlabel("bits")
        if col == 0:
            ax.set_ylabel(
                f"Throughput (queries/s) - {'Adversarial' if is_adversarial else 'Uniform'} distribution"
            )
            if row == 0:
                ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid()

fig.tight_layout()
plt.savefig(basedir + "/results/RS_rank_1.png", dpi=300)

cpu_select_0_dfs = {}
for name, df in cpu_dfs.items():
    cpu_select_0_dfs[name] = df[df["name"].str.lower().str.contains("binaryselect<0>")]

gpu_select_0_dfs = {}
for name, df in gpu_dfs.items():
    gpu_select_0_dfs[name] = df[df["name"].str.lower().str.contains("binaryselect<0>")]

# Plot configuration
fig, axes = plt.subplots(2, len(fill_rates), figsize=(5 * len(fill_rates), 10))

for row, is_adversarial in enumerate([0, 1]):
    for col, fill_rate in enumerate(fill_rates):
        ax = axes[row, col]
        for name, df in cpu_select_0_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            ax.plot(
                group["param.size"],
                1000 * (group["param.num_queries"] / group["median_real_time"]),
                label=name,
                marker="o",
                color="blue",
            )
        for name, df in gpu_select_0_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            ax.plot(
                group["param.size"],
                1000 * (group["param.num_queries"] / group["median_real_time"]),
                label=name,
                marker="o",
                color="red" if name == "A100" else "green",
            )
        ax.set_title("Fill count: " + str(fill_rate) + " %")
        ax.set_xlabel("bits")
        if col == 0:
            ax.set_ylabel(
                f"Throughput (queries/s) - {'Adversarial' if is_adversarial else 'Uniform'} distribution"
            )
            if row == 0:
                ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid()

fig.tight_layout()
plt.savefig(basedir + "/results/RS_select_0.png", dpi=300)

cpu_select_1_dfs = {}
for name, df in cpu_dfs.items():
    cpu_select_1_dfs[name] = df[df["name"].str.lower().str.contains("binaryselect<1>")]

gpu_select_1_dfs = {}
for name, df in gpu_dfs.items():
    gpu_select_1_dfs[name] = df[df["name"].str.lower().str.contains("binaryselect<1>")]

# Plot configuration
fig, axes = plt.subplots(2, len(fill_rates), figsize=(5 * len(fill_rates), 10))

for row, is_adversarial in enumerate([0, 1]):
    for col, fill_rate in enumerate(fill_rates):
        ax = axes[row, col]
        for name, df in cpu_select_1_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            ax.plot(
                group["param.size"],
                1000 * (group["param.num_queries"] / group["median_real_time"]),
                label=name,
                marker="o",
                color="blue",
            )
        for name, df in gpu_select_1_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            ax.plot(
                group["param.size"],
                1000 * (group["param.num_queries"] / group["median_real_time"]),
                label=name,
                marker="o",
                color="red" if name == "A100" else "green",
            )
        ax.set_title("Fill count: " + str(fill_rate) + " %")
        ax.set_xlabel("bits")
        if col == 0:
            ax.set_ylabel(
                f"Throughput (queries/s) - {'Adversarial' if is_adversarial else 'Uniform'} distribution"
            )
            if row == 0:
                ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid()

fig.tight_layout()
plt.savefig(basedir + "/results/RS_select_1.png", dpi=300)
