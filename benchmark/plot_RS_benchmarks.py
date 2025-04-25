import json
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import os

basedir = os.path.dirname(os.path.abspath(__file__))

cpu_bms_files = {
    "CPU-wide": basedir + "/results/benchmarks_RS_pasta.json",
    "CPU-flat": basedir + "/results/benchmarks_RS_pasta_flat.json",
}
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
plt.rcParams["font.size"] = 14

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
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid()

# add legend of first subfigure
handles = [
    plt.Line2D([0], [0], color="blue", lw=2),
    plt.Line2D([0], [0], color="red", lw=2),
    plt.Line2D([0], [0], color="green", lw=2),
]
labels = [
    "CPU",
    "A100",
    "3090",
]
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    fontsize=14,
)

fig.tight_layout()
# add extra space at the bottom
fig.subplots_adjust(bottom=0.1)
plt.savefig(basedir + "/results/RS_construction.png", dpi=300)

cpu_rank_dfs = {}
for name, df in cpu_dfs.items():
    cpu_rank_dfs[name] = df[df["name"].str.lower().str.contains("binaryrank")]

gpu_rank_dfs = {}
for name, df in gpu_dfs.items():
    gpu_rank_dfs[name] = df[df["name"].str.lower().str.contains("binaryrank")]

# Plot configuration
fig, axes = plt.subplots(2, len(fill_rates), figsize=(5 * len(fill_rates), 10))

for row, is_adversarial in enumerate([0, 1]):
    for col, fill_rate in enumerate(fill_rates):
        ax = axes[row, col]
        for name, df in cpu_rank_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            for query_type in [0, 1]:
                subgroup = group[
                    group["name"].str.lower().str.contains(f"binaryrank<{query_type}>")
                ]
                ax.plot(
                    subgroup["param.size"],
                    # convert to throughput
                    1000
                    * (subgroup["param.num_queries"] / subgroup["median_real_time"]),
                    marker="o",
                    color="blue",
                    linestyle="--" if query_type == 0 else "-",
                )
        for name, df in gpu_rank_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            for query_type in [0, 1]:
                subgroup = group[
                    group["name"].str.lower().str.contains(f"binaryrank<{query_type}>")
                ]
                ax.plot(
                    subgroup["param.size"],
                    # convert to throughput
                    1000
                    * (subgroup["param.num_queries"] / subgroup["median_real_time"]),
                    marker="o",
                    color="red" if name == "A100" else "green",
                    linestyle="--" if query_type == 0 else "-",
                )
        ax.set_title("Fill count: " + str(fill_rate) + " %")
        ax.set_xlabel("bits")
        if col == 0:
            ax.set_ylabel(
                f"Throughput (queries/s) - {'Adversarial' if is_adversarial else 'Uniform'} distribution"
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid()

# manual legend
handles = [
    plt.Line2D([0], [0], color="blue", lw=2),
    plt.Line2D([0], [0], color="red", lw=2),
    plt.Line2D([0], [0], color="green", lw=2),
    plt.Line2D([0], [0], color="black", lw=2, linestyle="--"),
    plt.Line2D([0], [0], color="black", lw=2),
]
labels = [
    "CPU",
    "A100",
    "3090",
    r"Rank$_0$",
    r"Rank$_1$",
]
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=5,
    fontsize=14,
)
fig.tight_layout()
# add extra space at the bottom
fig.subplots_adjust(bottom=0.1)
plt.savefig(basedir + "/results/RS_rank.png", dpi=300)

cpu_select_dfs = {}
for name, df in cpu_dfs.items():
    cpu_select_dfs[name] = df[df["name"].str.lower().str.contains("binaryselect")]

gpu_select_dfs = {}
for name, df in gpu_dfs.items():
    gpu_select_dfs[name] = df[df["name"].str.lower().str.contains("binaryselect")]

# Plot configuration
fig, axes = plt.subplots(2, len(fill_rates), figsize=(5 * len(fill_rates), 10))

for row, is_adversarial in enumerate([0, 1]):
    for col, fill_rate in enumerate(fill_rates):
        ax = axes[row, col]
        for name, df in cpu_select_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            for query_type in [0, 1]:
                subgroup = group[
                    group["name"]
                    .str.lower()
                    .str.contains(f"binaryselect<{query_type}>")
                ]
                ax.plot(
                    subgroup["param.size"],
                    # convert to throughput
                    1000
                    * (subgroup["param.num_queries"] / subgroup["median_real_time"]),
                    marker="o",
                    color="blue" if name == "CPU-wide" else "black",
                    linestyle="--" if query_type == 0 else "-",
                )
        for name, df in gpu_select_dfs.items():
            group = df[df["param.fill_rate"] == fill_rate]
            group = group[group["param.is_adversarial"] == is_adversarial]
            for query_type in [0, 1]:
                subgroup = group[
                    group["name"]
                    .str.lower()
                    .str.contains(f"binaryselect<{query_type}>")
                ]
                ax.plot(
                    subgroup["param.size"],
                    # convert to throughput
                    1000
                    * (subgroup["param.num_queries"] / subgroup["median_real_time"]),
                    marker="o",
                    color="red" if name == "A100" else "green",
                    linestyle="--" if query_type == 0 else "-",
                )
        ax.set_title("Fill count: " + str(fill_rate) + " %")
        ax.set_xlabel("bits")
        if col == 0:
            ax.set_ylabel(
                f"Throughput (queries/s) - {'Adversarial' if is_adversarial else 'Uniform'} distribution"
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid()

# manual legend
handles = [
    plt.Line2D([0], [0], color="blue", lw=2),
    plt.Line2D([0], [0], color="black", lw=2),
    plt.Line2D([0], [0], color="red", lw=2),
    plt.Line2D([0], [0], color="green", lw=2),
    plt.Line2D([0], [0], color="black", lw=2, linestyle="--"),
    plt.Line2D([0], [0], color="black", lw=2),
]
labels = [
    "CPU-wide",
    "CPU-flat",
    "A100",
    "3090",
    r"Select$_0$",
    r"Select$_1$",
]
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=6,
    fontsize=14,
)
fig.tight_layout()
# add extra space at the bottom
fig.subplots_adjust(bottom=0.1)
plt.savefig(basedir + "/results/RS_select.png", dpi=300)
plt.close("all")
