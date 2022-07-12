import glob
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import pi
from pyDOE import *


# latin hypercube sampling-maximize the minimum distance between points, but place the point in a randomized location within its interval
def lhc_samples_maximin(n, dim, ranges, seeds):
    np.random.seed(seeds)
    samples = lhs(dim, samples=n, criterion="maximin")
    for i in range(dim):
        samples[:, i] = (
            samples[:, i] * (ranges[(2 * i + 1)] - ranges[2 * i]) + ranges[2 * i]
        )
    return samples


# latin hypercube sampling-minimize the maximum correlation coefficient
def lhc_samples_corr(n, dim, ranges):
    samples = lhs(dim, samples=n, criterion="corr")
    for i in range(dim):
        samples[:, i] = (
            samples[:, i] * (ranges[(2 * i + 1)] - ranges[2 * i]) + ranges[2 * i]
        )
    return samples


# monte carlo sampling
def random_sampling(dim, n, ranges, seeds):
    np.random.seed(seeds)
    samples = np.random.rand(n, dim)
    for i in range(dim):
        samples[:, i] = (
            samples[:, i] * (ranges[(2 * i + 1)] - ranges[2 * i]) + ranges[2 * i]
        )
    return samples


def get_bo_data(n=50, search_glob="bo_L*"):
    flag = 0
    for file in glob.glob(search_glob):
        placeholder = pd.read_csv(file, delimiter="\t", dtype=float)
        placeholder["min"] = placeholder.Y.expanding(1).min()
        placeholder = placeholder.head(n)
        _place_ = placeholder["min"]
        _place_ = np.array(_place_)
        min_y = _place_.min()
        if placeholder["min"].shape[0] <= n:
            shorten = n - placeholder["min"].shape[0]
            _place_ = np.pad(_place_, (0, shorten), "constant", constant_values=min_y)
        if flag == 0:
            data_bo = _place_.reshape(1, -1)
            flag = 1
        else:
            data_bo = np.concatenate((data_bo, _place_.reshape(1, -1)), axis=0)

    return data_bo


def get_pymoo_data(n=50, search_glob="pymoo_G*"):
    flag = 0
    for file in glob.glob(search_glob):
        placeholder = pd.read_csv(file, delimiter=",", names=list("abntY"), header=None)
        placeholder["min"] = placeholder.Y.expanding(1).min()
        placeholder = placeholder.head(n)
        _place_ = placeholder["min"]
        _place_ = np.array(_place_)
        minY = _place_.min()
        if placeholder["min"].shape[0] <= n:
            shorten = n - placeholder["min"].shape[0]
            _place_ = np.pad(_place_, (0, shorten), "constant", constant_values=minY)
        if flag == 0:
            data_pymoo = _place_.reshape(1, -1)
            flag = 1
        else:
            data_pymoo = np.concatenate((data_pymoo, _place_.reshape(1, -1)), axis=0)

    return data_pymoo


def read_models_data():
    number = 50
    data_loc = Path(__file__).parent / "data"
    bo_lcb = get_bo_data(n=number, search_glob=f"{data_loc}/bo_L*")
    bo_ei = get_bo_data(n=number, search_glob=f"{data_loc}/bo_E*")
    data_ga = get_pymoo_data(n=number, search_glob=f"{data_loc}/pymoo_G*")
    data_nm = get_pymoo_data(n=number, search_glob=f"{data_loc}/pymoo_N*")
    data_lhc = get_pymoo_data(n=number, search_glob=f"{data_loc}/doe_lhc*")
    data_vmc = get_pymoo_data(n=number, search_glob=f"{data_loc}/doe_vmc*")

    data_labels = [
        (bo_ei, "$BO_{EI}$", "r"),
        (bo_lcb, "$BO_{LCB}$", "g"),
        (data_ga, "GA", "b"),
        (data_nm, "NM", "y"),
        (data_lhc, "LHC", "cyan"),
        (data_vmc, "VMC", "magenta"),
    ]

    return data_labels


def save_opt_evolution(filename):
    data_labels = read_models_data()
    fig, ax = plt.subplot_mosaic(
        [
            ["G", "G", "G"],
            ["G", "G", "G"],
            ["A", "B", "C"],
            ["D", "E", "F"],
        ],
        sharex=True,
        sharey=True,
    )
    all_axes = fig.get_axes()[1:]
    avg_plotting_axes = fig.get_axes()[0]
    for j, (data, label, color) in enumerate(data_labels):
        drag = np.average(data, axis=0)
        x = [x for x in range(0, 50)]
        all_axes[j].plot(
            x,
            drag,
            color=color,
            label=label,
            linewidth=1.0,
        )
        avg_plotting_axes.plot(
            x,
            drag,
            color=color,
            linewidth=1.0,
        )
        min_drag = np.min(data, axis=0)
        max_drag = np.max(data, axis=0)
        all_axes[j].fill_between(x, max_drag, min_drag, alpha=0.3, color=color)
        all_axes[j].grid(linestyle=":")

    avg_plotting_axes.grid(linestyle=":")
    twin_ax = avg_plotting_axes.twiny()
    twin_ax.set_xlim([0, 50])
    twin_ax.set_ylim([4, 10])  # Shared Works for all
    avg_plotting_axes.set_xlim([0, 50])  # Shared Works for all

    fig.text(0.05, 0.5, "Drag Force ($F_d$)", va="center", rotation=90)
    fig.text(0.5, 0.03, "Number of evaluated designs", ha="center")
    fig.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=7)
    plt.savefig(filename)


def run(args=None):
    parser = ArgumentParser(description="utils")
    parser.add_argument(
        "command",
        choices=["save-opt-evolution"],
    )
    parser.add_argument("--filename", default="./optimizers.pdf", type=str)

    arguments = parser.parse_args(args)
    if arguments.command == "save-opt-evolution":
        save_opt_evolution(filename=arguments.filename)
    else:
        parser.print_help()


if __name__ == "__main__":
    run()
