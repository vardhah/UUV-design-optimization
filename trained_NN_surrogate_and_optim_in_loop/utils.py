# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:11:45 2021

@author: HPP
"""
import glob
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

device = torch.device("cpu")
import copy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("tkagg")


class utilities:
    def __init__(self):
        self.lr = 0

    def set_lr(self, lr):
        self.lr = lr

    def set_lr_auto(self):
        self.lr = np.random.choice(np.logspace(-3, 0, base=10))

    def get_optimizer(self, model):
        optimizer_class = optim.Adam
        # print('***learning rate is:',self.lr)
        return optimizer_class(model.parameters(), lr=0.001)

    def get_lossfunc(self, net_type):
        if net_type == "S":
            return nn.L1Loss()
        elif net_type == "T":
            return nn.BCELoss()


def data_partioning(self, data, population):
    # based on population
    a_list = np.arange(data.shape[0])
    np.random.shuffle(a_list)
    a_split = np.array_split(a_list, population)
    data_split = []
    for i in range(len(a_split)):
        data_split.append(data[a_split[i]])
    return data_split


def append_data_splitmode(data, new_data, population):
    snew = np.array_split(new_data, population)
    for i in range(population):
        data[i] = np.append(data[i], snew[i], axis=0)
    return data


def append_data_randomly_splitmode(data, new_data, population):
    sel_pop = np.random.randint(population, size=1)
    # print('selected population:',sel_pop,"data size is", len(data) )
    data[sel_pop[0]] = np.append(data[sel_pop[0]], new_data, axis=0)
    return data


def scale_data(data, ranges):
    minimum = []
    total_range = []
    for i in range(data.shape[1]):
        minimum.append(ranges[2 * i])
        total_range.append((ranges[2 * i + 1] - ranges[2 * i]))
    # print('min is:',minimum,'Range is:',total_range,'data b/f scaling is:',data)
    minimum = np.array(minimum).reshape(1, -1)
    total_range = np.array(total_range).reshape(1, -1)
    data = np.divide((data - minimum), total_range)
    # print('min is:',minimum,'Range is:',total_range,'data a/f scaling is:',data)
    return data


def rescale_data(data, ranges, mask):
    # print('data b/f rescaling:',data)
    for i in range(len(mask)):
        # print('i is:',i,'mask is:',mask[i])

        if mask[i] == "real":
            data[:, i] = (data[:, i] * (ranges[2 * i + 1] - ranges[2 * i])) + ranges[
                2 * i
            ]
        elif mask[i] == "int":
            data[:, i] = (data[:, i] * (ranges[2 * i + 1] - ranges[2 * i])) + ranges[
                2 * i
            ]
            data[:, i] = np.array(data[:, i], dtype=np.int16)
    # print('data after rescaling:',data)
    return data


# on a given input prepare data for training
def data_preperation(data, mask, ranges, cat):
    for i in range(len(mask)):
        # print('i is:',i,'mask is:',mask[i])
        if mask[i] == "real":
            data[:, i] = (data[:, i] - ranges[2 * i]) / (
                ranges[2 * i + 1] - ranges[2 * i]
            )
        elif mask[i] == "int":
            enc = OneHotEncoder(categories=cat[i])
            data[:, i] = enc.transform(data[:, i]).toarray()
    return data


def update_lbtm(data, util_commandsize):
    return data[0:size, :]


def label_data(data, stest_pred):
    # create label for data(if predicted vale is >/< 10% of error then it labels it '1' or else it is '0')
    ones = np.ones(stest_pred.shape[0])
    zeros = np.zeros(stest_pred.shape[0])
    # print('test shape:',test_data[:,-1].shape,'zeros shape:',zeros.shape,'ones shape:',ones.shape,'stest shape',stest_pred.flatten().shape)
    result = np.where(
        np.absolute((data[:, -1] - stest_pred.flatten()))
        > (0.05 * np.absolute(data[:, -1])),
        ones,
        zeros,
    )
    data[:, -1] = result
    return data


def choose_samples_epsilon(
    pool_data, pool_pred, selection_prob, num_of_samples, epsilon
):
    a_list = np.arange(pool_data.shape[0])
    # print('pool prediction is:',pool_pred)
    index = np.where(
        pool_pred > selection_prob
    )  # find indices in pool_data which have high probability to fail
    # print('index of high prob is:',index)
    passed_samples_size = len(index[0])
    # print('total high prob samples size are:',passed_samples_size)
    passed_pool_data = pool_data[index[0]]
    leftover = np.delete(a_list, index[0])
    # print('left over samples in pool data are:',leftover)
    failed_pool_data = pool_data[leftover]
    print(
        "size of passed data:",
        passed_pool_data.shape,
        "failed pool data:",
        failed_pool_data.shape,
    )

    num_of_samples_from_greedy = int(
        num_of_samples * epsilon
    )  # find number of samples from being greedy
    # print('number of greedy samples:',num_of_samples_from_greedy)
    # if passed sample bucket is greater than number of sample required from greedy approach
    if passed_samples_size > num_of_samples_from_greedy:
        # print('++++++', 'In if loop')
        selected_samples, pass_pool_leftover = data_split_size(
            passed_pool_data, num_of_samples_from_greedy
        )
        # print('passed pool leftover:',pass_pool_leftover.shape,'selected samples:',selected_samples.shape)
        pool_data = np.concatenate((pass_pool_leftover, failed_pool_data), axis=0)
        num_of_random_samples = num_of_samples - selected_samples.shape[0]
        t_data, rest_pool_data = data_split_size(pool_data, num_of_random_samples)
        selected_samples = np.concatenate((selected_samples, t_data), axis=0)

    else:
        # print('++++++', 'In else loop')
        selected_samples = passed_pool_data
        num_of_random_samples = num_of_samples - selected_samples.shape[0]
        # print('number of random_samples:',num_of_random_samples,'no of passed samples:',selected_samples.shape)
        t_data, rest_pool_data = data_split_size(
            failed_pool_data, num_of_random_samples
        )
        selected_samples = np.concatenate((selected_samples, t_data), axis=0)
    # print("---->Total selected samples :",selected_samples.shape)
    return selected_samples, rest_pool_data


def choose_topb_samples(pool_data, pool_pred, selection_prob, num_of_samples):
    # print('pool_data shape:',pool_data.shape,'pool_pred shape',pool_pred.shape)
    pool_data_sorted = pool_data[np.argsort(-1 * pool_pred[:, 0])]
    pool_pred_sorted = pool_pred[np.argsort(-1 * pool_pred[:, 0])]
    a_list = np.arange(pool_data.shape[0])
    index = np.where(
        pool_pred > selection_prob
    )  # find indices in pool_data which have high probability to fail
    passed_samples_size = len(index[0])
    print("size of passed data:", passed_samples_size)
    selected_samples = pool_data_sorted[:num_of_samples, :]
    rest_pool_data = pool_data_sorted[num_of_samples:, :]
    selected_probability = pool_pred_sorted[:num_of_samples, :].flatten()
    print(
        "shape of selected-samples",
        selected_samples.shape,
        "rest pool data:",
        rest_pool_data.shape,
        "pool_data:",
        pool_data.shape,
    )
    return selected_samples, rest_pool_data, selected_probability


def choose_samples_weighted_diversity(
    data, pred, selection_prob, num_of_beta_samples, num_of_sel_samples
):
    print(
        "beta samples are",
        num_of_beta_samples,
        "selcted samples are:",
        num_of_sel_samples,
    )
    selected, rejected, weights = choose_topb_samples(
        data, pred, selection_prob, num_of_beta_samples
    )
    print("*******shape of weights:", weights.shape)
    closest_samples, leftoversamples = kmeancluster_weighted(
        selected, num_of_sel_samples, weights
    )
    total_leftover = np.concatenate((rejected, leftoversamples), axis=0)
    print(
        "size of in data:",
        data.shape,
        "closeset:",
        closest_samples.shape,
        "leftover:",
        total_leftover.shape,
    )
    return closest_samples, total_leftover


def kmeancluster(X, num_cluster):
    km = KMeans(n_clusters=num_cluster, init="k-means++").fit(X)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    return closest


def kmeancluster_weighted(X, num_cluster, weights):
    total_list = np.arange(X.shape[0])
    km = KMeans(n_clusters=num_cluster, init="k-means++").fit(
        X[:, :-1], sample_weight=weights
    )
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X[:, :-1])
    leftover = np.delete(total_list, closest)
    closest_data = X[closest]
    left_over_data = X[leftover]
    return closest_data, left_over_data


def data_split_size(data, size):
    # on a given dataset return the splitted data=> train_data(based on size),validate_data(leftover)
    a_list = np.arange(data.shape[0])
    np.random.shuffle(a_list)
    alist = a_list[0:size]
    train_data = data[alist]
    d = np.arange(data.shape[0])
    leftover = np.delete(d, alist)
    validate_data = data[leftover]
    return train_data, validate_data


def data_split(data, proportion=0.2):
    a_list = np.arange(data.shape[0])
    np.random.shuffle(a_list)
    alist = a_list[0 : int(data.shape[0] * (1 - proportion))]
    train_data = data[alist]
    d = np.arange(data.shape[0])
    leftover = np.delete(d, alist)
    validate_data = data[leftover]
    return train_data, validate_data


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # nn.init.kaiming_uniform_(m.weight.data)
        torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
        torch.nn.init.constant_(m.bias.data, 0)


def create_datafiles(data, test_fraction=0.1):
    a_list = np.arange(data.shape[0])
    np.random.shuffle(a_list)
    alist = a_list[0 : int(data.shape[0] * (1 - test_fraction))]
    train_data = data[alist]
    d = np.arange(data.shape[0])
    leftover = np.delete(d, alist)
    test_data = data[leftover]
    # print('train_data to create file:',train_data,'test data in create file:',test_data)
    np.savetxt("./data/train_data.txt", train_data, delimiter=" ")
    np.savetxt("./data/test_data.txt", test_data, delimiter=" ")
    return 0


def create_files(data, file_name):
    name_file = "./data/" + file_name + ".txt"
    np.savetxt(name_file, data, delimiter=" ")
    return 0


class SimDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        x_tmp = dataset[:, 0:-1]
        y_tmp = dataset[:, -1]
        # print('X_tmp is:',x_tmp,'Y_tmp is:',y_tmp)

        self.x = torch.tensor(x_tmp, dtype=torch.float32).to(device)
        self.y = torch.tensor(y_tmp, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)  # required

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x[idx, :]
        pol = self.y[idx]
        sample = {preds, pol}
        return preds, pol


def save_model_prediction_plot(file_location):
    """Plot predictions of the model."""
    data_loc = Path(__file__).resolve().parent / "data" / "gt_pred.txt"
    loaded_gt_pred = np.loadtxt(data_loc, delimiter=" ", skiprows=0, dtype=float)
    truth = loaded_gt_pred[:, 0]
    preds = loaded_gt_pred[:, 1]
    percentage_error = np.abs(truth - preds) * 100 / truth
    markers = np.empty(percentage_error.shape)
    plot_colors = np.empty(percentage_error.shape).astype(str)
    worst = np.where(percentage_error > 10)
    bad = np.where(np.logical_and(percentage_error >= 5, percentage_error <= 10))
    best = np.where(percentage_error < 5)

    plot_colors[bad] = "b"
    plot_colors[worst] = "r"
    plot_colors[best] = "g"

    markers[bad] = 5
    markers[worst] = 5
    markers[best] = 5

    plt.plot([1, 10], [1, 10], color="black")
    plt.scatter(truth, preds, color=plot_colors, s=markers)
    plt.xlabel("$F_d$ (Ground Truth)")
    plt.ylabel("$F_d$ (Predicted)")
    plt.xlim([1, 10])
    plt.ylim([1, 10])
    plt.savefig(file_location)


def save_residuals_vs_design_params(file_location):
    preds_loc = Path(__file__).resolve().parent / "data" / "gt_pred.txt"
    test_set_loc = (
        Path(__file__).resolve().parent / "data" / "dataware" / "test_data.txt"
    )
    loaded_gt_pred = np.loadtxt(preds_loc, delimiter=" ", skiprows=0, dtype=float)
    loaded_test_set = np.loadtxt(test_set_loc, delimiter=" ", skiprows=0, dtype=float)
    print(loaded_test_set.shape, loaded_gt_pred.shape)
    assert np.allclose(loaded_test_set[:, -1], loaded_gt_pred[:, 0])
    param_indexes = {"a": 0, "b": 1, "c": 2, "d": 3, "n": 4, "\\theta": 5}
    fd_pred = loaded_gt_pred[:, 1]
    fd_gt = loaded_gt_pred[:, 0]
    # residuals = np.abs(fd_pred - fd_gt)
    param_units = {
        "a": "mm",
        "b": "mm",
        "c": "mm",
        "d": "mm",
        "n": "",
        "\\theta": "degrees",
    }
    percentage_error = np.abs(fd_gt - fd_pred) * 100 / fd_gt
    markers = np.empty(percentage_error.shape)
    plot_colors = np.empty(percentage_error.shape).astype(str)
    worst = np.where(percentage_error > 10)
    bad = np.where(np.logical_and(percentage_error >= 5, percentage_error <= 10))
    best = np.where(percentage_error < 5)

    markers[bad] = 5
    markers[worst] = 5
    markers[best] = 5

    plot_colors[bad] = "b"
    plot_colors[worst] = "r"
    plot_colors[best] = "g"
    fig, axes = plt.subplots(nrows=len(param_indexes) // 2, ncols=3, sharey="none")

    gs = axes[2, 1].get_gridspec()
    # remove the underlying axes
    for axs in axes[2:, 0:3]:
        for ax in axs:
            ax.remove()

    axbig = fig.add_subplot(gs[2:, 0:3])
    axbig.plot([1, 10], [1, 10], color="black")
    axbig.scatter(fd_gt, fd_pred, color=plot_colors, s=markers)
    axbig.set_xlabel("$F_d$ (Ground Truth)")
    axbig.set_ylabel("$F_d$ (Predicted)")
    axbig.set_xlim([1, 10])
    axbig.set_ylim([1, 10])
    axbig.grid(linestyle=":")

    for j, ((key, value), ax) in enumerate(zip(param_indexes.items(), axes.flat)):
        design_param = loaded_test_set[:, value]
        param_label = (
            f"${key}$" + " " + (f"({param_units[key]})" if param_units[key] else "")
        )
        ax.scatter(design_param, percentage_error, s=markers, color=plot_colors)
        ax.set_xlabel(param_label)
        ax.xaxis.set_label_position("top")
        ax.set_xlim([min(design_param) - 1, max(design_param) + 1])
        ax.set_ylim([0, 110])
        ax.grid(linestyle=":")
        if j % 3 != 0:
            ax.set_yticklabels([])
        # ax.set_yticklabels([])

    plt.subplots_adjust(wspace=0.05, hspace=0.4)
    fig.text(0.04, 0.6, "Percentage Error", va="center", rotation=90)
    plt.savefig(file_location)


def _diameter_length_from_nn_vs_foam_results(filename):
    match = re.match(r".*_D(\d*)_L(\d*).csv", filename)
    if match:
        diam = float(match.group(1))
        length = float(match.group(2))
        return diam, length


def _partition_by_name(files):
    all_files = {}

    for file in files:
        v = _diameter_length_from_nn_vs_foam_results(file)
        if not all_files.get(v):
            all_files[v] = [None, None]
        if "_foam_" in file:
            all_files[v][0] = file
        else:
            all_files[v][1] = file

    return all_files


def plt_nn_vs_foam_data_bo_lcb_plot(dist_ax, drag_ax, running_minimum, euclid, color):
    drag_ax.plot(
        running_minimum,
        color=color,
        marker=".",
        markersize=10,
        label=f"$\min (F_d)$ = {round(np.min(running_minimum), 3)}",
    )
    dist_ax.plot(euclid, color=color, marker=".", markersize=10)
    dist_ax.set_ylabel("$L_2$ distances between samples")
    drag_ax.set_xlabel("Number of evaluated designs")
    dist_ax.set_xlabel("Number of evaluated designs")
    drag_ax.set_ylabel("Drag Force (${F_d}$)")
    drag_ax.legend(loc="upper right")


def save_nn_vs_foam_bo_lcb(file_location):
    all_files = glob.glob(f"{Path(__file__).parent}/exp_NNvsFoam_BO/*.csv")
    all_files = _partition_by_name(files=all_files)
    fig_layout = [[f"a{j}", f"b{j}"] for j in range(len(all_files) * 2)]

    inserted_count = 0
    for j in range(1, len(all_files) * 2):
        if j % 2 == 0:
            fig_layout.insert(j, [".", "."])
            inserted_count += 1
    total_height_offset = 0.001 * inserted_count
    axes_height = (1 - total_height_offset) / (len(fig_layout) - inserted_count)

    fig, axes = plt.subplot_mosaic(
        fig_layout,
        sharey=False,
        sharex=False,
        gridspec_kw=dict(
            height_ratios=[
                axes_height if item != [".", "."] else total_height_offset
                for item in fig_layout
            ]
        ),
        figsize=(15, 15),
    )

    j = 0
    for (k, v) in all_files.items():
        diam, length = k
        df_foam = pd.read_csv(v[0], header=None)
        df_nn = pd.read_csv(v[1], header=None)
        bo_lcb_foam_params = np.asarray(df_foam.loc[:, 0:5])
        bo_lcb_nn_params = np.asarray(df_nn.loc[:, 0:5])

        bo_lcb_foam_drag = np.asarray(df_foam[6])
        bo_lcb_nn_drag = np.asarray(df_nn[6])
        min_foam_drag = np.minimum.accumulate(bo_lcb_foam_drag, axis=0)
        min_nn_drag = np.minimum.accumulate(bo_lcb_nn_drag, axis=0)

        dist_foam = np.diff(bo_lcb_foam_params, axis=0)
        dist_foam_euclid = np.sqrt((dist_foam**2).sum(axis=1))

        dist_nn = np.diff(bo_lcb_nn_params, axis=0)
        dist_nn_euclid = np.sqrt((dist_nn**2).sum(axis=1))
        opt_foam_params = bo_lcb_foam_params[np.argmin(bo_lcb_foam_drag)]
        opt_nn_params = bo_lcb_nn_params[np.argmin(bo_lcb_nn_drag)]

        plt_nn_vs_foam_data_bo_lcb_plot(
            axes[f"a{j}"], axes[f"b{j}"], min_foam_drag, dist_foam_euclid, "red"
        )

        plt_nn_vs_foam_data_bo_lcb_plot(
            axes[f"a{j+1}"], axes[f"b{j+1}"], min_nn_drag, dist_nn_euclid, "blue"
        )
        axes[f"a{j}"].text(
            s=f"Diameter = {diam}, Length = {length}",
            x=0.80,
            y=1.1,
            fontdict=dict(fontsize=14, fontweight="bold"),
            transform=axes[f"a{j}"].transAxes,
        )

        opt_text_labels = ["$a$", "$b$", "$c$", "$d$", "$n$", "$\\theta$"]

        axes[f"a{j}"].text(
            s="\n".join(
                [
                    f"{text} = {round(value, 3)}"
                    for text, value in zip(opt_text_labels, opt_foam_params)
                ]
            ),
            x=0.03,
            y=0.65,
            fontdict=dict(fontsize=10),
            transform=axes[f"a{j}"].transAxes,
            bbox=dict(facecolor="white"),
        )

        axes[f"a{j+1}"].text(
            s="\n".join(
                [
                    f"{text} = {round(value, 3)}"
                    for text, value in zip(opt_text_labels, opt_nn_params)
                ]
            ),
            x=0.03,
            y=0.65,
            fontdict=dict(fontsize=10),
            transform=axes[f"a{j+1}"].transAxes,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        axes[f"a{j}"].grid(linestyle=":")
        axes[f"a{j+1}"].grid(linestyle=":")
        axes[f"b{j}"].grid(linestyle=":")
        axes[f"b{j+1}"].grid(linestyle=":")
        j += 2

    fig.subplots_adjust(
        hspace=0.2, wspace=0.2, top=0.95, left=0.07, right=0.99, bottom=0.05
    )

    legend_lines = [
        Line2D([0], [0], color="r", lw=2, marker=".", markersize=10),
        Line2D([0], [0], color="b", lw=2, marker=".", markersize=10),
    ]

    fig.legend(legend_lines, ["OpenFOAM", "Neural Network Surrogate"])
    plt.savefig(file_location)


def save_nn_vs_foam_bo_lcb_time(file_location):
    all_files = glob.glob(f"{Path(__file__).parent}/exp_NNvsFoam_BO/*.csv")
    all_files = _partition_by_name(files=all_files)
    colors = ["red", "green", "blue", "cyan"]
    j = 0

    for k, v in all_files.items():
        diam, length = k
        df_foam = pd.read_csv(v[0], header=None)
        df_nn = pd.read_csv(v[1], header=None)

        foam_time = np.asarray(df_foam[7]) * 1000
        nn_time = np.asarray(df_nn[7]) * 1000
        foam_time_cum_sum = np.cumsum(foam_time)
        nn_time_cum_sum = np.cumsum(nn_time)

        plt.plot(
            np.log(foam_time_cum_sum),
            markersize=5,
            marker=".",
            color=colors[j],
            label=f"openFOAM (\\textbf{{D}}={diam}, \\textbf{{L}}={length})",
        )
        plt.plot(
            np.log(nn_time_cum_sum),
            markersize=5,
            marker=".",
            color=colors[j + 1],
            label=f"NN Surrogate (\\textbf{{D}}={diam}, \\textbf{{L}}={length})",
        )
        plt.grid(linestyle=":")
        plt.ylabel("Log of the time elapsed (ms)")
        plt.xlabel("Number of evaluated designs")
        plt.xlim([0, max(foam_time_cum_sum.shape[0], nn_time_cum_sum.shape[0])])
        plt.ylim([0, 20])
        j += 2

    plt.legend(fontsize=7, loc="upper left", ncol=1)
    plt.tight_layout()
    plt.savefig(file_location)


def run(args=None):
    parser = ArgumentParser(description="utils")
    parser.add_argument(
        "command",
        choices=[
            "save-model-prediction",
            "save-bo-nn-vs-foam",
            "save-bo-nn-vs-foam-time",
            "save-residuals-vs-design-params",
        ],
    )
    parser.add_argument("--filename", default="./gt_prediction.pdf", type=str)

    arguments = parser.parse_args(args)
    if arguments.command == "save-model-prediction":
        save_model_prediction_plot(file_location=arguments.filename)
    elif arguments.command == "save-residuals-vs-design-params":
        save_residuals_vs_design_params(file_location=arguments.filename)
    elif arguments.command == "save-bo-nn-vs-foam":
        save_nn_vs_foam_bo_lcb(file_location=arguments.filename)
    elif arguments.command == "save-bo-nn-vs-foam-time":
        save_nn_vs_foam_bo_lcb_time(file_location=arguments.filename)
    else:
        parser.print_help()


if __name__ == "__main__":
    run()
