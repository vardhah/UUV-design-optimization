import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from ..utils import add_myring_hull_parameters, get_data_file_path
from .model import DragNet
from .utils import scale_sim_data

matplotlib.use("agg")


class DragEvaluator:
    def __init__(self, weights_dir, fig_save_format="pdf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = Path(weights_dir).resolve()
        self.model = self._load_model(self.save_dir, self.device)
        self.fig_save_format = fig_save_format

    def evaluate_test_set(self):
        loaded_test_data = np.loadtxt(
            get_data_file_path("surrogate/test_data.txt"),
            delimiter=" ",
            skiprows=0,
            dtype=np.float32,
        )

        test_data = scale_sim_data(loaded_test_data[:, :-1])
        test_labels = loaded_test_data[:, -1]

        test_set = TensorDataset(
            torch.from_numpy(test_data), torch.from_numpy(test_labels)
        )

        test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=False)

        outputs = []
        for X, y in test_loader:
            X = X.to(self.device)
            output = self.model(X)
            outputs.append(output.squeeze())

        preds = torch.cat(outputs).detach().cpu().numpy()
        self.save_test_metrics(preds, test_labels)
        self.save_design_vars_vs_prediction_plot(
            loaded_test_data[:, :-1], preds, test_labels
        )

    def save_test_metrics(self, preds, truth):
        residuals = (truth - preds) / truth
        accuracy = len(np.where(np.abs(residuals) <= 0.05)[0]) / len(truth)
        outliers = np.where(np.abs(residuals) > 0.10)[0]
        eta_outliers = len(outliers) / len(truth)

        metrics = {
            "no_samples": len(truth),
            "accuracy(%)": f"{accuracy*100}%",
            "no_outliers": len(outliers),
            "outliers(%)": f"{eta_outliers*100}%",
        }

        print("==== Begin Test Summary ============")
        print(json.dumps(metrics, indent=2))
        print("==== End Test Summary ===============")

        with (save_name := (self.save_dir / "test_metrics.json")).open(
            "w"
        ) as test_metrics_file:
            json.dump(metrics, test_metrics_file, indent=2)
            print(f"Test metrics saved in {save_name}")

    def save_design_vars_vs_prediction_plot(self, params, preds, truth):
        param_indexes = {"a": 0, "b": 1, "c": 2, "d": 3, "n": 4, "\\theta": 5}
        param_units = {
            "a": "mm",
            "b": "mm",
            "c": "mm",
            "d": "mm",
            "n": "",
            "\\theta": "degrees",
        }

        percentage_error = np.abs(truth - preds) * 100 / truth
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
        axbig.plot([0, 15], [0, 15], color="black")
        axbig.scatter(truth, preds, color=plot_colors, s=markers)
        axbig.set_xlabel("$F_d$ (Ground Truth)")
        axbig.set_ylabel("$F_d$ (Predicted)")
        axbig.set_xlim([0, 15])
        axbig.set_ylim([0, 15])
        axbig.grid(linestyle=":")

        for j, ((key, value), ax) in enumerate(zip(param_indexes.items(), axes.flat)):
            design_param = params[:, value]
            param_label = (
                f"${key}$" + " " + (f"({param_units[key]})" if param_units[key] else "")
            )
            if key == "d":
                design_param *= 2

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
        plt.savefig(
            save_name := (
                self.save_dir / f"preds_vs_design_params.{self.fig_save_format}"
            )
        )
        print(f"prediction plot saved in {save_name}")

    def estimate_drag(
        self, a: float, b: float, c: float, d: float, n: float, theta: float
    ) -> float:
        """Given design parameters a, b, c, r(d/2), n and theta, estimate the drag using surrogate."""
        self._assert_within_design_space(a, b, c, d, n, theta)
        scaled = scale_sim_data(
            np.array([[a, b, c, d / 2, n, theta]], dtype=np.float32)
        )
        estimated_drag = self.model(torch.from_numpy(scaled).to(self.device))
        return estimated_drag.squeeze().item()

    @staticmethod
    def _assert_within_design_space(a, b, c, d, n, theta):
        msg = "Parameter {} should be within range {}. Provided {}"
        min_max = {
            "a": [50, 600],
            "b": [1, 1850],
            "c": [50, 600],
            "d": [100, 400],
            "n": [1, 5],
            "theta": [0, 50],
        }
        assert 50 <= a <= 600, msg.format("a", min_max["a"], a)
        assert 1 <= b <= 1850, msg.format("b", min_max["b"], b)
        assert 50 <= c <= 600, msg.format("c", min_max["c"], c)
        assert 100 <= d <= 400, msg.format("d", min_max["d"], d)
        assert 1 <= n <= 5, msg.format("n", min_max["n"], n)
        assert 0 <= theta <= 50, msg.format("theta", min_max["theta"], theta)

    def drag_from_params_dict(self, hull_params):
        a = hull_params["a"]
        c = hull_params["c"]
        b = hull_params["total_length"] - a - c
        d = hull_params["d"]
        n = hull_params["n"]
        theta = hull_params["theta"]
        return self.estimate_drag(a, b, c, d, n, theta)

    @staticmethod
    def _load_model(weights_dir, device):
        model = DragNet(input_size=6, output_size=1)
        saved_weights = weights_dir / "weights_best.pt"

        model = model.to(device)
        model.load_state_dict(torch.load(saved_weights))
        model.eval()

        return model


def run(args):
    parser = ArgumentParser(
        "Surrogate model evaluation", formatter_class=ArgumentDefaultsHelpFormatter
    )

    commands = ["single", "test-set"]

    parser.add_argument(
        "command", help="The sub-command to execute", choices=sorted(commands)
    )
    parser.add_argument(
        "--weights-dir", help="The model weights directory", required=True
    )
    parser.add_argument(
        "--save-format",
        help="The save format for figures",
        required=False,
        default="pdf",
    )
    add_myring_hull_parameters(parser)

    args = parser.parse_args(args)
    evaluator = DragEvaluator(args.weights_dir, args.save_format)
    if args.command == "single":
        hull_params = {
            "d": args.diameter,
            "a": args.nose_length,
            "total_length": args.total_length,
            "c": args.tail_length,
            "theta": args.theta,
            "n": args.nose,
        }
        drag_estimate = evaluator.drag_from_params_dict(hull_params)
        print("===========Drag Estimation Summary============")
        print(json.dumps({**hull_params, **{"F_d(N)": drag_estimate}}, indent=2))
        print("===========End Estimation Summary============")
    elif args.command == "test-set":
        evaluator.evaluate_test_set()
    else:
        parser.print_help()
