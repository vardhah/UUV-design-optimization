import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from ..utils import get_data_file_path
from .utils import verify_freecad_existence


def estimate_nose(a, d, x, n):
    return 0.5 * d * np.power((1 - np.power(((x - a) / a), 2)), (1 / n))


def estimate_tail(a, b, c, d, x, theta):
    theta = theta * math.pi / 180
    y1 = 0.5 * d
    y2 = np.power((x - a - b), 2) * ((3 * d) / (2 * c * c) - math.tan(theta) / c)
    y3 = ((d / (c * c * c)) - math.tan(theta) / (c * c)) * np.power((x - a - b), 3)
    return y1 - y2 + y3


def parameterize_hull(hull_params: Dict[str, float]) -> "GliderVessel":
    verify_freecad_existence()
    from .myring_hull import GliderVessel

    vessel = GliderVessel(str(get_data_file_path("cad/Remus_Myring_hull.FCStd")))

    d = hull_params["d"]
    total_len = hull_params["total_length"]
    a = hull_params["a"]
    c = hull_params["c"]
    n = hull_params["n"]
    theta = hull_params["theta"]

    b = total_len - a - c

    r = d * 0.5

    vessel.set_fairing_rad(r)
    vessel.set_nose_len(a)
    vessel.set_fairing_len(b)
    vessel.set_tail_len(c)

    body = vessel.get_fairing_details()
    print("----> body is:", body)
    nose_loc = vessel.get_nose_x_loc()
    tail_loc = vessel.get_tail_x_loc() + a + b
    print("=> nose_x:", nose_loc)
    print("=> tail_x:", tail_loc)

    volume = vessel.get_outer_volume()
    myring = np.array([a, b, c, d, n, theta])
    dp = np.append(myring, volume)

    nose_y = estimate_nose(a, d, nose_loc, n)
    tail_y = estimate_tail(a, b, c, d, tail_loc, theta)
    nose_tail_y = np.append(nose_y, tail_y)
    print("=> nose_tail_y:", nose_tail_y)

    vessel.set_nose_tail_y(nose_tail_y)

    vessel.print_info()


def generate_plot(hull_params: Dict[str, float], save_name: str) -> None:
    d = hull_params["d"]
    total_len = hull_params["total_length"]
    a = hull_params["a"]
    c = hull_params["c"]
    n = hull_params["n"]
    theta = hull_params["theta"]

    b = total_len - a - c

    r = d * 0.5

    x_n = np.array([0, a / 5, 2 * a / 5, 3 * a / 5, 4 * a / 5, a])
    x_t = np.array(
        [
            a + b,
            a + b + c / 5,
            a + b + 2 * c / 5,
            a + b + 3 * c / 5,
            a + b + 4 * c / 5,
            a + b + c,
        ]
    )
    x_b = np.array(
        [
            a,
            a + b * 1 / 6,
            a + b * 2 / 6,
            a + b * 3 / 6,
            a + b * 4 / 6,
            a + b * 5 / 6,
            a + b,
        ]
    )
    y_b = np.array([r, r, r, r, r, r, r])
    y_b = y_b.astype("float64")

    y = estimate_nose(a, d, x_n, n)
    z = estimate_tail(a, b, c, d, x_t, theta)

    plt.figure(figsize=(10, 3))
    plt.plot(x_n, y)
    plt.plot(x_t, z)
    plt.plot(x_b, y_b)
    plt.plot(x_n, -1 * y)
    plt.plot(x_t, -1 * z)
    plt.plot(x_b, -1 * y_b)

    plt.savefig(save_name + ".png")

    plt.close()


def generate_shape(hull_params: Dict[str, float], save_name: str) -> None:
    vessel = parameterize_hull(hull_params)
    vessel.create_stl(1)


def run(args=None):
    parser = ArgumentParser(
        "Myring Hull Generator", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--diameter",
        "-d",
        help="Diameter of the hull",
        type=float,
        default=1.473602877432195442e02,
    )
    parser.add_argument(
        "--nose-length",
        "-a",
        help="Nose Length",
        type=float,
        default=1.241980002068568183e02,
    )
    parser.add_argument(
        "--total-length",
        "-l",
        help="Total Length",
        type=float,
        default=8.356338449468295266e02,
    )
    parser.add_argument(
        "--tail-length",
        "-c",
        help="Tail Length",
        type=float,
        default=5.224223705425469433e01,
    )
    parser.add_argument(
        "--theta",
        "-t",
        help="Tail profile",
        type=float,
        default=1.473602877432195442e02,
    )
    parser.add_argument(
        "--nose", "-n", help="Nose Profile", type=float, default=1.825793131346396692e01
    )
    commands = ["generate-plot", "generate-shape"]

    parser.add_argument(
        "command", choices=sorted(commands), help="The subcommand to execute"
    )
    parser.add_argument(
        "--save-name",
        help="The filename to save to (without extension)",
        type=str,
        default="myring",
    )

    args = parser.parse_args(args)

    hull_params = {
        "d": args.diameter,
        "a": args.nose_length,
        "total_length": args.total_length,
        "c": args.tail_length,
        "theta": args.theta,
        "n": args.nose,
    }

    if args.command == "generate-plot":
        generate_plot(hull_params, args.save_name)
    elif args.command == "generate-shape":
        generate_shape(hull_params, args.save_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    run()
