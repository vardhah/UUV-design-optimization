from pathlib import Path


def get_data_file_path(filename) -> Path:
    parent_dir = Path(__file__).resolve().parent / "data"
    return parent_dir / filename


def add_myring_hull_parameters(parser):
    parser.add_argument(
        "--diameter",
        "-d",
        help="Diameter of the hull",
        type=float,
        default=250,
    )
    parser.add_argument(
        "--nose-length",
        "-a",
        help="Nose Length",
        type=float,
        default=200,
    )
    parser.add_argument(
        "--total-length",
        "-l",
        help="Total Length",
        type=float,
        default=800,
    )
    parser.add_argument(
        "--tail-length",
        "-c",
        help="Tail Length",
        type=float,
        default=200,
    )
    parser.add_argument(
        "--theta",
        "-t",
        help="Tail profile",
        type=float,
        default=25,
    )
    parser.add_argument("--nose", "-n", help="Nose Profile", type=float, default=2)
