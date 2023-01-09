from pathlib import Path


def get_data_file_path(filename) -> Path:
    parent_dir = Path(__file__).resolve().parent / "data"
    return parent_dir / filename
