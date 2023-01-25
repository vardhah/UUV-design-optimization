import numpy as np


def scale_sim_data(dataset):
    """Scale the dataset into arrays between 0 and 1.

    Notes
    -----
    These values are hardcoded according to the data collection procedure
    """
    ranges = [[50, 600], [1, 1850], [50, 600], [50, 200], [1, 5], [1, 50]]
    values = []
    for idx, (low, high) in enumerate(ranges):
        col = dataset[:, idx]
        values.append(((col - low) / (high - low)).reshape(-1, 1))
    assert np.all(np.array(values) < 1.0)
    return np.concatenate(values, axis=1)
