import os
import pathlib

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ..utils import get_data_file_path
from .model import DragNet

matplotlib.use("agg")


def train_dragnet(batch_size: int, epochs: int, save_dir: str):
    train_set = np.loadtxt(
        get_data_file_path("surrogate/train_data.txt"),
        delimiter=" ",
        skiprows=0,
        dtype=np.float32,
    )

    train_data, train_labels = train_set[:, :-1], train_set[:, -1]
    train_set = TensorDataset(
        torch.from_numpy(train_data), torch.from_numpy(train_labels)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DragNet(input_size=6, output_size=1)
    learning_rate = 0.001

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.L1Loss()

    history = []
    for epoch in range(epochs):
        train_loss = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            loss = loss_fn(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        history.append(avg_loss := (train_loss / len(train_loader)))

        print(f"DragNet(Training) - Epoch {epoch} | Training Loss: {avg_loss}")

    save_dir = pathlib.Path(save_dir)
    if not save_dir.exists():
        os.makedirs(save_dir, exist_ok=True)

    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.savefig(save_dir / "train_history.png")
    torch.save(model.state_dict(), save_dir / "weights.pt")


def run(args):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        "Surrogate model training", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--epochs", type=int, default=30, help="The number of epochs")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="The batch size for the training"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./", help="The directory to save model weights"
    )

    args = parser.parse_args(args)

    train_dragnet(args.batch_size, args.epochs, args.save_dir)
