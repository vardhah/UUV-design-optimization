import os
import pathlib

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ..utils import get_data_file_path
from .model import DragNet
from .utils import scale_sim_data

matplotlib.use("agg")


def train_dragnet(batch_size: int, epochs: int, save_dir: str):
    """Train the DragNet model"""
    train_set = np.loadtxt(
        get_data_file_path("surrogate/train_data.txt"),
        delimiter=" ",
        skiprows=0,
        dtype=np.float32,
    )

    train_data = scale_sim_data(train_set[:, :-1])
    train_labels = train_set[:, -1]
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=0.1, random_state=42
    )
    train_set = TensorDataset(
        torch.from_numpy(train_data), torch.from_numpy(train_labels)
    )

    valid_set = TensorDataset(
        torch.from_numpy(valid_data), torch.from_numpy(valid_labels)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DragNet(input_size=6, output_size=1)
    learning_rate = 0.001

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.L1Loss()

    save_dir = pathlib.Path(save_dir)
    if not save_dir.exists():
        os.makedirs(save_dir, exist_ok=True)

    history = []
    best_valid_loss = 10e33
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device).view((-1, 1))
            output = model(X)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for X, y in valid_loader:
                X = X.to(device)
                y = y.to(device).view((-1, 1))
                output = model(X)
                loss = loss_fn(output, y)
                valid_loss += loss.item()

        history.append(
            (
                avg_train_loss := (train_loss / len(train_loader)),
                avg_valid_loss := (valid_loss / len(valid_loader)),
            )
        )

        if best_valid_loss > avg_valid_loss:
            torch.save(model.state_dict(), save_dir / "weights_best.pt")
            best_valid_loss = avg_valid_loss

        print(
            f"DragNet(Training) - Epoch {epoch} | Training Loss: {avg_train_loss} | Validation Loss: {avg_valid_loss}"
        )

    history = np.array(history)
    plt.plot(history[:, 0], label="Training")
    plt.plot(history[:, 1], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.savefig(save_dir / "train_history.png")


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
