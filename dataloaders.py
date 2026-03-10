from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "downloads" / "mnist_rotation_new" / "mnist_rotation_new"
DEFAULT_TRAIN_PATH = DEFAULT_DATA_ROOT / "mnist_all_rotation_normalized_float_train_valid.amat"
DEFAULT_TEST_PATH = DEFAULT_DATA_ROOT / "mnist_all_rotation_normalized_float_test.amat"


class RotatedMnistDataset(Dataset[tuple[Tensor, Tensor]]):
    """In-memory dataset for the `.amat` rotated-MNIST files."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Rotated MNIST file not found: {path}. "
                "Point `--train-path`/`--test-path` to a valid `.amat` file."
            )

        array = np.loadtxt(path, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != (28 * 28 + 1):
            raise ValueError(
                f"Expected 785 columns in {path}, got shape {array.shape}."
            )

        images = array[:, :-1].reshape(-1, 1, 28, 28)
        labels = array[:, -1].astype(np.int64)

        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.images[index], self.labels[index]


def create_rotated_mnist_dataloaders(
    *,
    train_path: Path = DEFAULT_TRAIN_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = RotatedMnistDataset(train_path)
    test_dataset = RotatedMnistDataset(test_path)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_TRAIN_PATH",
    "DEFAULT_TEST_PATH",
    "RotatedMnistDataset",
    "create_rotated_mnist_dataloaders",
]
