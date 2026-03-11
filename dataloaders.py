from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "downloads" / "mnist_rotation_new" / "mnist_rotation_new"
DEFAULT_TRAIN_PATH = DEFAULT_DATA_ROOT / "mnist_all_rotation_normalized_float_train_valid.amat"
DEFAULT_TEST_PATH = DEFAULT_DATA_ROOT / "mnist_all_rotation_normalized_float_test.amat"
DEFAULT_CIFAR10_ROOT = ROOT / "downloads" / "cifar-10-batches-py"
DEFAULT_NUM_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_ROTATED_MNIST_VAL_SIZE = 2_000
DEFAULT_CIFAR10_VAL_SIZE = 10_000
CIFAR10_TRAIN_BATCHES = ("data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5")
CIFAR10_TEST_BATCH = "test_batch"


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


def _load_cifar10_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")

    data = batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return data, labels


class Cifar10Dataset(Dataset[tuple[Tensor, Tensor]]):
    """In-memory dataset for the CIFAR-10 python batch files."""

    def __init__(self, root: Path, *, train: bool) -> None:
        if not root.exists():
            raise FileNotFoundError(
                f"CIFAR-10 root not found: {root}. "
                "Point `--data-root` to the extracted `cifar-10-batches-py` directory."
            )

        batch_names = CIFAR10_TRAIN_BATCHES if train else (CIFAR10_TEST_BATCH,)
        batch_paths = [root / batch_name for batch_name in batch_names]
        missing_paths = [path for path in batch_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(f"Missing CIFAR-10 batch files: {missing_paths}")

        images_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        for batch_path in batch_paths:
            images, labels = _load_cifar10_batch(batch_path)
            images_list.append(images)
            labels_list.append(labels)

        self.images = torch.from_numpy(np.concatenate(images_list, axis=0))
        self.labels = torch.from_numpy(np.concatenate(labels_list, axis=0))

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
    val_size: int = DEFAULT_ROTATED_MNIST_VAL_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_valid_dataset = RotatedMnistDataset(train_path)
    test_dataset = RotatedMnistDataset(test_path)

    if val_size <= 0 or val_size >= len(train_valid_dataset):
        raise ValueError(
            f"val_size must be between 1 and {len(train_valid_dataset) - 1}, got {val_size}."
        )

    train_size = len(train_valid_dataset) - val_size
    train_dataset = Subset(train_valid_dataset, range(train_size))
    val_dataset = Subset(train_valid_dataset, range(train_size, len(train_valid_dataset)))

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers >= 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


def create_cifar10_dataloaders(
    *,
    data_root: Path = DEFAULT_CIFAR10_ROOT,
    train_batch_size: int = 128,
    eval_batch_size: int = 256,
    val_size: int = DEFAULT_CIFAR10_VAL_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_valid_dataset = Cifar10Dataset(data_root, train=True)
    test_dataset = Cifar10Dataset(data_root, train=False)

    if val_size <= 0 or val_size >= len(train_valid_dataset):
        raise ValueError(
            f"val_size must be between 1 and {len(train_valid_dataset) - 1}, got {val_size}."
        )

    train_size = len(train_valid_dataset) - val_size
    train_dataset = Subset(train_valid_dataset, range(train_size))
    val_dataset = Subset(train_valid_dataset, range(train_size, len(train_valid_dataset)))

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers >= 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


__all__ = [
    "CIFAR10_TEST_BATCH",
    "CIFAR10_TRAIN_BATCHES",
    "Cifar10Dataset",
    "DEFAULT_CIFAR10_ROOT",
    "DEFAULT_CIFAR10_VAL_SIZE",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_ROTATED_MNIST_VAL_SIZE",
    "DEFAULT_TRAIN_PATH",
    "DEFAULT_TEST_PATH",
    "RotatedMnistDataset",
    "create_cifar10_dataloaders",
    "create_rotated_mnist_dataloaders",
]
