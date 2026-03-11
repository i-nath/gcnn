from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS_ROOT = ROOT / "downloads"

MNIST_ROT_URL = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"
MNIST_ROT_ARCHIVE = DOWNLOADS_ROOT / "mnist_rotation_new.zip"
MNIST_ROT_EXTRACT_DIR = DOWNLOADS_ROOT / "mnist_rotation_new" / "mnist_rotation_new"
MNIST_ROT_FILES = (
    "mnist_all_rotation_normalized_float_train_valid.amat",
    "mnist_all_rotation_normalized_float_test.amat",
)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE = DOWNLOADS_ROOT / "cifar-10-python.tar.gz"
CIFAR10_EXTRACT_DIR = DOWNLOADS_ROOT / "cifar-10-batches-py"
CIFAR10_FILES = (
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
    "batches.meta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets into the repo-local downloads directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("mnist-rot", "cifar10"),
        default=("mnist-rot", "cifar10"),
        help="Datasets to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download archives and re-extract datasets even if they already exist.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Using existing archive: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def rotated_mnist_is_ready() -> bool:
    return all((MNIST_ROT_EXTRACT_DIR / filename).exists() for filename in MNIST_ROT_FILES)


def cifar10_is_ready() -> bool:
    return all((CIFAR10_EXTRACT_DIR / filename).exists() for filename in CIFAR10_FILES)


def extract_rotated_mnist(*, force: bool) -> None:
    if rotated_mnist_is_ready() and not force:
        print(f"Rotated MNIST already available at: {MNIST_ROT_EXTRACT_DIR}")
        return

    target_dir = MNIST_ROT_EXTRACT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for filename in MNIST_ROT_FILES:
            path = target_dir / filename
            if path.exists():
                path.unlink()

    print(f"Extracting rotated MNIST into: {target_dir}")
    with zipfile.ZipFile(MNIST_ROT_ARCHIVE) as archive:
        archive.extractall(target_dir)


def extract_cifar10(*, force: bool) -> None:
    if CIFAR10_EXTRACT_DIR.exists() and force:
        shutil.rmtree(CIFAR10_EXTRACT_DIR)

    if cifar10_is_ready() and not force:
        print(f"CIFAR-10 already available at: {CIFAR10_EXTRACT_DIR}")
        return

    print(f"Extracting CIFAR-10 into: {DOWNLOADS_ROOT}")
    with tarfile.open(CIFAR10_ARCHIVE, mode="r:gz") as archive:
        archive.extractall(DOWNLOADS_ROOT, filter="data")


def verify_rotated_mnist() -> None:
    missing = [filename for filename in MNIST_ROT_FILES if not (MNIST_ROT_EXTRACT_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Rotated MNIST extraction incomplete, missing: {missing}")


def verify_cifar10() -> None:
    missing = [filename for filename in CIFAR10_FILES if not (CIFAR10_EXTRACT_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError(f"CIFAR-10 extraction incomplete, missing: {missing}")


def download_rotated_mnist(*, force: bool) -> None:
    download_file(MNIST_ROT_URL, MNIST_ROT_ARCHIVE, force=force)
    extract_rotated_mnist(force=force)
    verify_rotated_mnist()


def download_cifar10(*, force: bool) -> None:
    download_file(CIFAR10_URL, CIFAR10_ARCHIVE, force=force)
    extract_cifar10(force=force)
    verify_cifar10()


def main() -> None:
    args = parse_args()
    DOWNLOADS_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        if dataset == "mnist-rot":
            download_rotated_mnist(force=args.force)
        elif dataset == "cifar10":
            download_cifar10(force=args.force)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    print("Dataset download complete.")


if __name__ == "__main__":
    main()
