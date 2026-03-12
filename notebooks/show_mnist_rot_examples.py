#!/usr/bin/env python3
# %% [markdown]
# # Show rotated MNIST examples
#
# This notebook-style Python file loads the rotated-MNIST train and test splits
# from the repo `downloads/` directory and shows a few example digits.

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    return start


REPO_ROOT = find_repo_root(Path.cwd().resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders import DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, RotatedMnistDataset


rng = np.random.default_rng(0)

# %% [markdown]
# ## Load the rotated-MNIST splits
#
# If the data files are missing, run:
# `python scripts/download_datasets.py --datasets mnist-rot`

# %%
train_path = DEFAULT_TRAIN_PATH
test_path = DEFAULT_TEST_PATH

if not train_path.exists() or not test_path.exists():
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    raise FileNotFoundError(
        "Missing rotated-MNIST files:\n"
        + "\n".join(missing)
        + "\n\nRun `python scripts/download_datasets.py --datasets mnist-rot` from the repo root."
    )

train_dataset = RotatedMnistDataset(train_path)
test_dataset = RotatedMnistDataset(test_path)

print(f"Train examples: {len(train_dataset)}")
print(f"Test examples: {len(test_dataset)}")
print(f"Train tensor shape: {tuple(train_dataset.images.shape)}")
print(f"Test tensor shape: {tuple(test_dataset.images.shape)}")

# %% [markdown]
# ## Helpers

# %%
def image_for_display(image) -> np.ndarray:
    return image.squeeze(0).cpu().numpy().T


def show_examples(images, labels, indices, *, title: str) -> None:
    fig, axes = plt.subplots(1, len(indices), figsize=(1.8 * len(indices), 2.8))
    fig.suptitle(title, fontsize=14)

    for axis, idx in zip(axes, indices):
        axis.imshow(image_for_display(images[idx]), cmap="gray")
        axis.set_title(str(int(labels[idx])), fontsize=10)
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def first_example_per_digit(labels) -> list[int]:
    indices: list[int] = []
    for digit in range(10):
        matches = np.flatnonzero(labels.cpu().numpy() == digit)
        if len(matches) == 0:
            raise ValueError(f"No example found for digit {digit}.")
        indices.append(int(matches[0]))
    return indices


def random_indices(dataset_size: int, *, n: int) -> list[int]:
    return rng.choice(dataset_size, size=n, replace=False).tolist()

# %% [markdown]
# ## One example for each digit in the training split

# %%
train_digit_indices = first_example_per_digit(train_dataset.labels)
show_examples(
    train_dataset.images,
    train_dataset.labels,
    train_digit_indices,
    title="Rotated MNIST train examples (digits 0-9)",
)

# %% [markdown]
# ## A few random training examples

# %%
show_examples(
    train_dataset.images,
    train_dataset.labels,
    random_indices(len(train_dataset), n=10),
    title="Random rotated MNIST training examples",
)

# %% [markdown]
# ## A few random test examples

# %%
show_examples(
    test_dataset.images,
    test_dataset.labels,
    random_indices(len(test_dataset), n=10),
    title="Random rotated MNIST test examples",
)

# %%
# Show a few random training images without axis/borders for a presentation slide

import matplotlib.pyplot as plt

borderless_indices = random_indices(len(train_dataset), n=5)
images = train_dataset.images[borderless_indices]
labels = train_dataset.labels[borderless_indices]

fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
for ax, img, label in zip(axes, images, labels):
    ax.imshow(img.squeeze(), cmap='gray', interpolation='nearest')
    ax.set_title(f"{label.item()}", fontsize=16)
    ax.axis('off')  # Remove axis, ticks, and border

plt.tight_layout()
plt.show()
# %%
