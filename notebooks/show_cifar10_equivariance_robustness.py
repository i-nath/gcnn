#!/usr/bin/env python3
# %% [markdown]
# # CIFAR-10 equivariance and robustness
#
# This notebook-style script loads the trained CIFAR-10 base checkpoints from
# `checkpoints/cifar10_sizes/`, visualizes how early layers respond to rotated
# and reflected inputs, and compares prediction robustness under common
# augmentations.

# %%
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    return start


REPO_ROOT = find_repo_root(Path.cwd().resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders import Cifar10Dataset, DEFAULT_CIFAR10_ROOT
from p4mresnet import P4MResNet
from p4resnet import P4ResNet
from resnet import ResNet
from scripts.test_equivariance import (
    act_p4,
    act_p4m_reflection,
    act_p4m_rotation,
    reflect_z2,
    rotate_z2,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = REPO_ROOT / "configs" / "cifar10_sizes.yaml"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "cifar10_sizes"
CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

rng = np.random.default_rng(0)
torch.set_grad_enabled(False)

# %% [markdown]
# ## Load the trained CIFAR-10 models
#
# These paths target the trained base models in the size-sweep checkpoint
# layout that already exists in this repository.

# %%
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing model config: {CONFIG_PATH}")

if not DEFAULT_CIFAR10_ROOT.exists():
    raise FileNotFoundError(
        "Missing CIFAR-10 data directory:\n"
        f"{DEFAULT_CIFAR10_ROOT}\n\n"
        "Run `python scripts/download_datasets.py --datasets cifar10` from the repo root."
    )

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

checkpoint_paths = {
    "resnet": CHECKPOINT_ROOT / "resnet" / "base" / "seed_0" / "best.pt",
    "p4resnet": CHECKPOINT_ROOT / "p4resnet" / "base" / "seed_0" / "best.pt",
    "p4mresnet": CHECKPOINT_ROOT / "p4mresnet" / "base" / "seed_0" / "best.pt",
}

missing_paths = [path for path in checkpoint_paths.values() if not path.exists()]
if missing_paths:
    raise FileNotFoundError(
        "Missing trained CIFAR-10 checkpoints:\n"
        + "\n".join(str(path) for path in missing_paths)
    )


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(model_name: str) -> torch.nn.Module:
    model_cfg = config["models"][model_name]["base"]
    common_kwargs = {
        "n_blocks": int(model_cfg["n_blocks"]),
        "in_channels": int(config["dataset"]["in_channels"]),
        "num_classes": int(config["dataset"]["num_classes"]),
        "channel_dims": [int(x) for x in model_cfg["channel_dims"]],
    }
    if model_name == "resnet":
        return ResNet(**common_kwargs)
    if model_name == "p4resnet":
        return P4ResNet(**common_kwargs)
    if model_name == "p4mresnet":
        return P4MResNet(**common_kwargs)
    raise ValueError(f"Unsupported model name: {model_name}")


def load_trained_model(model_name: str) -> tuple[torch.nn.Module, dict]:
    model = build_model(model_name).to(DEVICE)
    checkpoint = load_checkpoint(checkpoint_paths[model_name], DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


trained_models: dict[str, torch.nn.Module] = {}
checkpoint_info: dict[str, dict] = {}
for model_name in checkpoint_paths:
    model, checkpoint = load_trained_model(model_name)
    trained_models[model_name] = model
    checkpoint_info[model_name] = checkpoint

for model_name, checkpoint in checkpoint_info.items():
    best_eval_acc = checkpoint.get("best_eval_acc")
    metrics = checkpoint.get("metrics", {})
    print(
        f"{model_name:9s} checkpoint={checkpoint_paths[model_name]} "
        f"best_eval_acc={best_eval_acc} metrics={metrics}"
    )

# %% [markdown]
# ## Load CIFAR-10 test data

# %%
test_dataset = Cifar10Dataset(DEFAULT_CIFAR10_ROOT, train=False)
print(f"Test examples: {len(test_dataset)}")
print(f"Image tensor shape: {tuple(test_dataset.images.shape)}")


def image_for_display(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def summarize_feature_map(feature: torch.Tensor) -> np.ndarray:
    feature = feature.detach().cpu()
    if feature.ndim == 3:
        summary = feature.abs().mean(dim=0)
    elif feature.ndim == 4:
        summary = feature.abs().mean(dim=(0, 1))
    elif feature.ndim == 5:
        summary = feature.abs().mean(dim=(0, 1, 2))
    else:
        raise ValueError(f"Unsupported feature rank: {feature.ndim}")
    return summary.numpy()


def topk_predictions(model: torch.nn.Module, image: torch.Tensor, *, k: int = 3) -> list[tuple[str, float]]:
    logits = model(image.unsqueeze(0).to(DEVICE))[0]
    probs = torch.softmax(logits, dim=0)
    values, indices = probs.topk(k)
    return [
        (CIFAR10_CLASS_NAMES[int(index)], float(value))
        for value, index in zip(values.cpu(), indices.cpu())
    ]


def find_consensus_example(max_examples: int = 512) -> int:
    for idx in range(200, min(max_examples, len(test_dataset))):
        image, label = test_dataset[idx]
        preds = []
        for model in trained_models.values():
            pred = int(model(image.unsqueeze(0).to(DEVICE)).argmax(dim=1).item())
            preds.append(pred)
        if all(pred == int(label) for pred in preds):
            return idx
    return 0


sample_index = find_consensus_example()
sample_image, sample_label = test_dataset[sample_index]
print(
    f"Chosen test example index: {sample_index}, "
    f"label={CIFAR10_CLASS_NAMES[int(sample_label)]}"
)

# %% [markdown]
# ## Example predictions under rotation and reflection
#
# The `p4resnet` model is designed for rotation equivariance, while
# `p4mresnet` extends this to include reflections.

# %%
sample_variants = OrderedDict(
    [
        ("clean", sample_image),
        ("rot90", rotate_z2(sample_image.unsqueeze(0), 1).squeeze(0)),
        ("rot180", rotate_z2(sample_image.unsqueeze(0), 2).squeeze(0)),
        ("reflect", reflect_z2(sample_image.unsqueeze(0)).squeeze(0)),
    ]
)

fig, axes = plt.subplots(len(trained_models) + 1, len(sample_variants), figsize=(12, 8))
for column, (variant_name, variant_image) in enumerate(sample_variants.items()):
    axes[0, column].imshow(image_for_display(variant_image))
    axes[0, column].set_title(variant_name, fontsize=11)
    axes[0, column].axis("off")

for row, (model_name, model) in enumerate(trained_models.items(), start=1):
    for column, (variant_name, variant_image) in enumerate(sample_variants.items()):
        top1 = topk_predictions(model, variant_image, k=1)[0]
        axes[row, column].imshow(image_for_display(variant_image))
        axes[row, column].set_title(f"{model_name}\n{top1[0]} ({top1[1]:.2f})", fontsize=10)
        axes[row, column].axis("off")

axes[0, 0].set_ylabel("input", fontsize=12)
axes[1, 0].set_ylabel("prediction", fontsize=12)
axes[2, 0].set_ylabel("prediction", fontsize=12)
axes[3, 0].set_ylabel("prediction", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Hook the first few layers
#
# We inspect the stem and the first two residual blocks. For the equivariant
# models, we compare the transformed activation with the group action applied to
# the original activation.

# %%
LAYER_NAMES = ("stem", "stage1_block1", "stage1_block2")


def first_layers(model: torch.nn.Module) -> OrderedDict[str, torch.nn.Module]:
    return OrderedDict(
        [
            ("stem", model.stem),
            ("stage1_block1", model.stages[0][0]),
            ("stage1_block2", model.stages[0][1]),
        ]
    )


def capture_layer_outputs(model: torch.nn.Module, image: torch.Tensor) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    handles = []

    def save_output(layer_name: str):
        def hook(_module, _inputs, output) -> None:
            outputs[layer_name] = output.detach().cpu()

        return hook

    for layer_name, module in first_layers(model).items():
        handles.append(module.register_forward_hook(save_output(layer_name)))
    try:
        model(image.unsqueeze(0).to(DEVICE))
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def apply_feature_action(model_name: str, feature: torch.Tensor, transform_name: str) -> torch.Tensor:
    if transform_name == "rot90":
        if model_name == "resnet":
            return rotate_z2(feature, 1)
        if model_name == "p4resnet":
            return act_p4(feature.unsqueeze(0), 1).squeeze(0)
        if model_name == "p4mresnet":
            return act_p4m_rotation(feature.unsqueeze(0), 1).squeeze(0)
    if transform_name == "reflect":
        if model_name == "resnet":
            return reflect_z2(feature)
        if model_name == "p4mresnet":
            return act_p4m_reflection(feature.unsqueeze(0)).squeeze(0)
    raise ValueError(f"Unsupported transform {transform_name!r} for model {model_name!r}")


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual - expected).abs().mean()
    denominator = expected.abs().mean().clamp_min(1e-8)
    return float((numerator / denominator).item())


def show_equivariance_grid(model_name: str, transform_name: str) -> None:
    model = trained_models[model_name]
    input_transform = sample_variants[transform_name]
    clean_outputs = capture_layer_outputs(model, sample_image)
    transformed_outputs = capture_layer_outputs(model, input_transform)

    fig, axes = plt.subplots(len(LAYER_NAMES), 3, figsize=(9, 8))
    fig.suptitle(f"{model_name}: {transform_name} feature comparison", fontsize=14)

    for row, layer_name in enumerate(LAYER_NAMES):
        clean_feature = clean_outputs[layer_name][0]
        actual_feature = transformed_outputs[layer_name][0]
        expected_feature = apply_feature_action(model_name, clean_feature, transform_name)
        err = relative_error(actual_feature, expected_feature)

        axes[row, 0].imshow(summarize_feature_map(clean_feature), cmap="magma")
        axes[row, 0].set_title(f"{layer_name}\nclean", fontsize=10)
        axes[row, 1].imshow(summarize_feature_map(actual_feature), cmap="magma")
        axes[row, 1].set_title(f"{layer_name}\nactual", fontsize=10)
        axes[row, 2].imshow(summarize_feature_map(expected_feature), cmap="magma")
        axes[row, 2].set_title(f"{layer_name}\nexpected (err={err:.3e})", fontsize=10)

        for axis in axes[row]:
            axis.axis("off")

    plt.tight_layout()
    plt.show()


def summarize_equivariance_errors(transform_name: str) -> None:
    rows: list[tuple[str, str, float]] = []
    for model_name, model in trained_models.items():
        if transform_name == "reflect" and model_name == "p4resnet":
            continue
        clean_outputs = capture_layer_outputs(model, sample_image)
        transformed_outputs = capture_layer_outputs(model, sample_variants[transform_name])
        for layer_name in LAYER_NAMES:
            clean_feature = clean_outputs[layer_name][0]
            actual_feature = transformed_outputs[layer_name][0]
            expected_feature = apply_feature_action(model_name, clean_feature, transform_name)
            rows.append((model_name, layer_name, relative_error(actual_feature, expected_feature)))

    print(f"Relative errors for {transform_name}:")
    for model_name, layer_name, err in rows:
        print(f"  {model_name:9s} {layer_name:13s} {err:.3e}")

# %% [markdown]
# ## Rotation equivariance in the first few layers

# %%
summarize_equivariance_errors("rot90")
for model_name in trained_models:
    show_equivariance_grid(model_name, "rot90")

# %% [markdown]
# ## Reflection behavior in the first few layers
#
# `p4resnet` is only rotation-equivariant, so reflection comparisons are shown
# for `resnet` and `p4mresnet`.

# %%
summarize_equivariance_errors("reflect")
for model_name in ("resnet", "p4mresnet"):
    show_equivariance_grid(model_name, "reflect")

# %% [markdown]
# ## Robustness to augmentations on a test subset
#
# We report both accuracy and prediction consistency with the clean image.

# %%
ROBUSTNESS_EVAL_SIZE = 1000


def apply_batch_transform(images: torch.Tensor, transform_name: str) -> torch.Tensor:
    if transform_name == "clean":
        return images
    if transform_name == "rot90":
        return rotate_z2(images, 1)
    if transform_name == "rot180":
        return rotate_z2(images, 2)
    if transform_name == "rot270":
        return rotate_z2(images, 3)
    if transform_name == "reflect":
        return reflect_z2(images)
    raise ValueError(f"Unsupported transform: {transform_name}")


def evaluate_robustness(
    model: torch.nn.Module,
    dataset: Cifar10Dataset,
    *,
    max_examples: int,
    batch_size: int = 128,
) -> dict[str, dict[str, float]]:
    indices = list(range(min(max_examples, len(dataset))))
    transform_names = ("clean", "rot90", "rot180", "rot270", "reflect")

    results = {name: {"correct": 0, "total": 0, "consistent": 0} for name in transform_names}

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = torch.stack([dataset[idx][0] for idx in batch_indices]).to(DEVICE)
        labels = torch.tensor([int(dataset[idx][1]) for idx in batch_indices], device=DEVICE)

        clean_logits = model(images)
        clean_predictions = clean_logits.argmax(dim=1)

        for transform_name in transform_names:
            transformed_images = apply_batch_transform(images, transform_name)
            logits = model(transformed_images)
            predictions = logits.argmax(dim=1)

            results[transform_name]["correct"] += int((predictions == labels).sum().item())
            results[transform_name]["total"] += int(labels.numel())
            results[transform_name]["consistent"] += int((predictions == clean_predictions).sum().item())

    summary: dict[str, dict[str, float]] = {}
    for transform_name, counts in results.items():
        total = max(1, counts["total"])
        summary[transform_name] = {
            "accuracy": counts["correct"] / total,
            "consistency": counts["consistent"] / total,
        }
    return summary


robustness_results = {
    model_name: evaluate_robustness(model, test_dataset, max_examples=ROBUSTNESS_EVAL_SIZE)
    for model_name, model in trained_models.items()
}

for model_name, result in robustness_results.items():
    print(model_name)
    for transform_name, metrics in result.items():
        print(
            f"  {transform_name:7s} "
            f"accuracy={metrics['accuracy']:.3f} "
            f"consistency={metrics['consistency']:.3f}"
        )

# %% [markdown]
# ## Plot the robustness comparison

# %%
transform_names = ["clean", "rot90", "rot180", "rot270", "reflect"]
model_names = list(trained_models.keys())
x = np.arange(len(transform_names))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for offset, model_name in enumerate(model_names):
    acc_values = [robustness_results[model_name][name]["accuracy"] for name in transform_names]
    cons_values = [robustness_results[model_name][name]["consistency"] for name in transform_names]
    axes[0].bar(x + (offset - 1) * width, acc_values, width=width, label=model_name)
    axes[1].bar(x + (offset - 1) * width, cons_values, width=width, label=model_name)

axes[0].set_title("Accuracy under augmentation")
axes[0].set_ylabel("accuracy")
axes[0].set_xticks(x)
axes[0].set_xticklabels(transform_names)
axes[0].set_ylim(0.0, 1.0)

axes[1].set_title("Prediction consistency vs clean input")
axes[1].set_ylabel("consistency")
axes[1].set_xticks(x)
axes[1].set_xticklabels(transform_names)
axes[1].set_ylim(0.0, 1.0)

for axis in axes:
    axis.legend()
    axis.grid(alpha=0.2, axis="y")

plt.tight_layout()
plt.show()

# %%
