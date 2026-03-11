from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloaders import (
    DEFAULT_CIFAR10_ROOT,
    DEFAULT_CIFAR10_VAL_SIZE,
    DEFAULT_NUM_WORKERS,
    create_cifar10_dataloaders,
)
from p4mresnet import P4MUNet
from p4resnet import P4UNet
from resnet import UNet

DATASET_NAME = "cifar10"
MODEL_CHOICES = ("unet", "p4unet", "p4munet")
MODEL_SIZE_CHOICES = ("small", "base", "large")
OPTIMIZER_CHOICES = ("adamw", "sgd")
DATASET_SPECS: dict[str, int] = {
    "in_channels": 3,
    "image_size": 32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a flow-matching model on CIFAR-10.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config with training defaults and model presets.",
    )
    parser.add_argument(
        "--model-size",
        choices=MODEL_SIZE_CHOICES,
        default="base",
        help="Model preset to load from `--config`.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="unet",
        help="Model family to train.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_CIFAR10_ROOT,
        help="Root directory for extracted CIFAR-10 batch files.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=DEFAULT_CIFAR10_VAL_SIZE,
        help="Examples reserved from the CIFAR-10 training split for validation.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_CHOICES,
        default="adamw",
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "cosine"),
        default="none",
        help="Optional learning rate scheduler to apply each optimizer step.",
    )
    parser.add_argument(
        "--cosine-t-max",
        type=int,
        default=None,
        help="CosineAnnealingLR period in optimizer steps. Defaults to total train steps.",
    )
    parser.add_argument(
        "--cosine-eta-min",
        type=float,
        default=0.0,
        help="Minimum learning rate for CosineAnnealingLR.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the selected optimizer.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum used when `--optimizer sgd` is selected.",
    )
    parser.add_argument(
        "--channel-dims",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Stage widths for the selected U-Net.",
    )
    parser.add_argument(
        "--bottleneck-channels",
        type=int,
        default=None,
        help="Optional bottleneck width override.",
    )
    parser.add_argument(
        "--time-embedding-dim",
        type=int,
        default=None,
        help="Optional timestep embedding dimension override.",
    )
    parser.add_argument(
        "--norm",
        choices=("batch", "group", "identity"),
        default="group",
        help="Normalization used inside the U-Net.",
    )
    parser.add_argument(
        "--norm-groups",
        type=int,
        default=8,
        help="Number of groups when `--norm group` is used.",
    )
    parser.add_argument(
        "--activation",
        choices=("relu", "gelu", "silu", "tanh", "identity"),
        default="silu",
        help="Activation used inside the U-Net.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. `cpu`, `cuda`, or `cuda:0`.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Autocast dtype to use when mixed precision is enabled.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Print a training update every N optimizer steps.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap on train batches per epoch for quick smoke tests.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional cap on eval batches for quick smoke tests.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gcnn-flow-matching",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional Weights & Biases entity/team.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional Weights & Biases tags.",
    )
    parser.add_argument(
        "--save-last-checkpoint",
        type=Path,
        default=None,
        help="Optional path for the last training checkpoint written after each epoch.",
    )
    parser.add_argument(
        "--save-best-checkpoint",
        type=Path,
        default=None,
        help="Optional path for the best checkpoint, updated when validation loss improves.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9999,
        help="Exponential moving average decay for model parameters.",
    )
    parser.add_argument(
        "--num-sample-steps",
        type=int,
        default=100,
        help="Number of Euler integration steps used for image generation.",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=16,
        help="Number of images to sample during periodic evaluation.",
    )
    parser.add_argument(
        "--num-final-samples",
        type=int,
        default=16,
        help="Number of images to sample at the end of training.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=16,
        help="Batch size used during sampling.",
    )
    parser.add_argument(
        "--sample-every-epochs",
        type=int,
        default=16,
        help="Save sampled images every N epochs. Set to 0 to disable periodic sampling.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("samples") / "flow_matching",
        help="Directory where generated sample grids are written.",
    )
    return parser


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = yaml.safe_load(path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping at the root of {path}, got {type(config).__name__}")
    return config


def build_config_defaults(
    config: dict[str, Any],
    model: str,
    model_size: str,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {}

    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name")
    if dataset_name is not None and dataset_name != DATASET_NAME:
        raise ValueError(f"Unsupported dataset in config: {dataset_name}")

    training_config = config.get("training", {})
    optimizer_config = training_config.get("optimizer", {})
    optimizer_name = optimizer_config.get("name")
    if optimizer_name is not None:
        optimizer_name = optimizer_name.lower()
        if optimizer_name not in OPTIMIZER_CHOICES:
            raise ValueError(f"Unsupported optimizer in config: {optimizer_name}")
        defaults["optimizer"] = optimizer_name
    if "lr" in optimizer_config:
        defaults["lr"] = optimizer_config["lr"]
    if "weight_decay" in optimizer_config:
        defaults["weight_decay"] = optimizer_config["weight_decay"]
    if "momentum" in optimizer_config:
        defaults["momentum"] = optimizer_config["momentum"]

    mixed_precision_config = training_config.get("mixed_precision", {})
    if "enabled" in mixed_precision_config:
        defaults["amp"] = mixed_precision_config["enabled"]
    if "dtype" in mixed_precision_config:
        defaults["amp_dtype"] = mixed_precision_config["dtype"]

    scheduler_config = training_config.get("lr_scheduler", {})
    scheduler_name = scheduler_config.get("name")
    if scheduler_name is not None:
        if scheduler_name not in {"none", "cosine"}:
            raise ValueError(f"Unsupported lr_scheduler in config: {scheduler_name}")
        defaults["lr_scheduler"] = scheduler_name
    if "t_max" in scheduler_config:
        defaults["cosine_t_max"] = scheduler_config["t_max"]
    if "eta_min" in scheduler_config:
        defaults["cosine_eta_min"] = scheduler_config["eta_min"]

    ema_config = training_config.get("ema", {})
    if "decay" in ema_config:
        defaults["ema_decay"] = ema_config["decay"]

    models_config = config.get("models", {})
    model_config = models_config.get(model, {}).get(model_size)
    if model_config is None and models_config:
        raise ValueError(f"Missing model preset for model={model!r}, model_size={model_size!r}")
    if model_config is not None:
        defaults["channel_dims"] = model_config["channel_dims"]
        if "bottleneck_channels" in model_config:
            defaults["bottleneck_channels"] = model_config["bottleneck_channels"]
        if "time_embedding_dim" in model_config:
            defaults["time_embedding_dim"] = model_config["time_embedding_dim"]
        if "norm" in model_config:
            defaults["norm"] = model_config["norm"]
        if "norm_groups" in model_config:
            defaults["norm_groups"] = model_config["norm_groups"]
        if "activation" in model_config:
            defaults["activation"] = model_config["activation"]
        checkpoint_config = model_config.get("checkpointing", {})
        if "save_last_checkpoint" in checkpoint_config:
            defaults["save_last_checkpoint"] = Path(checkpoint_config["save_last_checkpoint"])
        if "save_best_checkpoint" in checkpoint_config:
            defaults["save_best_checkpoint"] = Path(checkpoint_config["save_best_checkpoint"])

    return defaults


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_parser.add_argument("--model", choices=MODEL_CHOICES, default="unet")
    pre_parser.add_argument("--model-size", choices=MODEL_SIZE_CHOICES, default="base")
    pre_args, _ = pre_parser.parse_known_args()

    parser = build_parser()
    if pre_args.config is not None:
        parser.set_defaults(
            **build_config_defaults(
                config=load_yaml_config(pre_args.config),
                model=pre_args.model,
                model_size=pre_args.model_size,
            )
        )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(args: argparse.Namespace) -> nn.Module:
    common_kwargs: dict[str, Any] = {
        "in_channels": DATASET_SPECS["in_channels"],
        "out_channels": DATASET_SPECS["in_channels"],
        "channel_dims": args.channel_dims,
        "bottleneck_channels": args.bottleneck_channels,
        "time_embedding_dim": args.time_embedding_dim,
        "norm": args.norm,
        "norm_groups": args.norm_groups,
        "activation": args.activation,
    }
    if args.model == "unet":
        return UNet(**common_kwargs)
    if args.model == "p4unet":
        return P4UNet(**common_kwargs)
    if args.model == "p4munet":
        return P4MUNet(**common_kwargs)
    raise ValueError(f"Unsupported model: {args.model}")


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    return create_cifar10_dataloaders(
        data_root=args.data_root,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
    )


def resolve_amp_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype | None:
    if not args.amp:
        return None

    if args.amp_dtype == "float16":
        amp_dtype = torch.float16
    elif args.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported AMP dtype: {args.amp_dtype}")
    if device.type == "cpu" and amp_dtype != torch.bfloat16:
        raise ValueError("CPU mixed precision only supports --amp-dtype bfloat16.")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError(f"Mixed precision is not supported for device type: {device.type}")
    return amp_dtype


def build_grad_scaler(
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.amp.GradScaler | None:
    if not args.amp or device.type != "cuda" or amp_dtype != torch.float16:
        return None
    return torch.amp.GradScaler(device="cuda")


def build_scheduler(
    args: argparse.Namespace,
    optimizer: Optimizer,
    steps_per_epoch: int,
) -> CosineAnnealingLR | None:
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        t_max = (
            args.cosine_t_max
            if args.cosine_t_max is not None
            else args.epochs * steps_per_epoch
        )
        if t_max < 1:
            raise ValueError(f"cosine_t_max must be at least 1, got {t_max}")
        return CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=args.cosine_eta_min,
        )
    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_ema_model(args: argparse.Namespace, model: nn.Module) -> nn.Module | None:
    if not (0.0 < args.ema_decay <= 1.0):
        raise ValueError(f"ema_decay must be in (0, 1], got {args.ema_decay}")

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    ema_model.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    if ema_params.keys() != model_params.keys():
        raise ValueError("EMA model parameters do not match training model parameters")

    for name, ema_param in ema_params.items():
        ema_param.lerp_(model_params[name].detach(), 1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    if ema_buffers.keys() != model_buffers.keys():
        raise ValueError("EMA model buffers do not match training model buffers")

    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name].detach())


def normalize_cifar_images(images: Tensor) -> Tensor:
    return images * 2.0 - 1.0


def sample_flow_matching_batch(images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    x1 = normalize_cifar_images(images)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    t_view = t.view(-1, 1, 1, 1)
    xt = (1.0 - t_view) * x0 + t_view * x1
    target_velocity = x1 - x0
    return xt, t, target_velocity


def denormalize_cifar_images(images: Tensor) -> Tensor:
    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    *,
    num_samples: int,
    image_size: int,
    in_channels: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    num_steps: int,
    batch_size: int,
) -> Tensor:
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}")
    if batch_size < 1:
        raise ValueError(f"sample_batch_size must be at least 1, got {batch_size}")

    model_was_training = model.training
    model.eval()

    generated_batches: list[Tensor] = []
    dt = 1.0 / num_steps
    time_values = torch.linspace(0.0, 1.0 - dt, steps=num_steps, device=device)

    remaining = num_samples
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        x = torch.randn(
            current_batch_size,
            in_channels,
            image_size,
            image_size,
            device=device,
        )
        for t_value in time_values:
            t = torch.full((current_batch_size,), t_value, device=device, dtype=x.dtype)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                velocity = model(x, t)
            x = x + dt * velocity
        generated_batches.append(denormalize_cifar_images(x).cpu())
        remaining -= current_batch_size

    if model_was_training:
        model.train()
    return torch.cat(generated_batches, dim=0)


def save_image_grid(images: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu()
    num_images = images.shape[0]
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)
    height, width = images.shape[-2:]

    canvas = np.ones((grid_rows * height, grid_cols * width, 3), dtype=np.float32)
    for index, image in enumerate(images):
        row = index // grid_cols
        col = index % grid_cols
        image_np = image.permute(1, 2, 0).numpy()
        if image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)
        canvas[
            row * height : (row + 1) * height,
            col * width : (col + 1) * width,
        ] = image_np

    plt.imsave(path, canvas)


def maybe_generate_and_save_samples(
    *,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    filename: str,
) -> Path | None:
    if args.samples_dir is None:
        return None

    num_samples = args.num_eval_samples if filename.startswith("epoch_") else args.num_final_samples
    samples = generate_samples(
        model,
        num_samples=num_samples,
        image_size=DATASET_SPECS["image_size"],
        in_channels=DATASET_SPECS["in_channels"],
        device=device,
        amp_dtype=amp_dtype,
        num_steps=args.num_sample_steps,
        batch_size=args.sample_batch_size,
    )
    output_path = args.samples_dir / filename
    save_image_grid(samples, output_path)
    return output_path


def train_one_epoch(
    model: nn.Module,
    ema_model: nn.Module | None,
    ema_decay: float,
    loader: DataLoader,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler: CosineAnnealingLR | None,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    epoch: int,
    log_interval: int,
    max_batches: int | None,
    global_step: int,
    running_loss_total: float,
    running_examples_total: int,
    wandb_run: Any | None,
) -> tuple[float, int, float, int]:
    model.train()
    total_loss = 0.0
    total_examples = 0

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, (images, _) in enumerate(progress, start=1):
        if max_batches is not None and step > max_batches:
            break

        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            xt, t, target_velocity = sample_flow_matching_batch(images)
            predicted_velocity = model(xt, t)
            loss = F.mse_loss(predicted_velocity, target_velocity)

        optimizer_stepped = False
        if scaler is not None:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= scale_before
        else:
            loss.backward()
            optimizer.step()
            optimizer_stepped = True

        if optimizer_stepped:
            if scheduler is not None:
                scheduler.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, ema_decay)
            global_step += 1

        batch_size = images.shape[0]
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        running_examples_total += batch_size
        running_loss_total += loss.item() * batch_size

        avg_loss = total_loss / total_examples
        global_running_loss = running_loss_total / running_examples_total
        progress.set_postfix(loss=f"{avg_loss:.4f}")

        should_log_step = log_interval > 0 and step % log_interval == 0

        if wandb_run is not None and should_log_step:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/step": global_step,
                    "train/epoch_step": step,
                    "train/global_step": global_step,
                    "train/batch_loss": loss.item(),
                    "train/running_loss": global_running_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

        if should_log_step:
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"train_loss={avg_loss:.4f}"
            )

    return (
        total_loss / total_examples,
        global_step,
        running_loss_total,
        running_examples_total,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    max_batches: int | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    for step, (images, _) in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
        if max_batches is not None and step > max_batches:
            break

        images = images.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            xt, t, target_velocity = sample_flow_matching_batch(images)
            predicted_velocity = model(xt, t)
            loss = F.mse_loss(predicted_velocity, target_velocity)

        batch_size = images.shape[0]
        total_examples += batch_size
        total_loss += loss.item() * batch_size

    return total_loss / total_examples


def get_rng_state() -> dict[str, Any]:
    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def set_rng_state(rng_state: dict[str, Any]) -> None:
    python_state = rng_state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = rng_state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = rng_state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)

    torch_cuda_state = rng_state.get("torch_cuda")
    if torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_state)


def serialize_for_wandb(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [serialize_for_wandb(item) for item in value]
    if isinstance(value, list):
        return [serialize_for_wandb(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_for_wandb(item) for key, item in value.items()}
    return value


def maybe_init_wandb(
    args: argparse.Namespace,
    device: torch.device,
    train_steps_per_epoch: int,
    resume_run_id: str | None,
) -> Any | None:
    if not args.wandb:
        return None
    if wandb is None:
        raise ModuleNotFoundError("wandb is not installed. Install it or run without --wandb.")

    run_config = serialize_for_wandb(vars(args))
    run_config["device"] = str(device)
    run_config["dataset"] = DATASET_NAME
    run_config["train_steps_per_epoch"] = train_steps_per_epoch

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        tags=args.wandb_tags,
        config=run_config,
        id=resume_run_id,
        resume="allow" if resume_run_id is not None else None,
    )


def maybe_save_checkpoint(
    path: Path | None,
    model: nn.Module,
    ema_model: nn.Module | None,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler: CosineAnnealingLR | None,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    running_loss_total: float,
    running_examples_total: int,
    best_eval_loss: float,
    metrics: dict[str, float],
    wandb_run_id: str | None,
) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": None if ema_model is None else ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": None if scaler is None else scaler.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "global_step": global_step,
            "running_loss_total": running_loss_total,
            "running_examples_total": running_examples_total,
            "best_eval_loss": best_eval_loss,
            "metrics": metrics,
            "rng_state": get_rng_state(),
            "wandb_run_id": wandb_run_id,
        },
        path,
    )


def maybe_resume_from_checkpoint(
    path: Path | None,
    model: nn.Module,
    ema_model: nn.Module | None,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
    scheduler: CosineAnnealingLR | None,
    device: torch.device,
    train_steps_per_epoch: int,
) -> tuple[int, int, float, int, float, str | None]:
    if path is None:
        return 1, 0, 0.0, 0, float("inf"), None

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    ema_model_state_dict = checkpoint.get("ema_model_state_dict")
    if ema_model is not None and ema_model_state_dict is not None:
        ema_model.load_state_dict(ema_model_state_dict)
    elif ema_model is not None:
        ema_model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scaler_state_dict = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state_dict is not None:
        scaler.load_state_dict(scaler_state_dict)

    scheduler_state_dict = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        set_rng_state(rng_state)

    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(
        checkpoint.get("global_step", int(checkpoint["epoch"]) * train_steps_per_epoch)
    )
    running_loss_total = float(checkpoint.get("running_loss_total", 0.0))
    running_examples_total = int(checkpoint.get("running_examples_total", 0))
    best_eval_loss = float(
        checkpoint.get(
            "best_eval_loss",
            checkpoint.get("metrics", {}).get("val_loss", float("inf")),
        )
    )
    return (
        start_epoch,
        global_step,
        running_loss_total,
        running_examples_total,
        best_eval_loss,
        checkpoint.get("wandb_run_id"),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    train_loader, val_loader, test_loader = create_dataloaders(args)
    train_steps_per_epoch = (
        min(len(train_loader), args.max_train_batches)
        if args.max_train_batches is not None
        else len(train_loader)
    )
    amp_dtype = resolve_amp_dtype(args, device)
    model = build_model(args).to(device)
    ema_model = build_ema_model(args, model).to(device)
    optimizer = build_optimizer(args, model)
    scaler = build_grad_scaler(args, device, amp_dtype)
    scheduler = build_scheduler(args, optimizer, train_steps_per_epoch)
    (
        start_epoch,
        global_step,
        running_loss_total,
        running_examples_total,
        best_eval_loss,
        resume_wandb_run_id,
    ) = maybe_resume_from_checkpoint(
        path=args.resume,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        device=device,
        train_steps_per_epoch=train_steps_per_epoch,
    )
    wandb_run = maybe_init_wandb(
        args=args,
        device=device,
        train_steps_per_epoch=train_steps_per_epoch,
        resume_run_id=resume_wandb_run_id,
    )

    print(f"Using device: {device}")
    if args.config is not None:
        print(f"Using config: {args.config}")
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
    if wandb_run is not None:
        print(f"W&B run: {wandb_run.name} ({wandb_run.id})")
    print(
        "Mixed precision: "
        + ("disabled" if amp_dtype is None else f"enabled ({args.amp_dtype})")
    )
    print(f"Dataset: {DATASET_NAME}")
    print(f"Data root: {args.data_root}")
    print(f"Validation split size: {args.val_size}")
    print(f"Training model: {args.model}")
    print(f"Train examples: {len(train_loader.dataset)}")
    print(f"Val examples: {len(val_loader.dataset)}")
    print(f"Test examples: {len(test_loader.dataset)}")
    print(f"Samples dir: {args.samples_dir}")

    run_start_time = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        (
            train_loss,
            global_step,
            running_loss_total,
            running_examples_total,
        ) = train_one_epoch(
            model=model,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=device,
            amp_dtype=amp_dtype,
            epoch=epoch,
            log_interval=args.log_interval,
            max_batches=args.max_train_batches,
            global_step=global_step,
            running_loss_total=running_loss_total,
            running_examples_total=running_examples_total,
            wandb_run=wandb_run,
        )
        val_loss = evaluate(
            model=ema_model,
            loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
            max_batches=args.max_eval_batches,
        )
        elapsed_time_sec = time.perf_counter() - epoch_start_time
        total_elapsed_time_sec = time.perf_counter() - run_start_time

        print(
            f"epoch={epoch} "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"elapsed_time={elapsed_time_sec:.2f}s "
            f"best_val_loss={min(best_eval_loss, val_loss):.4f}"
        )

        is_best_checkpoint = val_loss < best_eval_loss
        best_eval_loss = min(best_eval_loss, val_loss)
        sample_path: Path | None = None
        if args.sample_every_epochs > 0 and epoch % args.sample_every_epochs == 0:
            sample_path = maybe_generate_and_save_samples(
                model=ema_model,
                args=args,
                device=device,
                amp_dtype=amp_dtype,
                filename=f"epoch_{epoch:04d}.png",
            )
            if sample_path is not None:
                print(f"saved_samples={sample_path}")
        if wandb_run is not None:
            log_payload: dict[str, Any] = {
                "epoch": epoch,
                "train/global_step": global_step,
                "train/loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "val/loss": val_loss,
                "elapsed_time_sec": elapsed_time_sec,
                "total_elapsed_time_sec": total_elapsed_time_sec,
                "best_val_loss": best_eval_loss,
            }
            if sample_path is not None:
                log_payload["samples/val_grid"] = wandb.Image(str(sample_path))
            wandb_run.log(log_payload, step=global_step)
            wandb_run.summary["best_val_loss"] = best_eval_loss

        maybe_save_checkpoint(
            path=args.save_last_checkpoint,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            args=args,
            epoch=epoch,
            global_step=global_step,
            running_loss_total=running_loss_total,
            running_examples_total=running_examples_total,
            best_eval_loss=best_eval_loss,
            metrics={
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_time_sec": elapsed_time_sec,
                "total_elapsed_time_sec": total_elapsed_time_sec,
            },
            wandb_run_id=None if wandb_run is None else wandb_run.id,
        )
        if is_best_checkpoint:
            maybe_save_checkpoint(
                path=args.save_best_checkpoint,
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                running_loss_total=running_loss_total,
                running_examples_total=running_examples_total,
                best_eval_loss=best_eval_loss,
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "elapsed_time_sec": elapsed_time_sec,
                    "total_elapsed_time_sec": total_elapsed_time_sec,
                },
                wandb_run_id=None if wandb_run is None else wandb_run.id,
            )

    test_loss = evaluate(
        model=ema_model,
        loader=test_loader,
        device=device,
        amp_dtype=amp_dtype,
        max_batches=args.max_eval_batches,
    )
    print(f"final_test_loss={test_loss:.4f}")
    final_sample_path = maybe_generate_and_save_samples(
        model=ema_model,
        args=args,
        device=device,
        amp_dtype=amp_dtype,
        filename="final_samples.png",
    )
    if final_sample_path is not None:
        print(f"final_samples={final_sample_path}")
    if wandb_run is not None:
        final_log_payload: dict[str, Any] = {
            "test/final_loss": test_loss,
        }
        if final_sample_path is not None:
            final_log_payload["samples/final_grid"] = wandb.Image(str(final_sample_path))
        wandb_run.log(final_log_payload, step=global_step)
        wandb_run.summary["final_test_loss"] = test_loss
        wandb_run.finish()


if __name__ == "__main__":
    main()
