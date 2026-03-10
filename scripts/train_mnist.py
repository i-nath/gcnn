from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloaders import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    create_rotated_mnist_dataloaders,
)
from p4mresnet import P4MResNet
from p4resnet import P4ResNet
from resnet import ResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on rotated MNIST.")
    parser.add_argument(
        "--model",
        choices=("resnet", "p4resnet", "p4mresnet"),
        default="resnet",
        help="Model family to train.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the rotated-MNIST training `.amat` file.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Path to the rotated-MNIST test `.amat` file.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=2,
        help="Residual blocks per stage.",
    )
    parser.add_argument(
        "--channel-dims",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Stage widths for the selected network.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. `cpu`, `cuda`, or `cuda:0`.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
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
        "--save-checkpoint",
        type=Path,
        default=None,
        help="Optional path for the best validation checkpoint.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
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
        "n_blocks": args.n_blocks,
        "in_channels": 1,
        "num_classes": 10,
        "channel_dims": args.channel_dims,
    }
    if args.model == "resnet":
        return ResNet(**common_kwargs)
    if args.model == "p4resnet":
        return P4ResNet(**common_kwargs)
    if args.model == "p4mresnet":
        return P4MResNet(**common_kwargs)
    raise ValueError(f"Unsupported model: {args.model}")


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    return create_rotated_mnist_dataloaders(
        train_path=args.train_path,
        test_path=args.test_path,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    max_batches: int | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, (images, labels) in enumerate(progress, start=1):
        if max_batches is not None and step > max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())

        avg_loss = total_loss / total_examples
        avg_acc = total_correct / total_examples
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"train_loss={avg_loss:.4f} train_acc={avg_acc:.4f}"
            )

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for step, (images, labels) in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
        if max_batches is not None and step > max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())

    return total_loss / total_examples, total_correct / total_examples


def maybe_save_checkpoint(
    path: Path | None,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    train_loader, test_loader = create_dataloaders(args)
    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Using device: {device}")
    print(f"Training model: {args.model}")
    print(f"Train examples: {len(train_loader.dataset)}")
    print(f"Test examples: {len(test_loader.dataset)}")

    best_test_acc = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            max_batches=args.max_train_batches,
        )
        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_eval_batches,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            maybe_save_checkpoint(
                path=args.save_checkpoint,
                model=model,
                args=args,
                epoch=epoch,
                metrics={
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                },
            )


if __name__ == "__main__":
    main()
