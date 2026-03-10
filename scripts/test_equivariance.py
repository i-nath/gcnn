from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p4mresnet import P4MP4MConvBlock, P4MZ2ConvBlock, p4m_reflect_action
from p4resnet import P4P4ConvBlock, P4Z2ConvBlock


ATOL = 1e-8
RTOL = 1e-6


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)


def rotate_z2(x: Tensor, k: int) -> Tensor:
    return torch.rot90(x, k=k % 4, dims=(-2, -1))


def reflect_z2(x: Tensor) -> Tensor:
    return torch.flip(x, dims=(-1,))


def act_p4(x: Tensor, k: int) -> Tensor:
    k = k % 4
    x = torch.rot90(x, k=k, dims=(-2, -1))
    return torch.roll(x, shifts=k, dims=2)


def reflect_p4m_group_axis(x: Tensor) -> Tensor:
    top, bot = x.split(4, dim=2)
    top = p4m_reflect_action(top, dim=2)
    bot = p4m_reflect_action(bot, dim=2)
    return torch.cat([bot, top], dim=2)


def act_p4m_rotation(x: Tensor, k: int) -> Tensor:
    k = k % 4
    x = torch.rot90(x, k=k, dims=(-2, -1))
    top, bot = x.split(4, dim=2)
    top = torch.roll(top, shifts=k, dims=2)
    bot = torch.roll(bot, shifts=k, dims=2)
    return torch.cat([top, bot], dim=2)


def act_p4m_reflection(x: Tensor) -> Tensor:
    x = torch.flip(x, dims=(-1,))
    return reflect_p4m_group_axis(x)


def assert_equivariant(name: str, actual: Tensor, expected: Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    max_error = (actual - expected).abs().max().item()
    print(f"[PASS] {name} (max abs error: {max_error:.3e})")


def make_p4_z2_block() -> P4Z2ConvBlock:
    return P4Z2ConvBlock(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        use_batch_norm=False,
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4_p4_block() -> P4P4ConvBlock:
    return P4P4ConvBlock(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        use_batch_norm=False,
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4m_z2_block() -> P4MZ2ConvBlock:
    return P4MZ2ConvBlock(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        use_batch_norm=False,
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4m_p4m_block() -> P4MP4MConvBlock:
    return P4MP4MConvBlock(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        use_batch_norm=False,
        activation=None,
    ).eval().to(dtype=torch.float64)


def test_p4_z2_conv_block() -> None:
    block = make_p4_z2_block()
    x = torch.randn(2, 3, 15, 15, dtype=torch.float64)

    for k in range(4):
        lhs = block(rotate_z2(x, k))
        rhs = act_p4(block(x), k)
        assert_equivariant(f"P4Z2ConvBlock rotation k={k}", lhs, rhs)


def test_p4_p4_conv_block() -> None:
    block = make_p4_p4_block()
    x = torch.randn(2, 3, 4, 19, 19, dtype=torch.float64)

    for k in range(4):
        lhs = block(act_p4(x, k))
        rhs = act_p4(block(x), k)
        assert_equivariant(f"P4P4ConvBlock rotation k={k}", lhs, rhs)


def test_p4m_z2_conv_block() -> None:
    block = make_p4m_z2_block()
    x = torch.randn(2, 3, 15, 15, dtype=torch.float64)

    for k in range(4):
        lhs = block(rotate_z2(x, k))
        rhs = act_p4m_rotation(block(x), k)
        assert_equivariant(f"P4MZ2ConvBlock rotation k={k}", lhs, rhs)

    lhs = block(reflect_z2(x))
    rhs = act_p4m_reflection(block(x))
    assert_equivariant("P4MZ2ConvBlock reflection", lhs, rhs)


def test_p4m_p4m_conv_block() -> None:
    block = make_p4m_p4m_block()
    x = torch.randn(2, 3, 8, 19, 19, dtype=torch.float64)

    for k in range(4):
        lhs = block(act_p4m_rotation(x, k))
        rhs = act_p4m_rotation(block(x), k)
        assert_equivariant(f"P4MP4MConvBlock rotation k={k}", lhs, rhs)

    lhs = block(act_p4m_reflection(x))
    rhs = act_p4m_reflection(block(x))
    assert_equivariant("P4MP4MConvBlock reflection", lhs, rhs)


def main() -> None:
    set_seed(0)
    torch.set_grad_enabled(False)

    print("Running equivariance checks for P4 and P4M blocks...")
    test_p4_z2_conv_block()
    test_p4_p4_conv_block()
    test_p4m_z2_conv_block()
    test_p4m_p4m_conv_block()
    print("All equivariance checks passed.")


if __name__ == "__main__":
    main()
