from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p4mresnet import (
    P4MP4MConvBlock,
    P4MUNet,
    P4MUNetDownBlock,
    P4MUNetLiftBlock,
    P4MUNetResidualBlock,
    P4MUNetUpBlock,
    P4MZ2ConvBlock,
    p4m_reflect_action,
)
from p4resnet import (
    P4P4ConvBlock,
    P4UNet,
    P4UNetDownBlock,
    P4UNetLiftBlock,
    P4UNetResidualBlock,
    P4UNetUpBlock,
    P4Z2ConvBlock,
)


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
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4_p4_block() -> P4P4ConvBlock:
    return P4P4ConvBlock(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4m_z2_block() -> P4MZ2ConvBlock:
    return P4MZ2ConvBlock(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4m_p4m_block() -> P4MP4MConvBlock:
    return P4MP4MConvBlock(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4_unet_lift_block() -> P4UNetLiftBlock:
    return P4UNetLiftBlock(
        in_channels=3,
        out_channels=4,
        time_embedding_dim=16,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4_unet_residual_block() -> P4UNetResidualBlock:
    return P4UNetResidualBlock(
        in_channels=3,
        out_channels=4,
        time_embedding_dim=16,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4_unet_down_block() -> P4UNetDownBlock:
    return P4UNetDownBlock(
        in_channels=3,
        out_channels=4,
        time_embedding_dim=16,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4_unet_up_block() -> P4UNetUpBlock:
    return P4UNetUpBlock(
        in_channels=3,
        skip_channels=5,
        out_channels=4,
        time_embedding_dim=16,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4_unet() -> P4UNet:
    return P4UNet(
        in_channels=3,
        out_channels=2,
        channel_dims=[8, 16],
        time_embedding_dim=16,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4m_unet_lift_block() -> P4MUNetLiftBlock:
    return P4MUNetLiftBlock(
        in_channels=3,
        out_channels=4,
        time_embedding_dim=16,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4m_unet_residual_block() -> P4MUNetResidualBlock:
    return P4MUNetResidualBlock(
        in_channels=3,
        out_channels=4,
        time_embedding_dim=16,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4m_unet_down_block() -> P4MUNetDownBlock:
    return P4MUNetDownBlock(
        in_channels=3,
        out_channels=4,
        time_embedding_dim=16,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4m_unet_up_block() -> P4MUNetUpBlock:
    return P4MUNetUpBlock(
        in_channels=3,
        skip_channels=5,
        out_channels=4,
        time_embedding_dim=16,
        norm="identity",
        activation=None,
    ).eval().to(dtype=torch.float64)


def make_p4m_unet() -> P4MUNet:
    return P4MUNet(
        in_channels=3,
        out_channels=2,
        channel_dims=[8, 16],
        time_embedding_dim=16,
        norm="identity",
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


def test_p4_unet_lift_block() -> None:
    block = make_p4_unet_lift_block()
    x = torch.randn(2, 3, 15, 15, dtype=torch.float64)
    t = torch.randn(2, 16, dtype=torch.float64)

    for k in range(4):
        lhs = block(rotate_z2(x, k), t)
        rhs = act_p4(block(x, t), k)
        assert_equivariant(f"P4UNetLiftBlock rotation k={k}", lhs, rhs)


def test_p4_unet_residual_block() -> None:
    block = make_p4_unet_residual_block()
    x = torch.randn(2, 3, 4, 19, 19, dtype=torch.float64)
    t = torch.randn(2, 16, dtype=torch.float64)

    for k in range(4):
        lhs = block(act_p4(x, k), t)
        rhs = act_p4(block(x, t), k)
        assert_equivariant(f"P4UNetResidualBlock rotation k={k}", lhs, rhs)


def test_p4_unet_down_block() -> None:
    block = make_p4_unet_down_block()
    x = torch.randn(2, 3, 4, 19, 19, dtype=torch.float64)
    t = torch.randn(2, 16, dtype=torch.float64)

    for k in range(4):
        lhs = block(act_p4(x, k), t)
        rhs = act_p4(block(x, t), k)
        assert_equivariant(f"P4UNetDownBlock rotation k={k}", lhs, rhs)


def test_p4_unet_up_block() -> None:
    block = make_p4_unet_up_block()
    x = torch.randn(2, 3, 4, 9, 9, dtype=torch.float64)
    skip = torch.randn(2, 5, 4, 19, 19, dtype=torch.float64)
    t = torch.randn(2, 16, dtype=torch.float64)

    for k in range(4):
        lhs = block(act_p4(x, k), act_p4(skip, k), t)
        rhs = act_p4(block(x, skip, t), k)
        assert_equivariant(f"P4UNetUpBlock rotation k={k}", lhs, rhs)


def test_p4_unet() -> None:
    model = make_p4_unet()
    x = torch.randn(2, 3, 31, 31, dtype=torch.float64)
    t = torch.rand(2, dtype=torch.float64)

    for k in range(4):
        lhs = model(rotate_z2(x, k), t)
        rhs = rotate_z2(model(x, t), k)
        assert_equivariant(f"P4UNet rotation k={k}", lhs, rhs)


def test_p4m_unet_lift_block() -> None:
    block = make_p4m_unet_lift_block()
    x = torch.randn(2, 3, 15, 15, dtype=torch.float64)
    t = torch.randn(2, 16, dtype=torch.float64)

    for k in range(4):
        lhs = block(rotate_z2(x, k), t)
        rhs = act_p4m_rotation(block(x, t), k)
        assert_equivariant(f"P4MUNetLiftBlock rotation k={k}", lhs, rhs)

    lhs = block(reflect_z2(x), t)
    rhs = act_p4m_reflection(block(x, t))
    assert_equivariant("P4MUNetLiftBlock reflection", lhs, rhs)


def test_p4m_unet_residual_block() -> None:
    block = make_p4m_unet_residual_block()
    x = torch.randn(2, 3, 8, 19, 19, dtype=torch.float64)
    t = torch.randn(2, 16, dtype=torch.float64)

    for k in range(4):
        lhs = block(act_p4m_rotation(x, k), t)
        rhs = act_p4m_rotation(block(x, t), k)
        assert_equivariant(f"P4MUNetResidualBlock rotation k={k}", lhs, rhs)

    lhs = block(act_p4m_reflection(x), t)
    rhs = act_p4m_reflection(block(x, t))
    assert_equivariant("P4MUNetResidualBlock reflection", lhs, rhs)


def test_p4m_unet_down_block() -> None:
    block = make_p4m_unet_down_block()
    x = torch.randn(2, 3, 8, 19, 19, dtype=torch.float64)
    t = torch.randn(2, 16, dtype=torch.float64)

    for k in range(4):
        lhs = block(act_p4m_rotation(x, k), t)
        rhs = act_p4m_rotation(block(x, t), k)
        assert_equivariant(f"P4MUNetDownBlock rotation k={k}", lhs, rhs)

    lhs = block(act_p4m_reflection(x), t)
    rhs = act_p4m_reflection(block(x, t))
    assert_equivariant("P4MUNetDownBlock reflection", lhs, rhs)


def test_p4m_unet_up_block() -> None:
    block = make_p4m_unet_up_block()
    x = torch.randn(2, 3, 8, 9, 9, dtype=torch.float64)
    skip = torch.randn(2, 5, 8, 19, 19, dtype=torch.float64)
    t = torch.randn(2, 16, dtype=torch.float64)

    for k in range(4):
        lhs = block(act_p4m_rotation(x, k), act_p4m_rotation(skip, k), t)
        rhs = act_p4m_rotation(block(x, skip, t), k)
        assert_equivariant(f"P4MUNetUpBlock rotation k={k}", lhs, rhs)

    lhs = block(act_p4m_reflection(x), act_p4m_reflection(skip), t)
    rhs = act_p4m_reflection(block(x, skip, t))
    assert_equivariant("P4MUNetUpBlock reflection", lhs, rhs)


def test_p4m_unet() -> None:
    model = make_p4m_unet()
    x = torch.randn(2, 3, 31, 31, dtype=torch.float64)
    t = torch.rand(2, dtype=torch.float64)

    for k in range(4):
        lhs = model(rotate_z2(x, k), t)
        rhs = rotate_z2(model(x, t), k)
        assert_equivariant(f"P4MUNet rotation k={k}", lhs, rhs)

    lhs = model(reflect_z2(x), t)
    rhs = reflect_z2(model(x, t))
    assert_equivariant("P4MUNet reflection", lhs, rhs)


def main() -> None:
    set_seed(0)
    torch.set_grad_enabled(False)

    print("Running equivariance checks for P4 and P4M blocks...")
    test_p4_z2_conv_block()
    test_p4_p4_conv_block()
    test_p4m_z2_conv_block()
    test_p4m_p4m_conv_block()
    test_p4_unet_lift_block()
    test_p4_unet_residual_block()
    test_p4_unet_down_block()
    test_p4_unet_up_block()
    test_p4_unet()
    test_p4m_unet_lift_block()
    test_p4m_unet_residual_block()
    test_p4m_unet_down_block()
    test_p4m_unet_up_block()
    test_p4m_unet()
    print("All equivariance checks passed.")


if __name__ == "__main__":
    main()
