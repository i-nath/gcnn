from __future__ import annotations
from typing import Literal
from torch import Tensor, nn


ActivationName = Literal["relu", "gelu", "silu", "tanh", "identity"] | None

def build_activation(name: ActivationName) -> nn.Module:
    """Create a small activation module from a string name."""
    if name in (None, "identity"):
        return nn.Identity()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    """Basic convolution -> normalization -> activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool | None = None,
        use_batch_norm: bool = True,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        if bias is None:
            bias = not use_batch_norm

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = build_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)  # (bs, out_c, out_h, out_w)
        x = self.norm(x)  # (bs, out_c, out_h, out_w)
        return self.activation(x)


class ResidualConvBlock(nn.Module):
    """A simple residual block for ordinary CNN feature maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        self.block1 = ConvBlock(
            in_channels,
            out_channels,
            stride=stride,
            activation=activation,
        )
        self.block2 = ConvBlock(
            out_channels,
            out_channels,
            activation=None,
        )
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = build_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)  # (bs, out_c, out_h, out_w)
        x = self.block1(x)  # (bs, out_c, out_h, out_w)
        x = self.block2(x)  # (bs, out_c, out_h, out_w)
        return self.activation(x + residual)


class ResNet(nn.Module):
    """A configurable ResNet with stage-wise residual blocks."""

    def __init__(
        self,
        n_blocks: int,
        in_channels: int,
        num_classes: int,
        channel_dims: list[int],
        *,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be at least 1, got {n_blocks}")
        if not channel_dims:
            raise ValueError("channel_dims must contain at least one channel dimension")

        self.n_blocks = n_blocks
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channel_dims = channel_dims

        self.stem = nn.Conv2d(
            in_channels,
            channel_dims[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.stem_norm = nn.BatchNorm2d(channel_dims[0])
        self.stem_activation = build_activation(activation)

        stages: list[nn.Module] = []
        stages.append(
            self._make_stage(
                in_channels=channel_dims[0],
                out_channels=channel_dims[0],
                stride=1,
                n_blocks=n_blocks,
                activation=activation,
            )
        )

        for prev_channels, next_channels in zip(channel_dims, channel_dims[1:]):
            stages.append(
                self._make_stage(
                    in_channels=prev_channels,
                    out_channels=next_channels,
                    stride=2,
                    n_blocks=n_blocks,
                    activation=activation,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channel_dims[-1], num_classes)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        n_blocks: int,
        activation: ActivationName,
    ) -> nn.Sequential:
        blocks = [
            ResidualConvBlock(
                in_channels,
                out_channels,
                stride=stride,
                activation=activation,
            )
        ]
        for _ in range(n_blocks - 1):
            blocks.append(
                ResidualConvBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    activation=activation,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)  # (bs, channel_dims[0], h, w)
        x = self.stem_norm(x)  # (bs, channel_dims[0], h, w)
        x = self.stem_activation(x)  # (bs, channel_dims[0], h, w)

        x = self.stages[0](x)  # (bs, channel_dims[0], h, w)

        for stage in self.stages[1:]:
            x = stage(x)  # (bs, next_c, h / 2^k, w / 2^k)

        x = self.pool(x)  # (bs, channel_dims[-1], 1, 1)

        x = x.flatten(start_dim=1)  # (bs, channel_dims[-1])

        x = self.classifier(x)  # (bs, num_classes)
        return x


__all__ = [
    "ActivationName",
    "ConvBlock",
    "ResidualConvBlock",
    "ResNet",
    "build_activation",
]
