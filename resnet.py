from __future__ import annotations
import math
from typing import Literal
import torch
from torch import Tensor, nn
from torch.nn import functional as F


ActivationName = Literal["relu", "gelu", "silu", "tanh", "identity"] | None
NormalizationName = Literal["batch", "group", "identity"] | None
DownsampleMode = Literal["stride", "avgpool", "maxpool"]

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


def build_norm(
    name: NormalizationName,
    num_channels: int,
    *,
    num_groups: int = 8,
) -> nn.Module:
    """Create a normalization layer for 2D feature maps."""
    if name in (None, "identity"):
        return nn.Identity()
    if name == "batch":
        return nn.BatchNorm2d(num_channels)
    if name == "group":
        if num_groups < 1:
            raise ValueError(f"num_groups must be at least 1, got {num_groups}")
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported normalization: {name}")


def build_spatial_downsample2d(mode: DownsampleMode, stride: int) -> nn.Module:
    """Create a 2D spatial downsampling module."""
    if stride == 1:
        return nn.Identity()
    if mode == "stride":
        return nn.Identity()
    if mode == "avgpool":
        return nn.AvgPool2d(kernel_size=stride, stride=stride)
    if mode == "maxpool":
        return nn.MaxPool2d(kernel_size=stride, stride=stride)
    raise ValueError(f"Unsupported downsample mode: {mode}")


def build_spatial_downsample3d(mode: DownsampleMode, stride: int) -> nn.Module:
    """Create a 3D downsampling module over spatial dimensions only."""
    if stride == 1:
        return nn.Identity()
    if mode == "stride":
        return nn.Identity()
    kernel_size = (1, stride, stride)
    if mode == "avgpool":
        return nn.AvgPool3d(kernel_size=kernel_size, stride=kernel_size)
    if mode == "maxpool":
        return nn.MaxPool3d(kernel_size=kernel_size, stride=kernel_size)
    raise ValueError(f"Unsupported downsample mode: {mode}")


class SinusoidalTimeEmbedding(nn.Module):
    """Create diffusion-style sinusoidal embeddings for scalar timesteps."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be at least 1, got {embedding_dim}")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return timesteps.unsqueeze(-1)

        frequency_step = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            -frequency_step * torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
        )
        angles = timesteps.unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


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
        norm: NormalizationName = "batch",
        norm_groups: int = 8,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        if bias is None:
            bias = norm in ("identity", None)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = build_norm(norm, out_channels, num_groups=norm_groups)
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
        norm: NormalizationName = "batch",
        downsample_mode: DownsampleMode = "stride",
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        conv_stride = stride if downsample_mode == "stride" else 1
        self.downsample = build_spatial_downsample2d(downsample_mode, stride)
        self.block1 = ConvBlock(
            in_channels,
            out_channels,
            stride=conv_stride,
            norm=norm,
            activation=activation,
        )
        self.block2 = ConvBlock(
            out_channels,
            out_channels,
            norm=norm,
            activation=None,
        )
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False,
                ),
                build_norm(norm, out_channels),
            )
        self.activation = build_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
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
        downsample_mode: DownsampleMode = "stride",
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
        self.downsample_mode = downsample_mode

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
                norm="batch",
                downsample_mode=downsample_mode,
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
                    norm="batch",
                    downsample_mode=downsample_mode,
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
        norm: NormalizationName,
        downsample_mode: DownsampleMode,
        activation: ActivationName,
    ) -> nn.Sequential:
        blocks = [
            ResidualConvBlock(
                in_channels,
                out_channels,
                stride=stride,
                norm=norm,
                downsample_mode=downsample_mode,
                activation=activation,
            )
        ]
        for _ in range(n_blocks - 1):
            blocks.append(
                ResidualConvBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    norm=norm,
                    downsample_mode="stride",
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


class UNetResidualBlock(nn.Module):
    """A diffusion-style residual block with time-based scale-shift modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_embedding_dim: int | None = None,
        norm: NormalizationName = "group",
        norm_groups: int = 8,
        activation: ActivationName = "silu",
    ) -> None:
        super().__init__()
        self.norm1 = build_norm(norm, in_channels, num_groups=norm_groups)
        self.activation1 = build_activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = build_norm(norm, out_channels, num_groups=norm_groups)
        self.activation2 = build_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.time_projection = None
        if time_embedding_dim is not None:
            self.time_projection = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embedding_dim, out_channels * 2),
            )

    def forward(self, x: Tensor, time_embedding: Tensor | None = None) -> Tensor:
        residual = self.shortcut(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        if self.time_projection is not None:
            if time_embedding is None:
                raise ValueError("time_embedding is required when time conditioning is enabled")
            scale_shift = self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
            scale, shift = scale_shift.chunk(2, dim=1)
            x = x * (1 + scale) + shift
        x = self.activation2(x)
        x = self.conv2(x)
        return x + residual


class UNetDownBlock(nn.Module):
    """Downsample once, then apply a time-conditioned residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_embedding_dim: int | None = None,
        norm: NormalizationName = "group",
        norm_groups: int = 8,
        activation: ActivationName = "silu",
    ) -> None:
        super().__init__()
        self.downsample = ConvBlock(
            in_channels,
            out_channels,
            stride=2,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
        )
        self.refine = UNetResidualBlock(
            out_channels,
            out_channels,
            time_embedding_dim=time_embedding_dim,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
        )

    def forward(self, x: Tensor, time_embedding: Tensor | None = None) -> Tensor:
        x = self.downsample(x)
        x = self.refine(x, time_embedding)
        return x


class UNetUpBlock(nn.Module):
    """Upsample decoder features and fuse them with the matching skip tensor."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        time_embedding_dim: int | None = None,
        norm: NormalizationName = "group",
        norm_groups: int = 8,
        activation: ActivationName = "silu",
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        self.up_conv = ConvBlock(
            in_channels,
            out_channels,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
        )
        self.fuse = UNetResidualBlock(
            out_channels + skip_channels,
            out_channels,
            time_embedding_dim=time_embedding_dim,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
        )

    def forward(
        self,
        x: Tensor,
        skip: Tensor,
        time_embedding: Tensor | None = None,
    ) -> Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.up_conv(x)
        x = torch.cat([skip, x], dim=1)
        return self.fuse(x, time_embedding)


class UNet(nn.Module):
    """A basic UNet with diffusion-style time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_dims: list[int],
        *,
        bottleneck_channels: int | None = None,
        time_embedding_dim: int | None = None,
        norm: NormalizationName = "group",
        norm_groups: int = 8,
        activation: ActivationName = "silu",
    ) -> None:
        super().__init__()
        if not channel_dims:
            raise ValueError("channel_dims must contain at least one channel dimension")

        if bottleneck_channels is None:
            bottleneck_channels = channel_dims[-1] * 2
        if time_embedding_dim is None:
            time_embedding_dim = channel_dims[0] * 4

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_dims = channel_dims
        self.bottleneck_channels = bottleneck_channels
        self.time_embedding_dim = time_embedding_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.stem = UNetResidualBlock(
            in_channels,
            channel_dims[0],
            time_embedding_dim=time_embedding_dim,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
        )

        down_blocks: list[nn.Module] = []
        for prev_channels, next_channels in zip(channel_dims, channel_dims[1:]):
            down_blocks.append(
                UNetDownBlock(
                    prev_channels,
                    next_channels,
                    time_embedding_dim=time_embedding_dim,
                    norm=norm,
                    norm_groups=norm_groups,
                    activation=activation,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bottleneck = UNetResidualBlock(
            channel_dims[-1],
            bottleneck_channels,
            time_embedding_dim=time_embedding_dim,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
        )

        up_blocks: list[nn.Module] = []
        decoder_channels = bottleneck_channels
        for skip_channels in reversed(channel_dims[:-1]):
            up_blocks.append(
                UNetUpBlock(
                    decoder_channels,
                    skip_channels,
                    skip_channels,
                    time_embedding_dim=time_embedding_dim,
                    norm=norm,
                    norm_groups=norm_groups,
                    activation=activation,
                )
            )
            decoder_channels = skip_channels
        self.up_blocks = nn.ModuleList(up_blocks)

        self.head = nn.Conv2d(decoder_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        elif t.ndim != 1:
            raise ValueError(f"Expected t to have shape (batch,) or (batch, 1), got {tuple(t.shape)}")
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"Expected t batch size {x.shape[0]}, got {t.shape[0]}")

        time_embedding = self.time_embedding(t.to(device=x.device))
        time_embedding = time_embedding.to(dtype=self.time_mlp[0].weight.dtype)
        time_embedding = self.time_mlp(time_embedding).to(dtype=x.dtype)
        skips: list[Tensor] = []

        x = self.stem(x, time_embedding)
        skips.append(x)

        for down_block in self.down_blocks:
            x = down_block(x, time_embedding)
            skips.append(x)

        x = self.bottleneck(x, time_embedding)

        for up_block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = up_block(x, skip, time_embedding)

        x = self.head(x)
        return x


__all__ = [
    "ActivationName",
    "ConvBlock",
    "DownsampleMode",
    "NormalizationName",
    "ResidualConvBlock",
    "ResNet",
    "SinusoidalTimeEmbedding",
    "UNet",
    "UNetDownBlock",
    "UNetResidualBlock",
    "UNetUpBlock",
    "build_activation",
    "build_norm",
    "build_spatial_downsample2d",
    "build_spatial_downsample3d",
]
