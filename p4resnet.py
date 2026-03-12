from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from resnet import (
    ActivationName,
    DownsampleMode,
    NormalizationName,
    SinusoidalTimeEmbedding,
    build_activation,
    build_norm,
    build_spatial_downsample3d,
)


def build_norm3d(
    name: NormalizationName,
    num_channels: int,
    *,
    num_groups: int = 8,
) -> nn.Module:
    """Create a normalization layer for group feature maps."""
    if name in (None, "identity"):
        return nn.Identity()
    if name == "batch":
        return nn.BatchNorm3d(num_channels)
    if name == "group":
        if num_groups < 1:
            raise ValueError(f"num_groups must be at least 1, got {num_groups}")
        groups = min(num_groups, num_channels)
        while num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported normalization: {name}")


def upsample_p4_feature_map(x: Tensor, size: tuple[int, int]) -> Tensor:
    """Upsample only the spatial dimensions of a P4 feature map."""
    batch_size, channels, orientations, height, width = x.shape
    x = x.reshape(batch_size, channels * orientations, height, width)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    new_height, new_width = size
    return x.reshape(batch_size, channels, orientations, new_height, new_width)

class P4Z2ConvBlock(nn.Module):

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
        if bias is None:
            bias = norm in ("identity", None)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels,)) if bias else None
        
        nn.init.kaiming_normal_(self.weight)

        self.stride = stride
        self.padding = padding
        if self.padding is None:
            self.padding = kernel_size // 2

        self.norm = build_norm3d(norm, out_channels, num_groups=norm_groups)
        self.activation = build_activation(activation)

    def forward(self, x: Tensor) -> Tensor: # (bs, c_in, h, w)

        w_rot = torch.stack(
            [torch.rot90(self.weight, k=k, dims=(-2, -1)) for k in range(4)],
            dim=1,
        ) # (c_out, 4, c_in, k, k)

        c_out, _, c_in, kh, kw = w_rot.shape
        w_flat = w_rot.reshape(c_out * 4, c_in, kh, kw)

        x = F.conv2d(x, w_flat, bias=None, padding=self.padding, stride=self.stride) # (bs, c_out * 4, h, w)

        bs, _, h, w = x.shape
        x = x.reshape(bs, c_out, 4, h, w)

        if self.bias is not None:
            x = x + self.bias.view(1, c_out, 1, 1, 1)

        x = self.norm(x)  # (bs, c_out, 4, h, w)
        return self.activation(x)

class P4P4ConvBlock(nn.Module):

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
        if bias is None:
            bias = norm in ("identity", None)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels,)) if bias else None
        
        nn.init.kaiming_normal_(self.weight)

        self.stride = stride
        self.padding = padding
        if self.padding is None:
            self.padding = kernel_size // 2

        self.norm = build_norm3d(norm, out_channels, num_groups=norm_groups)
        self.activation = build_activation(activation)

    def forward(self, x: Tensor) -> Tensor: # (bs, c_in, 4, h, w)

        w_rot = torch.stack(
            [torch.roll(
                torch.rot90(self.weight, k=k, dims=(-2, -1)),
                shifts=k,
                dims=2
                ) for k in range(4)],
            dim=1,
        ) # (c_out, 4, c_in, 4, k, k)

        c_out, _, c_in, _, kh, kw = w_rot.shape
        w_flat = w_rot.reshape(c_out * 4, c_in * 4, kh, kw)

        bs, c_in, _, h, w = x.shape
        x_flat = x.reshape(bs, c_in * 4, h, w)

        x = F.conv2d(x_flat, w_flat, bias=None, padding=self.padding, stride=self.stride) # (bs, c_out * 4, h, w)
        
        bs, _, h, w = x.shape # h, w may change due to stride
        x = x.reshape(bs, c_out, 4, h, w)

        if self.bias is not None:
            x = x + self.bias.view(1, c_out, 1, 1, 1)

        x = self.norm(x)  # (bs, c_out, 4, h, w)
        return self.activation(x)

class P4ResidualConvBlock(nn.Module):
    """A simple residual block for ordinary CNN feature maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        norm: NormalizationName = "batch",
        norm_groups: int = 8,
        downsample_mode: DownsampleMode = "stride",
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        conv_stride = stride if downsample_mode == "stride" else 1
        self.downsample = build_spatial_downsample3d(downsample_mode, stride)
        self.block1 = P4P4ConvBlock(
            in_channels,
            out_channels,
            stride=conv_stride,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
        )
        self.block2 = P4P4ConvBlock(
            out_channels,
            out_channels,
            norm=norm,
            norm_groups=norm_groups,
            activation=None,
        )
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, conv_stride, conv_stride),
                    bias=False,
                ),
                build_norm3d(norm, out_channels, num_groups=norm_groups),
            )
        self.activation = build_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        residual = self.shortcut(x)  # (bs, out_c, out_h, out_w)
        x = self.block1(x)  # (bs, out_c, out_h, out_w)
        x = self.block2(x)  # (bs, out_c, out_h, out_w)
        return self.activation(x + residual)


class P4ResNet(nn.Module):
    """A configurable ResNet with stage-wise residual blocks."""

    def __init__(
        self,
        n_blocks: int,
        in_channels: int,
        num_classes: int,
        channel_dims: list[int],
        *,
        norm: NormalizationName = "batch",
        norm_groups: int = 8,
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

        self.stem = P4Z2ConvBlock(
            in_channels,
            channel_dims[0],
            kernel_size=3,
            stride=1,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
        )

        stages: list[nn.Module] = []
        stages.append(
            self._make_stage(
                in_channels=channel_dims[0],
                out_channels=channel_dims[0],
                stride=1,
                n_blocks=n_blocks,
                norm=norm,
                norm_groups=norm_groups,
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
                    norm=norm,
                    norm_groups=norm_groups,
                    downsample_mode=downsample_mode,
                    activation=activation,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(channel_dims[-1], num_classes)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        n_blocks: int,
        norm: NormalizationName,
        norm_groups: int,
        downsample_mode: DownsampleMode,
        activation: ActivationName,
    ) -> nn.Sequential:
        blocks = [
            P4ResidualConvBlock(
                in_channels,
                out_channels,
                stride=stride,
                norm=norm,
                norm_groups=norm_groups,
                downsample_mode=downsample_mode,
                activation=activation,
            )
        ]
        for _ in range(n_blocks - 1):
            blocks.append(
                P4ResidualConvBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    norm=norm,
                    norm_groups=norm_groups,
                    downsample_mode="stride",
                    activation=activation,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)  # (bs, channel_dims[0], 4, h, w)

        x = self.stages[0](x)  # (bs, channel_dims[0], 4, h, w)

        for stage in self.stages[1:]:
            x = stage(x)  # (bs, next_c, 4,h / 2^k, w / 2^k)

        x = self.pool(x)  # (bs, channel_dims[-1], 1, 1, 1)

        x = x.flatten(start_dim=1)  # (bs, channel_dims[-1])

        x = self.classifier(x)  # (bs, num_classes)
        return x


class P4UNetLiftBlock(nn.Module):
    """Lift a Z2 image into P4 features with a residual, time-conditioned block."""

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
        self.conv1 = P4Z2ConvBlock(
            in_channels,
            out_channels,
            norm="identity",
            activation=None,
        )
        self.norm2 = build_norm3d(norm, out_channels, num_groups=norm_groups)
        self.activation2 = build_activation(activation)
        self.conv2 = P4P4ConvBlock(
            out_channels,
            out_channels,
            norm="identity",
            activation=None,
        )
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        self.shortcut = P4Z2ConvBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            norm="identity",
            activation=None,
        )
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
            scale_shift = self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            scale, shift = scale_shift.chunk(2, dim=1)
            x = x * (1 + scale) + shift
        x = self.activation2(x)
        x = self.conv2(x)
        return x + residual


class P4UNetResidualBlock(nn.Module):
    """A P4 residual block with diffusion-style time conditioning."""

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
        self.norm1 = build_norm3d(norm, in_channels, num_groups=norm_groups)
        self.activation1 = build_activation(activation)
        self.conv1 = P4P4ConvBlock(
            in_channels,
            out_channels,
            norm="identity",
            activation=None,
        )
        self.norm2 = build_norm3d(norm, out_channels, num_groups=norm_groups)
        self.activation2 = build_activation(activation)
        self.conv2 = P4P4ConvBlock(
            out_channels,
            out_channels,
            norm="identity",
            activation=None,
        )
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
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
            scale_shift = self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            scale, shift = scale_shift.chunk(2, dim=1)
            x = x * (1 + scale) + shift
        x = self.activation2(x)
        x = self.conv2(x)
        return x + residual


class P4UNetDownBlock(nn.Module):
    """Downsample spatially and refine with a residual P4 block."""

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
        self.downsample = P4P4ConvBlock(
            in_channels,
            out_channels,
            stride=2,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
        )
        self.refine = P4UNetResidualBlock(
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


class P4UNetUpBlock(nn.Module):
    """Upsample spatially, fuse the skip tensor, and refine with a residual block."""

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
        self.up_conv = P4P4ConvBlock(
            in_channels,
            out_channels,
            norm=norm,
            norm_groups=norm_groups,
            activation=activation,
        )
        self.fuse = P4UNetResidualBlock(
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
        x = upsample_p4_feature_map(x, size=skip.shape[-2:])
        x = self.up_conv(x)
        x = torch.cat([skip, x], dim=1)
        return self.fuse(x, time_embedding)


class P4UNet(nn.Module):
    """A P4-equivariant U-Net with time conditioning and Z2 output projection."""

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

        self.stem = P4UNetLiftBlock(
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
                P4UNetDownBlock(
                    prev_channels,
                    next_channels,
                    time_embedding_dim=time_embedding_dim,
                    norm=norm,
                    norm_groups=norm_groups,
                    activation=activation,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        self.bottleneck = P4UNetResidualBlock(
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
                P4UNetUpBlock(
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

        self.head = P4P4ConvBlock(
            decoder_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            norm="identity",
            activation=None,
        )
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
        x = x.mean(dim=2)
        return x


__all__ = [
    "P4UNet",
    "P4UNetDownBlock",
    "P4UNetLiftBlock",
    "P4UNetResidualBlock",
    "P4UNetUpBlock",
    "P4Z2ConvBlock",
    "P4P4ConvBlock",
    "P4ResidualConvBlock",
    "P4ResNet",
    "build_norm3d",
]
