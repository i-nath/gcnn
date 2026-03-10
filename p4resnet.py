from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from resnet import ActivationName, build_activation

class P4Z2ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool | None = False,
        use_batch_norm: bool = True,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels,)) if bias else None
        
        nn.init.kaiming_normal_(self.weight)

        self.stride = stride
        self.padding = padding
        if self.padding is None:
            self.padding = kernel_size // 2

        self.norm = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
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
        bias: bool | None = False,
        use_batch_norm: bool = True,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels,)) if bias else None
        
        nn.init.kaiming_normal_(self.weight)

        self.stride = stride
        self.padding = padding
        if self.padding is None:
            self.padding = kernel_size // 2

        self.norm = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
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
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        self.block1 = P4P4ConvBlock(
            in_channels,
            out_channels,
            stride=stride,
            activation=activation,
        )
        self.block2 = P4P4ConvBlock(
            out_channels,
            out_channels,
            activation=None,
        )
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels),
            )
        self.activation = build_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
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

        self.stem = P4Z2ConvBlock(
            in_channels,
            channel_dims[0],
            kernel_size=3,
            stride=1,
            activation=activation,
        )

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
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
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
            P4ResidualConvBlock(
                in_channels,
                out_channels,
                stride=stride,
                activation=activation,
            )
        ]
        for _ in range(n_blocks - 1):
            blocks.append(
                P4ResidualConvBlock(
                    out_channels,
                    out_channels,
                    stride=1,
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


__all__ = [
    "P4Z2ConvBlock",
    "P4P4ConvBlock",
    "P4ResidualConvBlock",
    "P4ResNet",
]
