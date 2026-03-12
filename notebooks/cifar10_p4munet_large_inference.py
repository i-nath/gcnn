# %% [markdown]
# # CIFAR-10 `p4munet` Large Inference
#
# This notebook-style script loads the trained
# `cifar10_p4munet_large_last.pt` checkpoint and generates CIFAR-10 samples
# with the same flow-matching sampling loop used in training.
#
# The checkpoint includes both the training args and an EMA copy of the model
# weights. By default, this script uses the EMA weights for inference.

# %%
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


def find_project_root() -> Path:
    candidates = [Path.cwd().resolve(), Path.cwd().resolve().parent]
    for candidate in candidates:
        if (candidate / "scripts" / "train_flow_matching.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from the current working directory.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_flow_matching import (  # noqa: E402
    DATASET_SPECS,
    build_model,
    denormalize_cifar_images,
    generate_samples,
    resolve_amp_dtype,
    resolve_device,
    save_image_grid,
    set_seed,
)


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cifar10_flow_matching" / "cifar10_p4munet_large_last.pt"
USE_EMA_WEIGHTS = True
DEVICE = None  # Set to "cpu", "cuda", or "cuda:0" to override auto-selection.
SEED = 0
NUM_SAMPLES = 16
NUM_SAMPLE_STEPS = 30
SAMPLE_BATCH_SIZE = 8
OUTPUT_PATH = PROJECT_ROOT / "samples" / "flow_matching" / "cifar10_p4munet_large" / "notebook_inference.png"

assert CHECKPOINT_PATH.exists(), f"Checkpoint not found: {CHECKPOINT_PATH}"
print(f"Project root: {PROJECT_ROOT}")
print(f"Checkpoint: {CHECKPOINT_PATH}")

# %%
set_seed(SEED)

device = resolve_device(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
args = SimpleNamespace(**checkpoint["args"])
model = build_model(args).to(device)

state_dict_name = (
    "ema_model_state_dict"
    if USE_EMA_WEIGHTS and checkpoint.get("ema_model_state_dict") is not None
    else "model_state_dict"
)
model.load_state_dict(checkpoint[state_dict_name])
model.eval()

amp_dtype = resolve_amp_dtype(args, device) if getattr(args, "amp", False) else None

print(f"Using device: {device}")
print(f"Loaded weights: {state_dict_name}")
print(f"Model family: {args.model}")
print(f"Model size: {getattr(args, 'model_size', 'unknown')}")
print(f"Channel dims: {args.channel_dims}")
print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Best eval loss: {checkpoint['best_eval_loss']:.6f}")

# %%
samples = generate_samples(
    model,
    num_samples=NUM_SAMPLES,
    image_size=DATASET_SPECS["image_size"],
    in_channels=DATASET_SPECS["in_channels"],
    device=device,
    amp_dtype=amp_dtype,
    num_steps=NUM_SAMPLE_STEPS,
    batch_size=SAMPLE_BATCH_SIZE,
)

print(f"Generated samples tensor: {tuple(samples.shape)}")
samples

# %%
num_images = samples.shape[0]
grid_cols = int(num_images**0.5)
while num_images % grid_cols != 0:
    grid_cols -= 1
grid_rows = (num_images + grid_cols - 1) // grid_cols

fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

for ax, image in zip(axes, samples):
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.axis("off")

for ax in axes[num_images:]:
    ax.axis("off")

fig.suptitle("CIFAR-10 P4MUNet Large Samples", fontsize=14)
fig.tight_layout()

# %%
save_image_grid(samples, OUTPUT_PATH)
print(f"Saved sample grid to: {OUTPUT_PATH}")

# %% [markdown]
# ## Equivariant Generation From Rotated Noise
#
# This section starts from one base Gaussian noise tensor on the model grid,
# extends it with additional Gaussian support outside the original crop, and
# upsamples it with nearest-neighbor replication so each original pixel keeps
# the same local statistics on a finer lattice.
#
# Rotations are then applied on that finer lattice before center-cropping and
# averaging back down to the model resolution, which gives smoother
# interpolation between angles such as `30` and `60` degrees.

# %%
@torch.no_grad()
def generate_from_initial_noise(
    model,
    initial_noise: torch.Tensor,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    num_steps: int,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")

    model_was_training = model.training
    model.eval()

    x = initial_noise.to(device)
    dt = 1.0 / num_steps
    time_values = torch.linspace(0.0, 1.0 - dt, steps=num_steps, device=device, dtype=x.dtype)

    for t_value in time_values:
        t = torch.full((x.shape[0],), t_value, device=device, dtype=x.dtype)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            velocity = model(x, t)
        x = x + dt * velocity

    if model_was_training:
        model.train()

    return denormalize_cifar_images(x).cpu()


def normalize_for_display(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().cpu()
    image_min = images.amin(dim=(1, 2, 3), keepdim=True)
    image_max = images.amax(dim=(1, 2, 3), keepdim=True)
    return (images - image_min) / (image_max - image_min).clamp_min(1e-6)


def plot_rotation_results(
    generated_images: torch.Tensor,
    labels: list[str],
    *,
    noise_images: torch.Tensor | None = None,
    show_noises: bool = True,
    title: str,
    max_columns: int = 3,
) -> None:
    num_images = len(labels)
    num_columns = min(num_images, max_columns)
    num_angle_rows = math.ceil(num_images / num_columns)
    rows_per_group = 2 if show_noises and noise_images is not None else 1

    fig, axes = plt.subplots(
        num_angle_rows * rows_per_group,
        num_columns,
        figsize=(num_columns * 3, num_angle_rows * (5 if rows_per_group == 2 else 2.75)),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for idx, label in enumerate(labels):
        row_block = idx // num_columns
        col = idx % num_columns

        output_ax = axes[row_block * rows_per_group + rows_per_group - 1, col]
        output_image = generated_images[idx].permute(1, 2, 0).numpy()
        output_ax.imshow(output_image)
        output_ax.set_title(f"{label}")
        output_ax.axis("off")

        if rows_per_group == 2:
            noise_ax = axes[row_block * rows_per_group, col]
            noise_image = noise_images[idx].permute(1, 2, 0).numpy()
            noise_ax.imshow(noise_image)
            noise_ax.set_title(f"Noise {label}")
            noise_ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()


def rotate_tensor(
    images: torch.Tensor,
    angle_degrees: float,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    normalized_angle = angle_degrees % 360.0
    quarter_turns = round(normalized_angle / 90.0)
    if abs(normalized_angle - 90.0 * quarter_turns) < 1e-8:
        return torch.rot90(images, k=quarter_turns % 4, dims=(-2, -1))

    angle_radians = torch.deg2rad(torch.tensor(angle_degrees, dtype=torch.float32))
    cos_theta = torch.cos(angle_radians).item()
    sin_theta = torch.sin(angle_radians).item()

    theta = torch.tensor(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
        ],
        dtype=images.dtype,
        device=images.device,
    ).unsqueeze(0).repeat(images.shape[0], 1, 1)

    grid = torch.nn.functional.affine_grid(
        theta,
        size=images.shape,
        align_corners=False,
    )
    return torch.nn.functional.grid_sample(
        images,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )


def center_crop(images: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, _, height, width = images.shape
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return images[:, :, top : top + crop_size, left : left + crop_size]


def extend_noise_field(base_noise: torch.Tensor, margin_pixels: int) -> torch.Tensor:
    if margin_pixels < 0:
        raise ValueError(f"margin_pixels must be non-negative, got {margin_pixels}")

    if margin_pixels == 0:
        return base_noise

    batch_size, channels, height, width = base_noise.shape
    extended = torch.randn(
        batch_size,
        channels,
        height + 2 * margin_pixels,
        width + 2 * margin_pixels,
        dtype=base_noise.dtype,
        device=base_noise.device,
    )
    extended[
        :,
        :,
        margin_pixels : margin_pixels + height,
        margin_pixels : margin_pixels + width,
    ] = base_noise
    return extended


def downsample_mean(images: torch.Tensor, factor: int) -> torch.Tensor:
    if factor < 1:
        raise ValueError(f"factor must be at least 1, got {factor}")
    if factor == 1:
        return images
    return torch.nn.functional.avg_pool2d(images, kernel_size=factor, stride=factor)


def make_fine_noise_with_exact_avgpool(
    coarse_noise: torch.Tensor,
    factor: int,
    *,
    detail_scale: float = 1.0,
) -> torch.Tensor:
    if factor < 1:
        raise ValueError(f"factor must be at least 1, got {factor}")
    if factor == 1:
        return coarse_noise

    batch_size, channels, height, width = coarse_noise.shape
    repeated = coarse_noise.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)

    detail = torch.randn_like(repeated)
    detail = detail.view(batch_size, channels, height, factor, width, factor)
    detail = detail - detail.mean(dim=(3, 5), keepdim=True)
    detail = detail.reshape_as(repeated)

    return repeated + detail_scale * detail

# %%

# Test for make_fine_noise_with_exact_avgpool
def _test_make_fine_noise_with_exact_avgpool():
    # Simple deterministic input for a single batch, single channel, 2x2
    torch.manual_seed(0)
    base = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    factor = 4

    out = make_fine_noise_with_exact_avgpool(base, factor, detail_scale=0.0)
    # Should tile each base value into a 4x4 block
    expected = base.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    assert torch.allclose(out, expected), "When detail_scale=0, fine noise should be repeated tiling"

    # Check shape
    assert out.shape == (1, 1, 8, 8)
    # Downsampling by avg pool should approximately recover the base, up to numerical error
    pooled = downsample_mean(out, factor)
    assert pooled.shape == base.shape
    # With detail_scale=0, this should be exact
    assert torch.allclose(pooled, base, atol=1e-6)

    # With detail_scale > 0, should remain unbiased on average
    torch.manual_seed(42)
    base2 = torch.randn(2, 3, 4, 4)
    out2 = make_fine_noise_with_exact_avgpool(base2, factor=2, detail_scale=1.0)
    pooled2 = downsample_mean(out2, 2)
    assert pooled2.shape == base2.shape
    err = (pooled2 - base2).abs().max().item()
    assert err < 1e-5, f"Avgpool error too high: {err}"

_test_make_fine_noise_with_exact_avgpool()
print("make_fine_noise_with_exact_avgpool tests passed.")

# Plot a grid of example outputs from make_fine_noise_with_exact_avgpool at various detail_scale values and upsample factors.

import matplotlib.pyplot as plt

def plot_make_fine_noise_examples():
    base = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    factors = [2, 4, 8]
    detail_scales = [0.0, 0.5, 1.0]
    num_rows = len(factors)
    num_cols = len(detail_scales)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 4, num_rows * 4),
        squeeze=False,
    )

    for row, factor in enumerate(factors):
        for col, detail_scale in enumerate(detail_scales):
            torch.manual_seed(42)
            fine_noise = make_fine_noise_with_exact_avgpool(base, factor, detail_scale=detail_scale)[0, 0].cpu().numpy()
            ax = axes[row, col]
            im = ax.imshow(fine_noise, cmap="viridis")
            ax.set_title(f"factor={factor}, detail_scale={detail_scale}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")

    plt.suptitle("make_fine_noise_with_exact_avgpool Examples")
    plt.tight_layout()
    plt.show()

plot_make_fine_noise_examples()


# %%

rotation_angles = [0.0, 18.0, 36.0, 54.0, 72.0, 90.0]
rotation_labels = [f"{int(angle)}°" for angle in rotation_angles]
# %%
image_size = DATASET_SPECS["image_size"]
noise_upsample_factor = 8
fine_noise_detail_scale = 1.0
rotation_margin_pixels = math.ceil((math.sqrt(2) - 1.0) * image_size / 2.0) + 1

base_noise = torch.randn(
    1,
    DATASET_SPECS["in_channels"],
    image_size,
    image_size,
)
extended_noise = extend_noise_field(base_noise, rotation_margin_pixels)
fine_noise = make_fine_noise_with_exact_avgpool(
    extended_noise,
    noise_upsample_factor,
    detail_scale=fine_noise_detail_scale,
)
fine_noise_avgpool_error = (
    downsample_mean(fine_noise, noise_upsample_factor) - extended_noise
).abs().max().item()

rotated_noise_batch = torch.cat(
    [
        downsample_mean(
            center_crop(
                rotate_tensor(fine_noise, angle),
                image_size * noise_upsample_factor,
            ),
            noise_upsample_factor,
        )
        for angle in rotation_angles
    ],
    dim=0,
)

rotated_generations = generate_from_initial_noise(
    model,
    rotated_noise_batch,
    device=device,
    amp_dtype=amp_dtype,
    num_steps=NUM_SAMPLE_STEPS,
)
display_noises = normalize_for_display(rotated_noise_batch)

print("Generated samples from a coarse Gaussian noise field rotated on a finer grid.")
print(f"Noise upsample factor: {noise_upsample_factor}")
print(f"Fine-noise detail scale: {fine_noise_detail_scale}")
print(f"Extended coarse field size: {tuple(extended_noise.shape[-2:])}")
print(f"Fine rotation field size: {tuple(fine_noise.shape[-2:])}")
print(f"Max avgpool reconstruction error: {fine_noise_avgpool_error:.3e}")
print(f"Number of rotated inputs: {len(rotation_angles)}")

plot_rotation_results(
    rotated_generations,
    rotation_labels,
    noise_images=display_noises,
    show_noises=True,
    title="Generation from smoothly rotated initial noise",
)

# %%

plot_rotation_results(
    rotated_generations,
    rotation_labels,
    show_noises=False,
    # title="Generated samples from smoothly rotated initial noise",
    title = ""
)

# %%


def _rot90_perm(N: int) -> np.ndarray:
    """Flat permutation for 90° counterclockwise rotation on an N x N grid."""
    perm = np.empty(N * N, dtype=int)
    for i in range(N):
        for j in range(N):
            i2, j2 = N - 1 - j, i
            perm[i * N + j] = i2 * N + j2
    return perm


def _find_rot90_cycles(perm: np.ndarray):
    """Decompose permutation into cycles."""
    n = len(perm)
    seen = np.zeros(n, dtype=bool)
    cycles = []
    fixed = []
    for s in range(n):
        if seen[s]:
            continue
        cyc = []
        t = s
        while not seen[t]:
            seen[t] = True
            cyc.append(t)
            t = perm[t]
        if len(cyc) == 1:
            fixed.append(cyc[0])
        elif len(cyc) == 4:
            cycles.append(cyc)
        else:
            raise ValueError(f"Unexpected orbit length {len(cyc)}")
    return cycles, fixed


def _basis_pair_matrix(dtype=torch.float64) -> torch.Tensor:
    """
    Orthonormal basis for two 4-cycles.
    Columns are [u1, u2, v1, v2, p1, q1, p2, q2].
    """
    U = torch.zeros(8, 8, dtype=dtype)

    # u1, u2
    U[:, 0] = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=dtype) / 2.0
    U[:, 1] = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=dtype) / 2.0

    # v1, v2
    U[:, 2] = torch.tensor([1, -1, 1, -1, 0, 0, 0, 0], dtype=dtype) / 2.0
    U[:, 3] = torch.tensor([0, 0, 0, 0, 1, -1, 1, -1], dtype=dtype) / 2.0

    # p1, q1
    U[:, 4] = torch.tensor([1, 0, -1, 0, 0, 0, 0, 0], dtype=dtype) / math.sqrt(2.0)
    U[:, 5] = torch.tensor([0, 1, 0, -1, 0, 0, 0, 0], dtype=dtype) / math.sqrt(2.0)

    # p2, q2
    U[:, 6] = torch.tensor([0, 0, 0, 0, 1, 0, -1, 0], dtype=dtype) / math.sqrt(2.0)
    U[:, 7] = torch.tensor([0, 0, 0, 0, 0, 1, 0, -1], dtype=dtype) / math.sqrt(2.0)

    return U


class ContinuousQuarterTurn32(nn.Module):
    """
    Continuous Gaussian-law-preserving interpolation between identity
    and exact 90° rotation on a 32x32 grid.

    Input shape: `(32, 32)`, `(C, 32, 32)`, or `(B, C, 32, 32)`
    Output shape: same

    Properties:
      - forward(x, 0)   == x
      - forward(x, 90)  == torch.rot90(x, 1, (-2, -1))
      - forward(x, 180) == torch.rot90(x, 2, (-2, -1))
      - forward(x, 270) == torch.rot90(x, 3, (-2, -1))

    For white Gaussian noise x ~ N(0, I), every forward(x, degrees)
    also has law N(0, I), because the transform is orthogonal.
    """

    def __init__(self):
        super().__init__()
        N = 32
        perm = _rot90_perm(N)
        cycles, fixed = _find_rot90_cycles(perm)

        if len(fixed) != 0:
            raise RuntimeError("32x32 should have no fixed points under 90° rotation.")
        if len(cycles) % 2 != 0:
            raise RuntimeError("32x32 should have an even number of 4-cycles.")

        # Pair 4-cycles into 8D blocks
        pairs = []
        for k in range(0, len(cycles), 2):
            pairs.append(cycles[k] + cycles[k + 1])

        self.register_buffer(
            "pairs_idx",
            torch.tensor(pairs, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "U_base",
            _basis_pair_matrix(dtype=torch.float64),
            persistent=False,
        )

    def _local_matrix(self, phi: torch.Tensor, dtype, device) -> torch.Tensor:
        """
        Batched 8x8 orthogonal matrices for one paired block.
        """
        phi = phi.reshape(-1)
        c = torch.cos(phi)
        s = torch.sin(phi)
        c2 = torch.cos(2.0 * phi)
        s2 = torch.sin(2.0 * phi)

        B = torch.eye(8, dtype=dtype, device=device).unsqueeze(0).repeat(phi.numel(), 1, 1)

        # Rotate (v1, v2) by 2phi
        B[:, 2, 2] = c2
        B[:, 2, 3] = -s2
        B[:, 3, 2] = s2
        B[:, 3, 3] = c2

        # Rotate (p1, q1) by phi
        B[:, 4, 4] = c
        B[:, 4, 5] = -s
        B[:, 5, 4] = s
        B[:, 5, 5] = c

        # Rotate (p2, q2) by phi
        B[:, 6, 6] = c
        B[:, 6, 7] = -s
        B[:, 7, 6] = s
        B[:, 7, 7] = c

        U = self.U_base.to(dtype=dtype, device=device).unsqueeze(0).expand(phi.numel(), -1, -1)
        return U @ B @ U.transpose(-1, -2)

    def forward(self, x: torch.Tensor, degrees: float | torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (32, 32):
            raise ValueError(f"Expected last two dims to be (32, 32), got {x.shape[-2:]}")

        added_batch_dim = False
        added_channel_dim = False
        if x.ndim == 2:
            x_work = x.unsqueeze(0).unsqueeze(0)
            added_batch_dim = True
            added_channel_dim = True
        elif x.ndim == 3:
            x_work = x.unsqueeze(0)
            added_batch_dim = True
        elif x.ndim == 4:
            x_work = x
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D input, got shape {x.shape}")

        batch_size, channels = x_work.shape[:2]
        degrees_tensor = torch.as_tensor(degrees, dtype=x_work.dtype, device=x_work.device)

        if degrees_tensor.ndim == 0:
            phi_rows = degrees_tensor.repeat(batch_size * channels)
        elif degrees_tensor.ndim == 1 and degrees_tensor.numel() == batch_size:
            phi_rows = degrees_tensor.repeat_interleave(channels)
        elif degrees_tensor.ndim == 1 and degrees_tensor.numel() == batch_size * channels:
            phi_rows = degrees_tensor
        else:
            raise ValueError(
                "degrees must be a scalar, one value per batch item, or one value per channel."
            )

        phi_rows = phi_rows * (math.pi / 180.0)
        M = self._local_matrix(phi_rows, dtype=x_work.dtype, device=x_work.device)

        flat = x_work.reshape(batch_size * channels, 32 * 32)
        local = flat[:, self.pairs_idx]
        transformed = torch.matmul(local, M.transpose(-1, -2))

        flat_out = torch.empty_like(flat)
        scatter_idx = self.pairs_idx.reshape(1, -1).expand(flat.shape[0], -1)
        flat_out.scatter_(1, scatter_idx, transformed.reshape(flat.shape[0], -1))

        output = flat_out.reshape(batch_size, channels, 32, 32)
        if added_batch_dim and added_channel_dim:
            return output[0, 0]
        if added_batch_dim:
            return output[0]
        return output

# %%
base_noise = torch.randn(
    1,
    DATASET_SPECS["in_channels"],
    image_size,
    image_size,
)
cq32 = ContinuousQuarterTurn32()
continuous_noise_batch = cq32(
    base_noise.repeat(len(rotation_angles), 1, 1, 1),
    torch.tensor(rotation_angles, dtype=base_noise.dtype),
)
rotated_generations_continuous = generate_from_initial_noise(
    model,
    continuous_noise_batch,
    device=device,
    amp_dtype=amp_dtype,
    num_steps=NUM_SAMPLE_STEPS,
)
display_noises_continuous = normalize_for_display(continuous_noise_batch)

print("Generated samples from `ContinuousQuarterTurn32` applied to the original base noise.")
print(f"Number of rotated inputs: {len(rotation_angles)}")

# %%
plot_rotation_results(
    rotated_generations_continuous,
    rotation_labels,
    noise_images=display_noises_continuous,
    show_noises=True,
    title="Generation from ContinuousQuarterTurn32 noise",
)

# %%
plot_rotation_results(
    rotated_generations_continuous,
    rotation_labels,
    show_noises=False,
    title="Generated samples from ContinuousQuarterTurn32 noise",
)

# %%
