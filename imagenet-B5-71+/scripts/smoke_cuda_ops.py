#!/usr/bin/env python
"""Run a tiny DCNv3 CUDA forward/backward check without training a model."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION = ROOT / "vendor" / "InternImage" / "segmentation"
for location in (ROOT, SEGMENTATION):
    value = str(location)
    if value not in sys.path:
        sys.path.insert(0, value)

import torch
from functions.dcnv3_func import DCNv3Function


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run this check inside a GPU-enabled container")

    batch, height, width, groups, channels_per_group = 1, 8, 8, 2, 4
    kernel_points = 9
    image = torch.randn(
        batch, height, width, groups * channels_per_group,
        device="cuda", requires_grad=True)
    offset = torch.zeros(
        batch, height, width, groups * kernel_points * 2,
        device="cuda", requires_grad=True)
    mask = torch.full(
        (batch, height, width, groups * kernel_points),
        1.0 / kernel_points, device="cuda", requires_grad=True)
    output = DCNv3Function.apply(
        image, offset, mask, 3, 3, 1, 1, 1, 1, 1, 1,
        groups, channels_per_group, 1.0, 1)
    output.square().mean().backward()
    torch.cuda.synchronize()

    if output.shape != image.shape or not torch.isfinite(output).all():
        raise RuntimeError("DCNv3 returned invalid output")
    if image.grad is None or not torch.isfinite(image.grad).all():
        raise RuntimeError("DCNv3 returned invalid input gradients")
    print("DCNv3_CUDA_FORWARD_BACKWARD_OK", tuple(output.shape), torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
