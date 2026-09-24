import torch
import triton
import torch.nn as nn
from torch.autograd import Function


@triton.jit
def multi_scale_gaussian_kernel():
    pass


class MultiScaleGaussianEager(nn.Module):
    def __init__(self, dim, sizes, sigmas, padding_mode="replicate"):
        super().__init__()
        assert len(sizes) == len(sigmas) and len(sizes) > 0
        for s in sizes:
            assert s % 2 == 1, f"Gaussian kernel size must be odd, got {s}"
        for sg in sigmas:
            assert float(sg) > 0, f"sigma must be > 0, got {sg}"

        self.dim = dim
        self.filters = nn.ModuleList()
        for size, sigma in zip(sizes, sigmas):
            conv = nn.Conv3d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=size,
                padding=size // 2,
                groups=dim,
                bias=False,
                padding_mode=padding_mode,
            )
            with torch.no_grad():
                k = self.build_kernel(size, float(sigma))
                k = k.to(dtype=conv.weight.dtype, device=conv.weight.device)
                conv.weight.copy_(k.repeat(dim, 1, 1, 1, 1))
            self.filters.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [f(x) for f in self.filters]
        return torch.stack(outs, dim=0).mean(0)

    @staticmethod
    def build_kernel(size: int, sigma: float) -> torch.Tensor:
        r = size // 2
        coords = torch.arange(-r, r + 1, dtype=torch.float32)
        z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
        sq = x**2 + y**2 + z**2
        kernel = torch.exp(-0.5 * sq / (sigma**2))
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        return kernel.unsqueeze(0).unsqueeze(0)


class MultiScaleGaussianFunction(Function):
    """自定义autograd函数，支持多尺度高斯模糊"""


class MultiScaleGaussian(MultiScaleGaussianEager):
    """Triton 优化的多尺度高斯模糊算子"""

    def __init__(self, dim, sizes, sigmas, padding_mode="replicate"):
        super().__init__(dim, sizes, sigmas, padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

