import torch
import triton
import torch.nn as nn
import torch.nn.functional as F
import triton.language as tl
from torch.autograd import Function


from nnunetv2.nets.zonoseg_kernel.utils import TORCH2TRITON


@triton.jit
def scharr_edge_kernel(
    input_ptr,
    output_ptr,
    gx_ptr,
    gy_ptr,
    gz_ptr,
    weight_x_ptr,
    weight_y_ptr,
    weight_z_ptr,
    batch_size,
    channels,
    depth,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """优化的3D Scharr边缘检测Triton kernel

    Args:
        input_ptr: 输入张量指针
        output_ptr: 输出张量指针
        weight_x_ptr: X方向权重张量指针 (3x3x3)
        weight_y_ptr: Y方向权重张量指针 (3x3x3)
        weight_z_ptr: Z方向权重张量指针 (3x3x3)
        batch_size, channels, depth, height, width: 张量维度
        BLOCK_SIZE: 块大小
    """
    pid = tl.program_id(0)

    total_elements = batch_size * channels * depth * height * width
    if pid * BLOCK_SIZE >= total_elements:
        return

    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements

    w_idx = idx % width
    h_idx = (idx // width) % height
    d_idx = (idx // (width * height)) % depth
    c_idx = (idx // (width * height * depth)) % channels
    b_idx = idx // (width * height * depth * channels)

    gx = tl.zeros([BLOCK_SIZE], dtype=DTYPE)
    gy = tl.zeros([BLOCK_SIZE], dtype=DTYPE)
    gz = tl.zeros([BLOCK_SIZE], dtype=DTYPE)

    for di in tl.static_range(3):
        for hi in tl.static_range(3):
            for wi in tl.static_range(3):
                d_pos = d_idx + di - 1
                h_pos = h_idx + hi - 1
                w_pos = w_idx + wi - 1

                valid = (
                    (d_pos >= 0)
                    & (d_pos < depth)
                    & (h_pos >= 0)
                    & (h_pos < height)
                    & (w_pos >= 0)
                    & (w_pos < width)
                    & mask
                )

                input_idx = (
                    b_idx * channels * depth * height * width
                    + c_idx * depth * height * width
                    + d_pos * height * width
                    + h_pos * width
                    + w_pos
                )

                input_val = tl.load(input_ptr + input_idx, mask=valid, other=0.0)

                weight_idx = di * 9 + hi * 3 + wi

                weight_x = tl.load(weight_x_ptr + weight_idx)
                weight_y = tl.load(weight_y_ptr + weight_idx)
                weight_z = tl.load(weight_z_ptr + weight_idx)

                gx += input_val * weight_x
                gy += input_val * weight_y
                gz += input_val * weight_z

    gx_sq = gx * gx
    gy_sq = gy * gy
    gz_sq = gz * gz

    edge = tl.sqrt(gx_sq + gy_sq + gz_sq + 1e-6)

    tl.store(output_ptr + idx, edge, mask=mask)
    tl.store(gx_ptr + idx, gx, mask=mask)
    tl.store(gy_ptr + idx, gy, mask=mask)
    tl.store(gz_ptr + idx, gz, mask=mask)


class ScharrEdgeEager(nn.Module):
    """3D Scharr 边缘检测算子"""

    def __init__(self, dim):
        super().__init__()

        scharr_x = torch.tensor(
            [
                [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                [[-6, 0, 6], [-20, 0, 20], [-6, 0, 6]],
                [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
            ],
            dtype=torch.float32,
        )
        scharr_y = torch.tensor(
            [
                [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                [[-6, -20, -6], [0, 0, 0], [6, 20, 6]],
                [[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
            ],
            dtype=torch.float32,
        )
        scharr_z = torch.tensor(
            [
                [[-3, -6, -3], [-10, -20, -10], [-3, -6, -3]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[3, 6, 3], [10, 20, 10], [3, 6, 3]],
            ],
            dtype=torch.float32,
        )

        self.weight_x = nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.weight_y = nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.weight_z = nn.Conv3d(dim, dim, 3, padding=1, groups=dim, bias=False)

        self.weight_x.weight.data = scharr_x.view(1, 1, 3, 3, 3).repeat(dim, 1, 1, 1, 1)
        self.weight_y.weight.data = scharr_y.view(1, 1, 3, 3, 3).repeat(dim, 1, 1, 1, 1)
        self.weight_z.weight.data = scharr_z.view(1, 1, 3, 3, 3).repeat(dim, 1, 1, 1, 1)

    def forward(self, x):
        """传统 PyTorch 实现"""
        gx = self.weight_x(x)
        gy = self.weight_y(x)
        gz = self.weight_z(x)

        edge = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-6)
        return edge


class ScharrEdgeFunction(Function):
    """自定义autograd函数，支持3D Scharr边缘检测的前向和反向传播"""

    @staticmethod
    def forward(ctx, x, weight_x, weight_y, weight_z):
        """前向传播"""
        batch_size, channels, depth, height, width = x.shape

        output = torch.empty_like(x, device=x.device, dtype=x.dtype)
        gx = torch.empty_like(x, device=x.device, dtype=x.dtype)
        gy = torch.empty_like(x, device=x.device, dtype=x.dtype)
        gz = torch.empty_like(x, device=x.device, dtype=x.dtype)

        weight_x = weight_x.to(dtype=x.dtype, device=x.device)
        weight_y = weight_y.to(dtype=x.dtype, device=x.device)
        weight_z = weight_z.to(dtype=x.dtype, device=x.device)

        weight_x_flat = weight_x.view(-1).contiguous()
        weight_y_flat = weight_y.view(-1).contiguous()
        weight_z_flat = weight_z.view(-1).contiguous()

        total_elements = batch_size * channels * depth * height * width
        BLOCK_SIZE = 1024
        grid_size = triton.cdiv(total_elements, BLOCK_SIZE)

        scharr_edge_kernel[(grid_size,)](
            x.contiguous(),
            output,
            gx.contiguous(),
            gy.contiguous(),
            gz.contiguous(),
            weight_x_flat,
            weight_y_flat,
            weight_z_flat,
            batch_size,
            channels,
            depth,
            height,
            width,
            BLOCK_SIZE=BLOCK_SIZE,
            DTYPE=TORCH2TRITON[x.dtype],
        )

        ctx.save_for_backward(x, weight_x, weight_y, weight_z, gx, gy, gz, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        x, weight_x, weight_y, weight_z, gx, gy, gz, output = ctx.saved_tensors

        grad_gx = grad_output * gx / (output + 1e-8)
        grad_gy = grad_output * gy / (output + 1e-8)
        grad_gz = grad_output * gz / (output + 1e-8)

        weight_x_flipped = torch.flip(weight_x, [2, 3, 4])
        weight_y_flipped = torch.flip(weight_y, [2, 3, 4])
        weight_z_flipped = torch.flip(weight_z, [2, 3, 4])

        grad_input = (
            F.conv3d(grad_gx, weight_x_flipped, groups=x.shape[1], padding=1)
            + F.conv3d(grad_gy, weight_y_flipped, groups=x.shape[1], padding=1)
            + F.conv3d(grad_gz, weight_z_flipped, groups=x.shape[1], padding=1)
        )

        grad_weight_x = None
        grad_weight_y = None
        grad_weight_z = None

        if weight_x.requires_grad:
            grad_weight_x = torch.zeros_like(weight_x)
            for c in range(x.shape[1]):
                x_c = x[:, c : c + 1]
                grad_gx_c = grad_gx[:, c : c + 1]

                grad_w = F.conv3d(x_c, grad_gx_c.flip([2, 3, 4]), padding=1)
                grad_weight_x[c, 0] = grad_w[0, 0]

        if weight_y.requires_grad:
            grad_weight_y = torch.zeros_like(weight_y)
            for c in range(x.shape[1]):
                x_c = x[:, c : c + 1]
                grad_gy_c = grad_gy[:, c : c + 1]

                grad_w = F.conv3d(x_c, grad_gy_c.flip([2, 3, 4]), padding=1)
                grad_weight_y[c, 0] = grad_w[0, 0]

        if weight_z.requires_grad:
            grad_weight_z = torch.zeros_like(weight_z)
            for c in range(x.shape[1]):
                x_c = x[:, c : c + 1]
                grad_gz_c = grad_gz[:, c : c + 1]

                grad_w = F.conv3d(x_c, grad_gz_c.flip([2, 3, 4]), padding=1)
                grad_weight_z[c, 0] = grad_w[0, 0]

        return grad_input, grad_weight_x, grad_weight_y, grad_weight_z


class ScharrEdge(ScharrEdgeEager):
    """Triton 优化的 3D Scharr 边缘检测算子"""

    def __init__(self, dim):
        super().__init__(dim)

    def forward(self, x):
        if not x.is_cuda or (self.training and torch.is_grad_enabled()):
            return super().forward(x)

        return ScharrEdgeFunction.apply(
            x, self.weight_x.weight, self.weight_y.weight, self.weight_z.weight
        )

