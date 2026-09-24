import torch
import triton
import triton.language as tl
from typing import Tuple

dtype = torch.float32
device = "cuda:0"


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def conv3d_kernel(
    input_ptr,
    input_batch_stride,
    input_channel_stride,
    input_depth_stride,
    input_row_stride,
    input_col_stride,
    depth,
    height,
    width,
    channels,
    kernel_ptr,
    kernel_depth,
    kernel_height,
    kernel_width,
    kernel_dim_stride,
    kernel_channel_stride,
    kernel_depth_stride,
    kernel_row_stride,
    kernel_col_stride,
    bias_ptr,
    output_ptr,
    output_depth,
    output_height,
    output_width,
    output_batch_stride,
    output_channel_stride,
    output_depth_stride,
    output_row_stride,
    output_col_stride,
    BLOCK_SIZE_DEPTH: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kernel_idx = tl.program_id(1)
    spatial_idx = tl.program_id(2)
    
    depth_idx = spatial_idx // output_height
    row_idx = spatial_idx % output_height

    bias_offset = kernel_idx
    bias = tl.load(bias_ptr + bias_offset)

    batch_offset = batch_idx * input_batch_stride

    output_batch_offset = batch_idx * output_batch_stride
    output_channel_offset = kernel_idx * output_channel_stride
    output_depth_offset = depth_idx * output_depth_stride
    output_row_offset = row_idx * output_row_stride

    kernel_depth_offset = tl.arange(0, BLOCK_SIZE_DEPTH)
    kernel_depth_mask = kernel_depth_offset[:, None, None] < kernel_depth
    kernel_depth_offset = kernel_depth_offset[:, None, None] * kernel_depth_stride
    
    kernel_row_offset = tl.arange(0, BLOCK_SIZE_ROW)
    kernel_row_mask = kernel_row_offset[None, :, None] < kernel_height
    kernel_row_offset = kernel_row_offset[None, :, None] * kernel_row_stride
    
    kernel_col_offset = tl.arange(0, BLOCK_SIZE_COL)
    kernel_col_mask = kernel_col_offset[None, None, :] < kernel_width
    kernel_col_offset = kernel_col_offset[None, None, :] * kernel_col_stride
    
    kernel_mask = kernel_depth_mask & kernel_row_mask & kernel_col_mask

    for col_idx in range(output_width):
        elem = 0.0

        input_depth_offset = depth_idx * kernel_depth + tl.arange(0, BLOCK_SIZE_DEPTH)
        input_depth_mask = input_depth_offset[:, None, None] < depth
        input_depth_offset = input_depth_offset[:, None, None] * input_depth_stride
        
        input_row_offset = row_idx * kernel_height + tl.arange(0, BLOCK_SIZE_ROW)
        input_row_mask = input_row_offset[None, :, None] < height
        input_row_offset = input_row_offset[None, :, None] * input_row_stride

        input_col_offset = col_idx * kernel_width + tl.arange(0, BLOCK_SIZE_COL)
        input_col_mask = input_col_offset[None, None, :] < width
        input_col_offset = input_col_offset[None, None, :] * input_col_stride
        
        input_mask = input_depth_mask & input_row_mask & input_col_mask

        for c in range(channels):
            input_offset = (
                input_ptr
                + batch_offset
                + c * input_channel_stride
                + input_depth_offset
                + input_row_offset
                + input_col_offset
            )
            input_data = tl.load(
                input_offset, input_mask
            )

            kernel_offset = (
                kernel_ptr
                + kernel_idx * kernel_dim_stride
                + c * kernel_channel_stride
                + kernel_depth_offset
                + kernel_row_offset
                + kernel_col_offset
            )
            kernel_data = tl.load(kernel_offset, kernel_mask)
            dot_prdct = input_data * kernel_data
            elem += tl.sum(dot_prdct)

        output_offset = (
            output_ptr
            + output_batch_offset
            + output_channel_offset
            + output_depth_offset
            + output_row_offset
            + col_idx
        )
        tl.store(output_offset, elem + bias)


def conv3d_triton(
    input: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    assert input.is_cuda and kernel.is_cuda, "Input or kernel is not on GPU"
    assert len(input.shape) == 5, (
        f"Input needs to be 5 dimensional, provided: {input.shape}"
    )
    assert len(kernel.shape) == 5, (
        f"Kernel size needs to be 5 dimensional, provided: {kernel.shape}"
    )
    assert bias.shape[0] == kernel.shape[0], (
        f"Bias dimension should be same as the kernel 1st dimension"
    )

    batch_size, channels, depth, height, width = input.shape
    num_kernels, kernel_depth_dim, kernel_depth, kernel_height, kernel_width = kernel.shape

    assert depth % kernel_depth == 0 and height % kernel_height == 0 and width % kernel_width == 0, (
        f"Input depth, height and width should be divisible by the kernel depth, height and width"
    )
    assert channels == kernel_depth_dim, (
        f"Kernel channel depth ({kernel_depth_dim}) and input channel depth ({channels}) should be same"
    )

    output = torch.empty(
        (batch_size, num_kernels, depth // kernel_depth, height // kernel_height, width // kernel_width),
        device=device,
        dtype=dtype,
    )

    BLOCK_SIZE_DEPTH = triton.next_power_of_2(kernel_depth)
    BLOCK_SIZE_ROW = triton.next_power_of_2(kernel_height)
    BLOCK_SIZE_COL = triton.next_power_of_2(kernel_width)
    output_depth_size = depth // kernel_depth
    output_height_size = height // kernel_height
    grid = (batch_size, num_kernels, output_depth_size * output_height_size)

    conv3d_kernel[grid](
        input_ptr=input,
        input_batch_stride=input.stride(0),
        input_channel_stride=input.stride(1),
        input_depth_stride=input.stride(2),
        input_row_stride=input.stride(3),
        input_col_stride=input.stride(4),
        depth=depth,
        height=height,
        width=width,
        channels=channels,
        kernel_ptr=kernel,
        kernel_depth=kernel_depth,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        kernel_dim_stride=kernel.stride(0),
        kernel_channel_stride=kernel.stride(1),
        kernel_depth_stride=kernel.stride(2),
        kernel_row_stride=kernel.stride(3),
        kernel_col_stride=kernel.stride(4),
        bias_ptr=bias,
        output_ptr=output,
        output_depth=output_depth_size,
        output_height=output_height_size,
        output_width=width // kernel_width,
        output_batch_stride=output.stride(0),
        output_channel_stride=output.stride(1),
        output_depth_stride=output.stride(2),
        output_row_stride=output.stride(3),
        output_col_stride=output.stride(4),
        BLOCK_SIZE_DEPTH=BLOCK_SIZE_DEPTH,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )

    return output


def conv3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for 3D convolution using PyTorch's autograd temporarily
    """
    batch_size, channels, depth, height, width = input.shape
    num_kernels, kernel_channels, kernel_depth, kernel_height, kernel_width = kernel.shape
    
    input_copy = input.clone().detach().requires_grad_(True)
    kernel_copy = kernel.clone().detach().requires_grad_(True)
    bias_copy = torch.zeros(num_kernels, device=input.device, dtype=input.dtype, requires_grad=True)
    
    output = torch.nn.functional.conv3d(
        input_copy, 
        kernel_copy, 
        bias_copy,
        stride=(kernel_depth, kernel_height, kernel_width)
    )
    
    output.backward(grad_output)
    
    return input_copy.grad, kernel_copy.grad, bias_copy.grad


class Conv3DTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return conv3d_triton(input, weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv3d_input(
                input.shape, weight, grad_output,
                stride=(weight.shape[2], weight.shape[3], weight.shape[4])
            )
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv3d_weight(
                input, weight.shape, grad_output,
                stride=(weight.shape[2], weight.shape[3], weight.shape[4])
            )
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3, 4))
        
        return grad_input, grad_weight, grad_bias


class Conv3DTriton(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple):
        super().__init__()

        assert type(kernel_size) == tuple and len(kernel_size) == 3, (
            f"Param kernel size should be a tuple of size 3"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = torch.nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Conv3DTritonFunction.apply(x, self.conv.weight, self.conv.bias)

