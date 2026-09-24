import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_WIDTH": 4}, num_warps=4),
        triton.Config({"BLOCK_WIDTH": 8}, num_warps=4),
        triton.Config({"BLOCK_WIDTH": 16}, num_warps=4),
        triton.Config({"BLOCK_WIDTH": 32}, num_warps=4),
        triton.Config({"BLOCK_WIDTH": 64}, num_warps=4),
        triton.Config({"BLOCK_WIDTH": 8}, num_warps=8),
        triton.Config({"BLOCK_WIDTH": 16}, num_warps=8),
        triton.Config({"BLOCK_WIDTH": 32}, num_warps=8),
        triton.Config({"BLOCK_WIDTH": 64}, num_warps=8),
        triton.Config({"BLOCK_WIDTH": 16}, num_warps=16),
        triton.Config({"BLOCK_WIDTH": 32}, num_warps=16),
        triton.Config({"BLOCK_WIDTH": 64}, num_warps=16),
        triton.Config({"BLOCK_WIDTH": 128}, num_warps=8),
        triton.Config({"BLOCK_WIDTH": 128}, num_warps=16),
    ],
    key=["channels", "output_width"],
)
@triton.jit
def conv3d_dual_kernel_3x5(
    input_ptr,
    input_batch_stride,
    input_channel_stride,
    input_depth_stride,
    input_row_stride,
    input_col_stride,
    orig_depth,
    orig_height,
    orig_width,
    channels,
    kernel1_ptr,
    kernel1_dim_stride,
    kernel1_depth_stride,
    kernel1_row_stride,
    kernel1_col_stride,
    kernel2_ptr,
    kernel2_dim_stride,
    kernel2_depth_stride,
    kernel2_row_stride,
    kernel2_col_stride,
    output_ptr,
    output_depth,
    output_height,
    output_width,
    output_batch_stride,
    output_channel_stride,
    output_depth_stride,
    output_row_stride,
    output_col_stride,
    BLOCK_WIDTH: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    spatial_idx = tl.program_id(2)

    depth_idx = spatial_idx // output_height
    row_idx = spatial_idx % output_height

    batch_offset = batch_idx * input_batch_stride
    channel_offset = channel_idx * input_channel_stride

    output_batch_offset = batch_idx * output_batch_stride
    output_channel_offset = channel_idx * output_channel_stride
    output_depth_offset = depth_idx * output_depth_stride
    output_row_offset = row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_WIDTH)
    col_mask = col_offsets < output_width

    accumulator_dtype = tl.float32
    results1 = tl.zeros([BLOCK_WIDTH], dtype=accumulator_dtype)
    results2 = tl.zeros([BLOCK_WIDTH], dtype=accumulator_dtype)
    
    base_input_offset = batch_offset + channel_offset
    base_kernel1_offset = channel_idx * kernel1_dim_stride
    base_kernel2_offset = channel_idx * kernel2_dim_stride

    for kd in tl.static_range(3):
        for kr in tl.static_range(3):
            for kc in tl.static_range(3):
                input_d = depth_idx + kd - 1
                input_r = row_idx + kr - 1
                input_c = col_offsets + kc - 1

                input_d_clamped = tl.maximum(0, tl.minimum(input_d, orig_depth - 1))
                input_r_clamped = tl.maximum(0, tl.minimum(input_r, orig_height - 1))
                input_c_clamped = tl.maximum(0, tl.minimum(input_c, orig_width - 1))

                input_offsets = (
                    base_input_offset
                    + input_d_clamped * input_depth_stride
                    + input_r_clamped * input_row_stride
                    + input_c_clamped * input_col_stride
                )
                input_vals = tl.load(input_ptr + input_offsets, mask=col_mask, other=0.0)

                kernel1_offset = (
                    base_kernel1_offset
                    + kd * kernel1_depth_stride
                    + kr * kernel1_row_stride
                    + kc * kernel1_col_stride
                )
                kernel1_val = tl.load(kernel1_ptr + kernel1_offset)
                results1 = tl.fma(input_vals, kernel1_val, results1)

    for kd in tl.static_range(5):
        for kr in tl.static_range(5):
            for kc in tl.static_range(5):
                input_d = depth_idx + kd - 2
                input_r = row_idx + kr - 2
                input_c = col_offsets + kc - 2

                input_d_clamped = tl.maximum(0, tl.minimum(input_d, orig_depth - 1))
                input_r_clamped = tl.maximum(0, tl.minimum(input_r, orig_height - 1))
                input_c_clamped = tl.maximum(0, tl.minimum(input_c, orig_width - 1))

                input_offsets = (
                    base_input_offset
                    + input_d_clamped * input_depth_stride
                    + input_r_clamped * input_row_stride
                    + input_c_clamped * input_col_stride
                )
                input_vals = tl.load(input_ptr + input_offsets, mask=col_mask, other=0.0)

                kernel2_offset = (
                    base_kernel2_offset
                    + kd * kernel2_depth_stride
                    + kr * kernel2_row_stride
                    + kc * kernel2_col_stride
                )
                kernel2_val = tl.load(kernel2_ptr + kernel2_offset)
                results2 = tl.fma(input_vals, kernel2_val, results2)

    final_results = tl.fma(results1, 0.5, results2 * 0.5)
    
    output_base_offset = (
        output_batch_offset
        + output_channel_offset
        + output_depth_offset
        + output_row_offset
    )
    output_offsets = output_base_offset + col_offsets * output_col_stride
    
    tl.store(output_ptr + output_offsets, final_results, mask=col_mask)


def conv3d_dual_triton(
    input: torch.Tensor,
    kernel1: torch.Tensor,
    kernel2: torch.Tensor,
    padding1: int = 0,
    padding2: int = 0,
) -> torch.Tensor:
    assert input.is_cuda and kernel1.is_cuda and kernel2.is_cuda, (
        "Input or kernels are not on GPU"
    )
    assert len(input.shape) == 5, (
        f"Input needs to be 5 dimensional, provided: {input.shape}"
    )
    assert len(kernel1.shape) == 5, (
        f"Kernel1 size needs to be 5 dimensional, provided: {kernel1.shape}"
    )
    assert len(kernel2.shape) == 5, (
        f"Kernel2 size needs to be 5 dimensional, provided: {kernel2.shape}"
    )

    assert input.dtype in [torch.float32, torch.float16, torch.bfloat16], (
        f"Unsupported input dtype: {input.dtype}. Supported: float32, float16, bfloat16"
    )
    assert kernel1.dtype in [torch.float32, torch.float16, torch.bfloat16], (
        f"Unsupported kernel1 dtype: {kernel1.dtype}. Supported: float32, float16, bfloat16"
    )
    assert kernel2.dtype in [torch.float32, torch.float16, torch.bfloat16], (
        f"Unsupported kernel2 dtype: {kernel2.dtype}. Supported: float32, float16, bfloat16"
    )
    assert input.dtype == kernel1.dtype == kernel2.dtype, (
        f"Input and kernels must have the same dtype, got {input.dtype}, {kernel1.dtype}, {kernel2.dtype}"
    )

    batch_size, channels, depth, height, width = input.shape
    out_channels1, kernel1_depth_dim, kernel_size1_d, kernel_size1_h, kernel_size1_w = (
        kernel1.shape
    )
    out_channels2, kernel2_depth_dim, kernel_size2_d, kernel_size2_h, kernel_size2_w = (
        kernel2.shape
    )

    assert kernel_size1_d == kernel_size1_h == kernel_size1_w, (
        f"All kernel1 dimensions must be equal, got {kernel_size1_d}, {kernel_size1_h}, {kernel_size1_w}"
    )
    assert kernel_size2_d == kernel_size2_h == kernel_size2_w, (
        f"All kernel2 dimensions must be equal, got {kernel_size2_d}, {kernel_size2_h}, {kernel_size2_w}"
    )
    kernel_size1 = kernel_size1_d
    kernel_size2 = kernel_size2_d

    assert depth >= kernel_size1 and height >= kernel_size1 and width >= kernel_size1, (
        f"Input dimensions must be at least as large as kernel1 size for stride=1"
    )
    assert depth >= kernel_size2 and height >= kernel_size2 and width >= kernel_size2, (
        f"Input dimensions must be at least as large as kernel2 size for stride=1"
    )
    assert kernel1_depth_dim == 1, (
        f"For groups=channels, kernel1 channel depth should be 1, got {kernel1_depth_dim}"
    )
    assert kernel2_depth_dim == 1, (
        f"For groups=channels, kernel2 channel depth should be 1, got {kernel2_depth_dim}"
    )
    assert channels == out_channels1 == out_channels2, (
        f"Input channels ({channels}) and output channels ({out_channels1}, {out_channels2}) must be the same"
    )

    output_depth = depth
    output_height = height
    output_width = width

    output = torch.empty(
        (batch_size, channels, output_depth, output_height, output_width),
        device=input.device,
        dtype=input.dtype,
    )

    assert kernel_size1 == 3 and kernel_size2 == 5, (
        f"Only 3x5 kernel sizes are supported, got {kernel_size1}x{kernel_size2}"
    )

    grid = (batch_size, channels, output_depth * output_height)

    conv3d_dual_kernel_3x5[grid](
        input_ptr=input,
        input_batch_stride=input.stride(0),
        input_channel_stride=input.stride(1),
        input_depth_stride=input.stride(2),
        input_row_stride=input.stride(3),
        input_col_stride=input.stride(4),
        orig_depth=depth,
        orig_height=height,
        orig_width=width,
        channels=channels,
        kernel1_ptr=kernel1,
        kernel1_dim_stride=kernel1.stride(0),
        kernel1_depth_stride=kernel1.stride(2),
        kernel1_row_stride=kernel1.stride(3),
        kernel1_col_stride=kernel1.stride(4),
        kernel2_ptr=kernel2,
        kernel2_dim_stride=kernel2.stride(0),
        kernel2_depth_stride=kernel2.stride(2),
        kernel2_row_stride=kernel2.stride(3),
        kernel2_col_stride=kernel2.stride(4),
        output_ptr=output,
        output_depth=output_depth,
        output_height=output_height,
        output_width=output_width,
        output_batch_stride=output.stride(0),
        output_channel_stride=output.stride(1),
        output_depth_stride=output.stride(2),
        output_row_stride=output.stride(3),
        output_col_stride=output.stride(4),
    )

    return output


class Conv3DTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight1, weight2):
        ctx.save_for_backward(input, weight1, weight2)

        kernel_size1 = weight1.shape[2]
        kernel_size2 = weight2.shape[2]

        padding1 = kernel_size1 // 2
        padding2 = kernel_size2 // 2

        output = conv3d_dual_triton(input, weight1, weight2, padding1, padding2)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, weight2 = ctx.saved_tensors

        kernel_size1 = weight1.shape[2]
        kernel_size2 = weight2.shape[2]

        padding1 = kernel_size1 // 2
        padding2 = kernel_size2 // 2

        grad_input = None
        grad_weight1 = None
        grad_weight2 = None

        grad_output_half = grad_output / 2.0

        if ctx.needs_input_grad[0]:
            with torch.enable_grad():
                input_for_grad = input.detach().clone().requires_grad_(True)

                padded_input1 = torch.nn.functional.pad(
                    input_for_grad,
                    (padding1, padding1, padding1, padding1, padding1, padding1),
                    mode="replicate",
                )
                padded_input2 = torch.nn.functional.pad(
                    input_for_grad,
                    (padding2, padding2, padding2, padding2, padding2, padding2),
                    mode="replicate",
                )
                output1 = torch.nn.functional.conv3d(
                    padded_input1, weight1, padding=0, groups=input.shape[1]
                )
                output2 = torch.nn.functional.conv3d(
                    padded_input2, weight2, padding=0, groups=input.shape[1]
                )

                grad_input = torch.autograd.grad(
                    outputs=[output1, output2],
                    inputs=input_for_grad,
                    grad_outputs=[grad_output_half, grad_output_half],
                )[0]

        if ctx.needs_input_grad[1]:
            input_padded = torch.nn.functional.pad(
                input,
                (padding1, padding1, padding1, padding1, padding1, padding1),
                mode="replicate",
            )
            grad_weight1 = torch.nn.grad.conv3d_weight(
                input_padded,
                weight1.shape,
                grad_output_half,
                stride=1,
                padding=0,
                groups=input.shape[1],
            )

        if ctx.needs_input_grad[2]:
            input_padded = torch.nn.functional.pad(
                input,
                (padding2, padding2, padding2, padding2, padding2, padding2),
                mode="replicate",
            )
            grad_weight2 = torch.nn.grad.conv3d_weight(
                input_padded,
                weight2.shape,
                grad_output_half,
                stride=1,
                padding=0,
                groups=input.shape[1],
            )

        return grad_input, grad_weight1, grad_weight2, None


class Conv3DTriton(torch.nn.Module):
    def __init__(self, channels: int, kernel_sizes: Tuple[int, int]):
        super().__init__()

        self.channels = channels
        self.kernel_size1 = kernel_sizes[0]
        self.kernel_size2 = kernel_sizes[1]

        self.conv1 = torch.nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=self.kernel_size1,
            stride=1,
            bias=False,
            groups=channels,
        )
        self.conv2 = torch.nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=self.kernel_size2,
            stride=1,
            bias=False,
            groups=channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Conv3DTritonFunction.apply(x, self.conv1.weight, self.conv2.weight)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "channels", "depth", "height", "width"],
        x_vals=[
            (2, 96, 16, 16, 16),
            (4, 24, 32, 64, 64),
            (8, 48, 32, 32, 32),
            (8, 64, 32, 64, 64),
            (8, 24, 32, 128, 128),
        ],
        line_arg="provider",
        line_vals=["Triton", "PyTorch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="Runtime (ms)",
        plot_name="3D Convolution Performance",
        args={
            "kernel_size1": 3,
            "kernel_size2": 5,
            "dtype": torch.float32,
        },
    )
)
def benchmark(
    batch_size,
    channels,
    depth,
    height,
    width,
    kernel_size1,
    kernel_size2,
    dtype,
    provider,
):
    padding1 = kernel_size1 // 2
    padding2 = kernel_size2 // 2

    input_tensor = torch.randn(
        batch_size,
        channels,
        depth,
        height,
        width,
        device="cuda",
        dtype=dtype,
    )

    if provider == "Triton":
        triton_model = Conv3DTriton(channels, (kernel_size1, kernel_size2)).to(
            device="cuda", dtype=dtype
        )
        ms = triton.testing.do_bench(lambda: triton_model(input_tensor))
    else:
        padding1_torch = kernel_size1 // 2
        padding2_torch = kernel_size2 // 2

        conv3d_torch1 = torch.nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size1,
            stride=1,
            padding=padding1_torch,
            bias=False,
            device="cuda",
            dtype=dtype,
            groups=channels,
            padding_mode="replicate",
        )
        conv3d_torch2 = torch.nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size2,
            stride=1,
            padding=padding2_torch,
            bias=False,
            device="cuda",
            dtype=dtype,
            groups=channels,
            padding_mode="replicate",
        )

        def torch_fn():
            output1 = conv3d_torch1(input_tensor)
            output2 = conv3d_torch2(input_tensor)
            return (output1 + output2) / 2.0

        ms = triton.testing.do_bench(torch_fn)

    return ms

