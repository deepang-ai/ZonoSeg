import torch
import triton.language as tl

TORCH2TRITON = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
}
