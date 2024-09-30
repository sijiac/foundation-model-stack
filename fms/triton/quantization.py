from typing import Optional, Tuple

import torch
import triton  # @manual

import triton.language as tl  # @manual
from torch._tensor import Tensor
from triton import Config  # @manual


def get_fp8_constants() -> Tuple[torch.dtype, tl.dtype, float, float]:
    """
    Helper function to get constant values for the current platform.

    Returns:
        pt_dtype (torch.dtype): The correct torch fp8 datatype.
        tl_dtype (tl.dtype): The correct triton fp8 datatype.
        max_fp8 (float): The maximum reprsentable value for the fp8 datatype.
        eps (float): Minimum clip value to prevent divide by zero.
    """
    if torch.version.hip is not None:
        pt_fp8_dtype = torch.float8_e4m3fnuz
        tl_fp8_dtype = tl.float8e4b8
    else:
        pt_fp8_dtype = torch.float8_e4m3fn
        tl_fp8_dtype = tl.float8e4nv
    return pt_fp8_dtype, tl_fp8_dtype, torch.finfo(pt_fp8_dtype).max, 1e-12


@triton.autotune(
    configs=[
        Config({"BLOCK_SIZE": 512}),
        Config({"BLOCK_SIZE": 1024}),
        Config({"BLOCK_SIZE": 2048}),
        Config({"BLOCK_SIZE": 4096}),
        Config({"BLOCK_SIZE": 8192}),
    ],
    key=["N"],
)
@triton.jit
def _kernel_quantize_fp8_row(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    M,
    N,
    stride_am,
    stride_an,
    stride_om,
    stride_on,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Quantize and scale each row.

    Scale per row i is computed as MAX_FP8 / max(abs(A[i, :]))

    Kernel naively iterates through  matrix with [1, BLOCK_SIZE] tiles
    in a max pass then scale/quantize pass.

    Todo:
        * Better tiling schemes.

    Args:
        A (Tensor): [m, n] higher precision input tensor.
        A_scale (Tensor): [m] reciprocal scale tensor per row.
        A_fp8 (Tensor): [m, n] fp8 scaled tensor. A_fp8 = A / a_scale
        scale_ub (Tensor): [1] Maximum value allowed for scale.
        M (int): Number of rows.
        N (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_an (int): Stride of n dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_on (int): Stride of n dimension of output.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whethar to apply scale_ub.
        BLOCK_SIZE (int): Block size for reduction.
    """
    pid = tl.program_id(0)
    n_offset = tl.arange(0, BLOCK_SIZE)

    # Calculate max.
    cur_max = 0.0
    for _k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        a = tl.load(
            A + pid * stride_am + n_offset * stride_an, mask=n_offset < N, other=0.0
        )
        tile_max = tl.max(tl.abs(a))
        cur_max = tl.maximum(tile_max, cur_max)

        n_offset += BLOCK_SIZE

    # Clamp max value appropriately.
    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        cur_max = tl.clamp(cur_max, EPS, ub)
    else:
        cur_max = tl.maximum(cur_max, EPS)
    # Scale and quantize.
    a_scale = MAX_FP8 / cur_max
    tl.store(A_scale + pid, 1.0 / a_scale)
    n_offset = tl.arange(0, BLOCK_SIZE)
    for _k in range(0, tl.cdiv(N, BLOCK_SIZE)):
        a = tl.load(
            A + pid * stride_am + n_offset * stride_an, mask=n_offset < N, other=0.0
        )
        a_fp8 = a * a_scale
        # Clamp A to fp8 range to make sure there's no overflow.
        # This is required for AMD. Nvidia's default saturation
        # handles it, but it's nice to have anyway.
        a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8)
        a_fp8.to(TL_FP8_DTYPE)
        tl.store(
            A_fp8 + pid * stride_om + n_offset * stride_on, a_fp8, mask=n_offset < N
        )
        n_offset += BLOCK_SIZE


def triton_quantize_fp8_row(
    a: Tensor, scale_ub: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Call the triton quantize fp8 row kernel to quantize a tensor to fp8 with row-wise scalings.

    Args:
        a (Tensor): [m, n] higher precision input tensor.
        scale_ub (Tensor): Maximum allowed value for scale.

    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: reciprocal scale tensor per row.
    """
    a_shape = a.shape
    # breakpoint()
    a = a.reshape(-1, a.size(-1))
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()

    num_rows = a.shape[0]
    a_scale = torch.empty((num_rows), dtype=torch.float32, device=a.device)
    a_fp8 = torch.empty((a.shape[0], a.shape[1]), device=a.device, dtype=pt_dtype)

    grid = (num_rows,)
    _kernel_quantize_fp8_row[grid](
        a,
        a_scale,
        a_fp8,
        scale_ub,
        a.shape[0],
        a.shape[1],
        a.stride(0),
        a.stride(1),
        a_fp8.stride(0),
        a_fp8.stride(1),
        TL_FP8_DTYPE=tl_dtype,
        MAX_FP8=max_fp8,
        EPS=eps,
        CLAMP_MAX=scale_ub is not None,
    )

    return a_fp8.view(a_shape), a_scale.view(a_shape[:-1])

def quantize_fp8_tensorwise_pt(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    x_scale = x.abs().max() / fp8_max
    # pyre-ignore [58]
    x_invs_scale = 1.0 / x_scale
    xq = (x * x_invs_scale).to(torch.float8_e4m3fn)
    x_scale = x_scale.float()
    return xq, x_scale
