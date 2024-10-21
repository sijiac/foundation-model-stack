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


@triton.jit
def _kernel_quantize_fp8_block(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    M,
    K,
    stride_am,
    stride_ak,
    stride_om,
    stride_ok,
    stride_a_scale_m,
    stride_a_scale_k,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Quantize and scale each [BLOCK_M, BLOCK_K] block.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(A[i:i+BLOCK_M, j:j+BLOCK_K])))

    Kernel naively iterates through  matrix with [BLOCK_M, BLOCK_K] tiles.

    Todo:
        * Better tiling and ordering schemes.

    Args:
        A (Tensor): [M, K] higher precision input tensor.
        A_scale (Tensor): [cdiv(M, BLOCK_M), cdiv(K, BLOCK_K)] reciprocal scale tensor per block.
        A_fp8 (Tensor): [M, K] fp8 scaled tensor. A_fp8 = A * a_scale
        scale_ub (Tensor): [1] Maximum allowed value for scale.
        M (int): Number of rows.
        K (int): Number of columns.
        stride_am (int): Stride of m dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_om (int): Stride of m dimension of output.
        stride_ok (int): Stride of k dimension of output.
        stride_a_scale_m (int): Stride of m dimension of A_scale.
        stride_a_scale_k (int): Stride of k dimension of A_scale.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maxmimum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whether to apply scale_ub.
        BLOCK_M (int): Block size for M dimension of A_scale and kernel.
        BLOCK_K (int): Block size for K dimension of A_scale and kernel.
    """
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BLOCK_K)
    block_m = pid // grid_k
    block_k = pid % grid_k
    rm = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_offset = rm[:, None] * stride_am + rk[None, :] * stride_ak
    out_offset = rm[:, None] * stride_om + rk[None, :] * stride_ok
    a_mask = (rm < M)[:, None] & (rk < K)[None, :]
    a_block = tl.load(A + a_offset, mask=a_mask, other=0.0)

    block_max = tl.max(tl.abs(a_block))
    # Apply appropriate clamping.
    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        block_max = tl.clamp(block_max, EPS, ub)
    else:
        block_max = tl.maximum(block_max, EPS)
    scale = MAX_FP8 / block_max

    tl.store(
        A_scale + block_m * stride_a_scale_m + block_k * stride_a_scale_k, 1.0 / scale
    )
    a_fp8 = a_block * scale
    # Clamp A to fp8 range to make sure there's no overflow.
    # This is required for AMD. Nvidia's default saturation
    # handles it, but it's nice to have anyway.
    a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8)
    a_fp8.to(TL_FP8_DTYPE)
    tl.store(A_fp8 + out_offset, a_fp8, mask=a_mask)


def triton_quantize_fp8_block(
    x: torch.Tensor,
    block_m: int = 256,
    block_k: int = 256,
    scale_ub: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with block-wise scalings.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(x[i:i+block_m, j:j+block_k])))

    Args:
        x (torch.Tensor): [M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        block_k (int): Block size for K dimension of scale.
        scale_ub: Maximum allowed value for scale.

    Returns:
        torch.Tensor : [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] reciprocal scale tensor per block.
    """
    assert x.device != torch.device(
        "cpu"
    ), "Blockwise quantization not support on cpu, please use row-wise quantization instead."
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()
    M, K = x.shape
    grid_m = triton.cdiv(M, block_m)
    grid_k = triton.cdiv(K, block_k)
    x_scale = torch.ones((grid_m, grid_k), device=x.device, dtype=torch.float32)
    x_fp8 = torch.empty((M, K), device=x.device, dtype=pt_dtype)

    _kernel_quantize_fp8_block[(grid_m * grid_k,)](
        x,
        x_scale,
        x_fp8,
        scale_ub,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        # pyre-ignore[6]: Incompatible parameter type [6]
        TL_FP8_DTYPE=tl_dtype,
        # pyre-ignore[6]: Incompatible parameter type [6]
        MAX_FP8=max_fp8,
        # pyre-ignore[6]: Incompatible parameter type [6]
        EPS=eps,
        # pyre-ignore[6]: Incompatible parameter type [6]
        CLAMP_MAX=scale_ub is not None,
        # pyre-ignore[6]: Incompatible parameter type [6]
        BLOCK_M=block_m,
        # pyre-ignore[6]: Incompatible parameter type [6]
        BLOCK_K=block_k,
    )

    return x_fp8.view(x_shape), x_scale.view(list(x_shape[:-2]) + [-1, grid_k])


def triton_quantize_fp8_block_per_head(
    x: torch.Tensor,
    block_m: int = 256,
    scale_ub: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 4D tensor to fp8 with block-wise scalings.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(x[b,h,i:i+block_m, :])))

    Args:
        x (torch.Tensor): [B, H, M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        scale_ub: Maximum allowed value for scale.

    Returns:
        torch.Tensor : [B, H, M, K] fp8 scaled tensor.
        torch.Tensor : [B, H, num_of_bm] reciprocal scale tensor per block.
    """
    assert x.device != torch.device(
        "cpu"
    ), "Blockwise quantization not supported on cpu, please use row-wise quantization instead."

    B, H, M, K = x.shape

    # Get constant values.
    pt_dtype, tl_dtype, max_fp8, eps = get_fp8_constants()

    grid_bh = triton.cdiv(B * H, 1)
    grid_m = triton.cdiv(M, block_m)
    x_scale = torch.empty((B, H, grid_m), device=x.device, dtype=torch.float32)
    x_fp8 = torch.empty_like(x, dtype=pt_dtype)

    _kernel_quantize_fp8_block_per_head[(grid_bh, grid_m)](
        x,
        x_scale,
        x_fp8,
        scale_ub,
        B,
        H,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_fp8.stride(2),
        x_fp8.stride(3),
        x_scale.stride(0),
        x_scale.stride(1),
        x_scale.stride(2),
        TL_FP8_DTYPE=tl_dtype,
        MAX_FP8=max_fp8,
        EPS=eps,
        CLAMP_MAX=scale_ub is not None,
        BLOCK_M=block_m,
        BLOCK_K=K,
    )

    return x_fp8, x_scale


@triton.jit
def _kernel_quantize_fp8_block_per_head(
    A,
    A_scale,
    A_fp8,
    scale_ub,
    B,
    H,
    M,
    K,
    stride_ab,
    stride_ah,
    stride_am,
    stride_ak,
    stride_fp8_b,
    stride_fp8_h,
    stride_fp8_m,
    stride_fp8_k,
    stride_scale_b,
    stride_scale_h,
    stride_scale_m,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """Quantize and scale each [BLOCK_M, BLOCK_K] block.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(A[b,h,i:i+BLOCK_M, :])))

    Kernel naively iterates through matrix with [BLOCK_M, BLOCK_K] tiles.

    Todo:
        * Better tiling and ordering schemes.

    Args:
        A (Tensor): [B, H, M, K] higher precision input tensor.
        A_scale (Tensor): [B, H, num_of_bm] reciprocal scale tensor per block.
        A_fp8 (Tensor): [B, H, M, K] fp8 scaled tensor. A_fp8 = A * a_scale
        scale_ub (Tensor): [1] Maximum allowed value for scale.
        B (int): Batch size.
        H (int): Number of heads.
        M (int): Number of rows.
        K (int): Number of columns.
        stride_ab (int): Stride of b dimension of A.
        stride_ah (int): Stride of h dimension of A.
        stride_am (int): Stride of m dimension of A.
        stride_ak (int): Stride of k dimension of A.
        stride_fp8_b (int): Stride of b dimension of output.
        stride_fp8_h (int): Stride of h dimension of output.
        stride_fp8_m (int): Stride of m dimension of output.
        stride_fp8_k (int): Stride of k dimension of output.
        stride_scale_b (int): Stride of b dimension of A_scale.
        stride_scale_h (int): Stride of h dimension of A_scale.
        stride_scale_m (int): Stride of m dimension of A_scale.
        TL_FP8_DTYPE (tl.dtype): Target fp8 datatype.
        MAX_FP8 (float): Maximum expressible value for FP8.
        EPS (float): Epsilon value for numerical stability.
        CLAMP_MAX (bool): Whether to apply scale_ub.
        BLOCK_M (int): Block size for M dimension of A_scale and kernel.
        BLOCK_K (int): Block size for K dimension of A_scale and kernel.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    bm = pid_m

    b = pid_bh // H
    h = pid_bh % H

    m = bm * BLOCK_M

    a_offset = b * stride_ab + h * stride_ah + m * stride_am
    out_offset = b * stride_fp8_b + h * stride_fp8_h + m * stride_fp8_m
    a_scale_offset = b * stride_scale_b + h * stride_scale_h + bm * stride_scale_m

    a_block = tl.load(
        A
        + a_offset
        + tl.arange(0, BLOCK_M)[:, None] * stride_am
        + tl.arange(0, BLOCK_K)[None, :] * stride_ak,
        mask=(tl.arange(0, BLOCK_M)[:, None] < M - m)
        & (tl.arange(0, BLOCK_K)[None, :] < K),
        other=0.0,
    )

    block_max = tl.max(tl.abs(a_block))

    if CLAMP_MAX:
        ub = tl.load(scale_ub)
        block_max = tl.clamp(block_max, EPS, ub)
    else:
        block_max = tl.maximum(block_max, EPS)

    scale = MAX_FP8 / block_max

    tl.store(A_scale + a_scale_offset, 1.0 / scale)

    a_fp8 = a_block * scale

    # Clamp A to fp8 range to make sure there's no overflow.
    # This is required for AMD. Nvidia's default saturation
    # handles it, but it's nice to have anyway.
    a_fp8 = tl.clamp(a_fp8, -MAX_FP8, MAX_FP8)
    a_fp8 = a_fp8.to(TL_FP8_DTYPE)

    tl.store(
        A_fp8
        + out_offset
        + tl.arange(0, BLOCK_M)[:, None] * stride_fp8_m
        + tl.arange(0, BLOCK_K)[None, :] * stride_fp8_k,
        a_fp8,
        mask=(tl.arange(0, BLOCK_M)[:, None] < M - m)
        & (tl.arange(0, BLOCK_K)[None, :] < K),
    )


def quantize_fp8_block_eager(
    x: torch.Tensor,
    block_m: int = 128,
    scale_ub: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with block-wise scalings and optionally move to output device.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(x[i:i+block_m, j:j+block_k])))

    Args:
        x (Tensor): [M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        block_k (int): Block size for K dimension of scale.
        scale_ub: Maximum allowed value for scale.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.

    Returns:
        torch.Tensor: [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] reciprocal scale tensor per block.
    """

    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()

    B, H, M, K = x.shape
    grid_m = triton.cdiv(M, block_m)

    # Pad x to multiple of block size.
    padded_m = grid_m * block_m
    x_padded = torch.zeros(B, H, padded_m, K, dtype=x.dtype, device=x.device)
    x_padded[:, :, :M, :] = x

    # Blockwise max.
    block_max = (
        x_padded.abs().reshape(B, H, grid_m, block_m, K).amax(dim=(-2, -1))
    )

    # Apply clamping.
    if scale_ub is not None:
        block_max = torch.clamp(block_max, min=eps, max=scale_ub.item())
    else:
        # pyre-ignore[6]: Incompatible parameter type [6]
        block_max = torch.clamp(block_max, min=eps)
    x_scale = torch.empty((B, H, grid_m), dtype=torch.float32, device=x.device)
    x_scale = max_fp8 / block_max.to(torch.float32)  # pyre-ignore
    # pyre-ignore[16]: Undefined attribute [16]
    x_scale[x_scale == float("inf")] = 1.0
    x_fp8 = (
        x_padded
        # pyre-ignore[16]: Undefined attribute [16]
        * x_scale.repeat_interleave(block_m, dim=2).unsqueeze(-1)
    )[:, :, :M, :]

    # Cast and move data to output device (for cpu weight loading).
    x_fp8 = x_fp8.to(device=x.device, dtype=pt_dtype)
    x_scale = x_scale.to(x.device)  # pyre-ignore
    del x, x_padded
    return x_fp8, 1.0 / x_scale  # pyre-ignore

def quantize_fp8_block_eager(
    x: torch.Tensor,
    block_m: int = 128,
    scale_ub: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with block-wise scalings and optionally move to output device.

    Scale per block i, j is computed as 1 / (MAX_FP8 / max(abs(x[i:i+block_m, j:j+block_k])))

    Args:
        x (Tensor): [M, K] higher precision input tensor.
        block_m (int): Block size for M dimension of scale.
        block_k (int): Block size for K dimension of scale.
        scale_ub: Maximum allowed value for scale.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.

    Returns:
        torch.Tensor: [M, K] fp8 scaled tensor.
        torch.Tensor: [cdiv(M, block_m), cdiv(K, block_k)] reciprocal scale tensor per block.
    """

    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()

    B, H, M, K = x.shape
    grid_m = triton.cdiv(M, block_m)

    # Pad x to multiple of block size.
    padded_m = grid_m * block_m
    x_padded = torch.zeros(B, H, padded_m, K, dtype=x.dtype, device=x.device)
    x_padded[:, :, :M, :] = x

    # Blockwise max.
    block_max = (
        x_padded.abs().reshape(B, H, grid_m, block_m, K).amax(dim=(-2, -1))
    )

    # Apply clamping.
    if scale_ub is not None:
        block_max = torch.clamp(block_max, min=eps, max=scale_ub.item())
    else:
        # pyre-ignore[6]: Incompatible parameter type [6]
        block_max = torch.clamp(block_max, min=eps)
    x_scale = torch.empty((B, H, grid_m), dtype=torch.float32, device=x.device)
    x_scale = max_fp8 / block_max.to(torch.float32)  # pyre-ignore
    # pyre-ignore[16]: Undefined attribute [16]
    x_scale[x_scale == float("inf")] = 1.0
    x_fp8 = (
        x_padded
        # pyre-ignore[16]: Undefined attribute [16]
        * x_scale.repeat_interleave(block_m, dim=2).unsqueeze(-1)
    )[:, :, :M, :]

    # Cast and move data to output device (for cpu weight loading).
    x_fp8 = x_fp8.to(device=x.device, dtype=pt_dtype)
    x_scale = x_scale.to(x.device)  # pyre-ignore
    del x, x_padded
    return x_fp8, 1.0 / x_scale  # pyre-ignore

def triton_quantize_fp8_head():
    pass

def quantize_fp8_per_head(
    a: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    use_triton: bool = False,
    output_device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to fp8 with per-head scalings (B, H) and optionally move to output device.

    Args:
        a (torch.Tensor): Input high precision tensor with shape (B, H, M, K).
        scale_ub (torch.Tensor): Maximum allowed value for scale.
        use_triton (bool): Whether to use triton kernel or pytorch.
        output_device (torch.device): Device to optionally move the scaled tensors to.

    Returns:
        torch.Tensor: fp8 scaled tensor.
        torch.Tensor: The reciprocal scale tensor per head (B, H).
    """
    a_shape = a.shape
    # Reshape to focus on head dimension (B, H)
    a_flat = a.view(a.size(0), a.size(1), -1)  # (B, H, M*K)

    if a.device == torch.device("cpu"):
        use_triton = False

    if use_triton:
        aq, a_scale = triton_quantize_fp8_head(a_flat, scale_ub)
        return aq.view(a_shape), a_scale

    if not output_device:
        output_device = a.device

    # Get constants.
    pt_dtype, _, max_fp8, eps = get_fp8_constants()
    
    # Compute max per head (B, H) across the flattened last dimensions (M*K)
    head_max: torch.Tensor = torch.max(torch.abs(a_flat), dim=-1)[0]

    # Apply clamping.
    if scale_ub is not None:
        head_max = torch.clamp(head_max, min=eps, max=scale_ub.item())
    else:
        head_max = torch.clamp(head_max, min=eps)

    # Compute scaling factors (B, H).
    a_scale = max_fp8 / head_max.to(torch.float32)
    a_scale[a_scale == float("inf")] = 1.0  # Avoid division by zero issues.

    # Apply scaling per head.
    a_fp8 = a_flat * a_scale[:, :, None]  # Apply scaling across the last dimensions.

    # Cast to desired precision and move to output device if necessary.
    a_fp8 = a_fp8.to(device=output_device, dtype=pt_dtype)
    a_scale = a_scale.to(output_device)

    # Reshape the output tensor to match the original shape.
    return a_fp8.view(a_shape), 1 / a_scale  # Reciprocal of scaling (B, H).
