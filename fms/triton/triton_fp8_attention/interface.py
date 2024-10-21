import torch

from .envs import USE_FP8_ATTENTION, USE_HDT, USE_SMOOTH_K
from .hadmard_transform import hadamard_transform_ref
from .quantization import (
    get_fp8_constants,
    quantize_fp8_block_eager,
    quantize_fp8_per_head,
    quantize_fp8_tensorwise_pt,
    # triton_quantize_fp8_block_v2,
    triton_quantize_fp8_row,
)
from .triton_flash_attention_QK_rowwise_V_blockwise import flash_QK_rowwise_V_blockwise
from .triton_flash_attention_QK_rowwise_V_per_head_tensorwise import (
    flash_QK_rowwise_V_per_head_tensorwise,
)
from .triton_flash_attention_QK_rowwise_V_tensorwise import (
    flash_QK_rowwise_V_tensorwise,
)
from .triton_flash_attention_QKV_tensorwise import flash_QKV_tensorwise

def fp8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    head_dim = q.shape[-1]
    ori_dtype = q.dtype

    if USE_SMOOTH_K:
        k -= k.mean(dim=-2, keepdim=True)

    if USE_HDT:
        q = hadamard_transform_ref(q, scale=1.0 / (head_dim**0.5))
        k = hadamard_transform_ref(k, scale=1.0 / (head_dim**0.5))
        v = v

    # collect_abs_percentiles(q, MAX_PERCENT_Q)
    # collect_abs_percentiles(k, MAX_PERCENT_K)

    # since QK is transformed, applying scale_ub to QK may cause accuracy regression
    scale_ub = torch.tensor([10.0], dtype=torch.float32, device=q.device)

    # 0 = no fp8 quantization, SDPA
    # 1 = QK rowwise, P blockwise, V tensorwise (full tensor)
    # 2 = QKV tensorwise
    # 3 = QKV direct cast
    # 4 = QK rowwise, V bf16
    # 5 = QK rowwise, P blockwise, V blockwise
    # 6 = OK rowwise, P blockwise, V tensorwise (per head)

    if USE_FP8_ATTENTION == 1:
        q_fp8, scale_q = triton_quantize_fp8_row(q)
        k_fp8, scale_k = triton_quantize_fp8_row(k)
        v_fp8, scale_v_tensor = quantize_fp8_tensorwise_pt(v)
        attn = flash_QK_rowwise_V_tensorwise(
            q_fp8,
            k_fp8,
            v_fp8,
            scale_q,
            scale_k,
            scale_v_tensor,
            output_dtype=ori_dtype,
        )
    elif USE_FP8_ATTENTION == 2:
        q_fp8, scale_q_tensor = quantize_fp8_tensorwise_pt(q)
        k_fp8, scale_k_tensor = quantize_fp8_tensorwise_pt(k)
        v_fp8, scale_v_tensor = quantize_fp8_tensorwise_pt(v)
        attn = flash_QKV_tensorwise(
            q_fp8,
            k_fp8,
            v_fp8,
            scale_q_tensor,
            scale_k_tensor,
            scale_v_tensor,
            output_dtype=ori_dtype,
        )
    elif USE_FP8_ATTENTION == 3:
        ptype, _, _, _ = get_fp8_constants()
        q_fp8 = q.to(ptype)
        k_fp8 = k.to(ptype)
        v_fp8 = v.to(ptype)
        qs = torch.tensor([1.0], dtype=torch.float32, device=q.device)
        attn = flash_QKV_tensorwise(
            q_fp8, k_fp8, v_fp8, qs, qs, qs, output_dtype=ori_dtype
        )
    elif USE_FP8_ATTENTION == 4:
        q_fp8, scale_q = triton_quantize_fp8_row(q)
        k_fp8, scale_k = triton_quantize_fp8_row(k)
        vs = torch.tensor([1.0], dtype=torch.float32, device=q.device)
        attn = flash_QK_rowwise_V_tensorwise(
            q_fp8, k_fp8, v, scale_q, scale_k, vs, output_dtype=ori_dtype
        )
    elif USE_FP8_ATTENTION == 5:
        BLOCK_N = 128
        q_fp8, scale_q = triton_quantize_fp8_row(q)
        k_fp8, scale_k = triton_quantize_fp8_row(k)
        v_fp8, scale_v = quantize_fp8_block_eager(v, block_m=BLOCK_N, scale_ub=scale_ub)

        B, H, M, K = v.shape
        v_ = (
            v_fp8.to(torch.float32)
            * scale_v.repeat_interleave(BLOCK_N, dim=-1)[:, :, :M].unsqueeze(-1)
        ).to(ori_dtype)
        scale_v_fake = torch.ones_like(scale_v)

        # Block-wise quantization for V still has numerical issues, so we're using fake quantization for V here.
        attn = flash_QK_rowwise_V_blockwise(
            q_fp8, k_fp8, v_, scale_q, scale_k, scale_v_fake, output_dtype=ori_dtype
        )
    elif USE_FP8_ATTENTION == 6:
        q_fp8, scale_q = triton_quantize_fp8_row(q)
        k_fp8, scale_k = triton_quantize_fp8_row(k)
        v_fp8, scale_v = quantize_fp8_per_head(v, scale_ub=scale_ub)

        attn = flash_QK_rowwise_V_per_head_tensorwise(
            q_fp8, k_fp8, v_fp8, scale_q, scale_k, scale_v, output_dtype=ori_dtype
        )
    
    return attn
