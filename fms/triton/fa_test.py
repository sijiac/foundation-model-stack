import abc
import functools
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed

from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule
from fms.triton.quantization import triton_quantize_fp8_row, get_fp8_constants, quantize_fp8_tensorwise_pt
from fms.triton.hadmard_transform import hadamard_transform_ref
from fms.triton.triton_flash_attention import flash_QK_rowwise_V_tensorwise as flash_amd_triton_kernel
from fms.triton.triton_linear import TritonLinear
from torch import nn, Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional as F
from triton.runtime.jit import reinterpret as tl_reinterpret, TensorWrapper  # @manual
import triton  # @manual

def main(length=35):
    SHAPE = [1, 32, length, 128]

    queries = torch.randn(SHAPE, dtype=torch.float16, device="cuda")
    keys_e = torch.randn(SHAPE, dtype=torch.float16, device="cuda")
    values_e = torch.randn(SHAPE, dtype=torch.float16, device="cuda")

    head_dim = queries.shape[-1]
    ori_dtype = queries.dtype

    # q_ht = hadamard_transform_ref(queries, scale=1.0 / (head_dim ** 0.5))
    # k_ht = hadamard_transform_ref(keys_e, scale=1.0 / (head_dim ** 0.5))

    q_ht = queries
    k_ht = keys_e
    
    q_fp8, scale_q = triton_quantize_fp8_row(q_ht)
    k_fp8, scale_k = triton_quantize_fp8_row(k_ht)
    v_fp8, scale_v_tensor = quantize_fp8_tensorwise_pt(values_e)

    # q_fp8 = convert_fp8_type(q_fp8, tl_dtype)
    # k_fp8 = convert_fp8_type(k_fp8, tl_dtype)

    q_dequant =  (q_fp8.to(torch.float32) * scale_q.unsqueeze(-1)).to(ori_dtype)
    k_dequant =  (k_fp8.to(torch.float32) * scale_k.unsqueeze(-1)).to(ori_dtype)
    v_dequant =  (v_fp8.to(torch.float32) * scale_v_tensor).to(ori_dtype)

    scale_q_fake = torch.ones_like(scale_q)
    scale_k_fake = torch.ones_like(scale_k)
    scale_v_fake = torch.ones_like(scale_v_tensor)

    # breakpoint()
    
    attn_ref = flash_amd_triton_kernel(q_dequant, k_dequant, v_dequant, scale_q_fake, scale_k_fake, scale_v_fake, ori_dtype)
    attn = flash_amd_triton_kernel(q_fp8, k_fp8, v_fp8, scale_q, scale_k, scale_v_tensor, ori_dtype)

    print(attn[0][0][0])
    print(attn_ref[0][0][0])
    print("----")
    print(torch.isnan(attn).any())
    print(attn[0][0][1])
    print(attn_ref[0][0][1])
    print("----")
    print(attn[0][0][2])
    print(attn_ref[0][0][2])
    breakpoint()


if __name__ == "__main__":
    main(35)
    main(100)
    main(200)
    main(300)
