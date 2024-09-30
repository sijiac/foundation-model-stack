#!/usr/bin/env python
"""
Code copied from https://github.com/ROCm/triton/blob/triton-mlir/python/perf-kernels/flash-attention.py
"""

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team, AMD ML Frameworks Triton team

Features supported:

1) Fwd with causal masking
2) Any sequence lengths without padding (currently fwd kernel only)
3) Support for different sequence lengths for q and k
4) Nested tensor API currently does not support dropout or bias.

Not currently supported:

1) Non power of two head dims

"""

import argparse
import random
import sys
import torch

import triton
import triton.language as tl

torch_dtype:tl.constexpr = torch.float16

#TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2fnuz')
#if TORCH_HAS_FP8E5:
#    torch_dtype:tl.constexpr = torch.float8_e5m2fnuz
TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2')
if TORCH_HAS_FP8E5:
    torch_dtype:tl.constexpr = torch.float8_e5m2

class MetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    num_contexts = 0
    varlen = False
    dropout_p, return_encoded_softmax = 0.0, False

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range (0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i+1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i+1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_dropout(dropout_p, return_encoded_softmax):
        self.dropout_p = dropout_p
        self.return_encoded_softmax = return_encoded_softmax

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()
        if self.varlen:
            assert q.dim() == 3
            total_q, nheads_q, head_size = q.shape
            total_k, nheads_k, _ = k.shape
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias == None
            # TODO:Remove once dropout is supported with varlen
            assert self.dropout_p == 0.0
            assert not self.return_encoded_softmax
        else:
            assert q.dim() == 4
            batch, nheads_q, seqlen_q, head_size = q.shape
            _, nheads_k, seqlen_k, _ = k.shape
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0

@triton.jit
def cdiv_fn(x,y):
    return (x + y - 1) // y

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]

@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)

@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep

@triton.jit
def load_fn(block_ptr, first, second, pad):
    if first and second:
        tensor = tl.load(block_ptr, boundary_check=(0,1), padding_option=pad)
    elif first:
        tensor = tl.load(block_ptr, boundary_check=(0,), padding_option=pad)
    elif second:
        tensor = tl.load(block_ptr, boundary_check=(1,), padding_option=pad)
    else:
        tensor = tl.load(block_ptr)
    return tensor

@triton.jit
def load_scale_fn(block_ptr, first, pad):
    if True:
        tensor = tl.load(block_ptr, boundary_check=(0,), padding_option=pad)
    else:
        tensor = tl.load(block_ptr)
    return tensor

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, qs,
    K_block_ptr, Ks_block_ptr, V_block_ptr,
    vs,
    start_m,
    actual_seqlen_k,
    actual_seqlen_q,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    encoded_softmax_block_ptr,
    block_min, block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    bias_ptr,
    alibi_slope,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    MAX_FP8: tl.constexpr,
):
    # loop over k, v, and update accumulator
    for start_n in range (block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        k = load_fn(K_block_ptr, PADDED_HEAD, MASK_STEPS and (n_extra_tokens != 0), "zero")
        ks = load_scale_fn(Ks_block_ptr, MASK_STEPS and (n_extra_tokens != 0), "zero")
        ks = tl.where(ks < 1e-6, 1.0, ks)
        if PRE_LOAD_V:
            v = load_fn(V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None,:]
                mask = size_n < boundary_m[:,None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        # -- compute qk ----
        # breakpoint()
        # qk_scale = tl.where(qk_scale < 1e-4, 1.0, qk_scale)
        # qk_scale = tl.where(qk_scale >= 1.0 + 1e-4, 1.0, qk_scale)

        # tl.device_print("qk_scale", qk_scale)
        # qk_scale = tl.full([BLOCK_M, BLOCK_N], 1.0, dtype=tl.float32)
        qk += tl.dot(q, k)
        qk_scale = qs[:, None] * ks[None, :]
        qk *= qk_scale
        # tl.device_print("qk_scale", qk_scale)
        if bias_ptr is not None:
            bias = load_fn(bias_ptr, False, MASK_STEPS and (n_extra_tokens != 0), "zero")
            # While bias is added after multiplying qk with sm_scale,
            # our optimization to use 2^x instead of e^x results in an additional
            # scale factor of log2(e) which we must also multiply the bias with.
            qk += (bias * 1.44269504089)
           
        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m*BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)

            # Compute the relative position using the global positions
            relative_pos_block = global_m_positions[:,None] + actual_seqlen_k - global_n_positions[None,:] - actual_seqlen_q
            relative_pos_block = tl.abs(relative_pos_block)


            alibi_block = -1 * alibi_slope  * relative_pos_block

            qk += (alibi_block * 1.44269504089) # scale factor of log2(e)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk,1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        # p_max = tl.max(tl.abs(p))
        # p_scale = p_max / MAX_FP8
        # p_invs_scale = 1.0 / p_scale
        # # p_fp8 = (p * p_invs_scale)
        p_fp8 = p


        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                tl.store(encoded_softmax_block_ptr, tl.where(keep, p, -p).to(encoded_softmax_block_ptr.type.element_ty))
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_softmax_block_ptr, p.to(encoded_softmax_block_ptr.type.element_ty))
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero")
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p_fp8.to(V_block_ptr.type.element_ty), v) * vs
        # acc += tl.dot(p_fp8.to(V_block_ptr.type.element_ty), v) * p_scale * vs
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        Ks_block_ptr = tl.advance(Ks_block_ptr, (BLOCK_N,))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def attn_fwd(
    Q, K, V, Qs, Ks, Vs, bias, sm_scale, L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_qsz, stride_qsh, stride_qsm,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_ksz, stride_ksh, stride_ksn,
    stride_vz, stride_vh, stride_vk, stride_vn,
    # stride_vsz, stride_vsh, stride_vsk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_az, stride_ah,
    cu_seqlens_q, cu_seqlens_k,
    dropout_p, philox_seed, philox_offset_base, encoded_softmax,
    hq, hk,
    alibi_slopes,
    ACTUAL_BLOCK_DMODEL:tl.constexpr,
    MAX_SEQLENS_Q:tl.constexpr, MAX_SEQLENS_K:tl.constexpr,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    MAX_FP8: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if (IS_CAUSAL):
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.
        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = cdiv_fn(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q,
            BLOCK_N
        )
        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)
        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            o_offset = off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
            O_block_ptr = tl.make_block_ptr(
                base=Out + o_offset,
                shape=(seqlen_q, BLOCK_DMODEL),
                strides=(stride_om, stride_on),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_DMODEL),
                order=(1, 0)
            )
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
            # We still need to write 0s to the result
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,1))
            l_ptrs = L + off_z * hq * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
            # We store inf to LSE, not -inf because in the bwd pass, we subtract this
            # from qk which makes it -inf, such that exp(qk - inf) = 0 for these masked blocks.
            l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
            tl.store(l_ptrs, l)
            # TODO: Should dropout and return encoded softmax be handled here too?
            return

    is_mqa = hq != hk
    off_h_k = off_h_q % hk if is_mqa else off_h_q
    need_padding = False
    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        need_padding = True
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        need_padding = True
        n_extra_tokens = seqlen_k % BLOCK_N
    padded_head = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

    # Compute pointers for all the tensors used in this kernel.
    q_offset = off_z * stride_qz +  off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    qs_offset = off_z * stride_qsz +  off_h_q * stride_qsh + cu_seqlens_q_start * stride_qsm
    Qs_block_ptr = tl.make_block_ptr(
        base=Qs + qs_offset,
        shape=(seqlen_q,),
        strides=(stride_qsm,),
        offsets=(start_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,)
    )
    k_offset = off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(ACTUAL_BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    ks_offset = off_z * stride_ksz +  off_h_k * stride_ksh + cu_seqlens_k_start * stride_ksn
    Ks_block_ptr = tl.make_block_ptr(
        base=Ks + ks_offset,
        shape=(seqlen_k,),
        strides=(stride_ksn,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,)
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e4nv else (1, 0)
    v_offset = off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, ACTUAL_BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=v_order,
    )
    vs = tl.load(Vs)
    if BIAS_TYPE != 0:
        b_offset = off_h_q * stride_bh # Note: this might get large enough to overflow on some configs
        bias_ptr = tl.make_block_ptr(
            base=bias + b_offset,
            shape=(seqlen_q, seqlen_k),
            strides=(stride_bm, stride_bn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        bias_ptr = None

    if USE_ALIBI != 0:
        a_offset = off_z * stride_az +  off_h_q * stride_ah 
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    if ENABLE_DROPOUT:
        batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    # We can ask to return the dropout mask without actually doing any dropout. In
    # this case, we return an invalid pointer so indicate the mask is not valid.
    # TODO: Fix encoded softmax. It currently uses just h_q in the base offset.
    if RETURN_ENCODED_SOFTMAX:
        encoded_softmax_block_ptr = tl.make_block_ptr(
                base=encoded_softmax + off_h_q * seqlen_q * seqlen_k,
                shape=(seqlen_q, seqlen_k),
                strides=(seqlen_k, 1),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)
                )
    else:
        encoded_softmax_block_ptr = 0
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
    # have native e^x support in HW.
    qk_scale = sm_scale * 1.44269504089
    # Q is loaded once at the beginning and shared by all N blocks.
    q = load_fn(Q_block_ptr, True, padded_head, "zero")
    qs = load_scale_fn(Qs_block_ptr, True, "zero")
    qs = tl.where(qs < 1e-6, 1.0, qs)

    # start_m = tl.program_id(0)
    # off_h_q = tl.program_id(1)
    # off_z = tl.program_id(2)
    # if start_m == 0 and off_h_q == 0 and off_z == 0:
    # tl.device_print("qs", qs)

    # qs = tl.where(tl.arange(0, BLOCK_M) < seqlen_q - BLOCK_M * , qs, 1.0)
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, qs, K_block_ptr, Ks_block_ptr, V_block_ptr, vs,
            start_m, seqlen_k, seqlen_q,
            dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
            # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
            block_min, block_max, 0, 0, 0, bias_ptr, alibi_slope,
            # IS_CAUSAL, ....
            False, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V, False, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, padded_head, MAX_FP8
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    tl.debug_barrier()
    # Remaining blocks, if any, are full / not masked.
    if (masked_blocks > 0):
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        K_block_ptr = tl.advance(K_block_ptr, (0, n_full_blocks*BLOCK_N))
        Ks_block_ptr = tl.advance(Ks_block_ptr, (n_full_blocks*BLOCK_N,))
        V_block_ptr = tl.advance(V_block_ptr, (n_full_blocks*BLOCK_N, 0))
        if bias_ptr is not None:
            bias_ptr = tl.advance(bias_ptr, (0, n_full_blocks*BLOCK_N))
        if RETURN_ENCODED_SOFTMAX:
            encoded_softmax_block_ptr = tl.advance(encoded_softmax_block_ptr,
                                                   (0, n_full_blocks))
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, qs, K_block_ptr, Ks_block_ptr, V_block_ptr, vs,
            start_m, seqlen_k, seqlen_q,
            dropout_p, philox_seed, batch_philox_offset, encoded_softmax_block_ptr,
            block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, bias_ptr, alibi_slope,
            IS_CAUSAL, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V, True, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, padded_head, MAX_FP8
        )
    # epilogue
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    acc = acc.to(Out.type.element_ty)
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL,), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
    # write back LSE
    l_ptrs = L + off_z * hq * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last M block. For others, overflow_size will be -ve
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        # This is a > check because mask being 0 blocks the store.
        l_ptrs_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    # write back O
    o_offset = off_z * stride_oz + cu_seqlens_q_start * stride_om + off_h_q * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # Need boundary check on this to make sure the padding from the
    # Q and KV tensors in both dims are not part of what we store back.
    # TODO: Do the boundary check optionally.
    tl.store(O_block_ptr, acc, boundary_check=(0,1))



# def attention(q, k, v, sm_scale):

#     o = torch.empty_like(q, dtype=v.dtype)

#     batch, nheads_q, seqlen_q, head_size = q.shape
#     _, nheads_k, seqlen_k, _ = k.shape

#     max_seqlens_q = seqlen_q

#     q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
#     k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
#     v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
#     o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))

#     # Get closest power of 2 over or equal to 32.
#     unpadded_head_dims = {32, 64, 128, 256}
#     if head_size not in unpadded_head_dims:
#         padded_d_model = None
#         for i in unpadded_head_dims:
#             if i > head_size:
#                 padded_d_model = i
#                 break
#         assert padded_d_model is not None
#     else:
#         padded_d_model = head_size


#     # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'PRE_LOAD_V': False}, num_stages=1, num_warps=4)

#     BLOCK_M = 128
#     BLOCK_N = 128
#     PRE_LOAD_V = False
#     num_stages = 1
#     num_warps = 4

#     grid = (triton.cdiv(max_seqlens_q, BLOCK_M), nheads_q, batch)

#     # encoded_softmax is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
#     # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
#     # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
#     # only.  This return holds no useful output aside from debugging.

#     encoded_softmax = None

#     M = torch.empty((batch, nheads_q, max_seqlens_q), device=q.device, dtype=torch.float32)

#     # Seed the RNG so we get reproducible results for testing.
#     philox_seed = 0x1BF52
#     philox_offset = 0x1D4B42
    
#     bias_strides = (0,0,0,0)
#     alibi_strides = (0, 0)
    
#     max_fp8 = torch.finfo(torch.float8_e4m3fn).max

#     attn_fwd[grid](
#         q, k, v, None, sm_scale, M, o,
#         *q_strides, *k_strides, *v_strides, *o_strides, *bias_strides, *alibi_strides,
#         None, None,
#         BLOCK_M=BLOCK_M,
#         PRE_LOAD_V=PRE_LOAD_V,
#         BLOCK_N=BLOCK_N,
#         dropout_p=0.0,
#         philox_seed=philox_seed,
#         philox_offset_base=philox_offset,
#         encoded_softmax=encoded_softmax,
#         hq=nheads_q, hk=nheads_k,
#         alibi_slopes = None,
#         ACTUAL_BLOCK_DMODEL=head_size,
#         MAX_SEQLENS_Q=seqlen_q, 
#         MAX_SEQLENS_K=seqlen_k,
#         IS_CAUSAL=False, ########################
#         VARLEN=False,
#         BLOCK_DMODEL=padded_d_model,
#         BIAS_TYPE=0,
#         USE_ALIBI=0,
#         ENABLE_DROPOUT=False,
#         RETURN_ENCODED_SOFTMAX=False,
#         BATCH_SIZE= q.shape[0],
#         MAX_FP8=max_fp8,
#     )
#     return o


@torch.library.custom_op("triton::flash", mutates_args=())
def flash(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    qs: torch.Tensor, 
    ks: torch.Tensor, 
    vs: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:

    o = torch.empty_like(q, dtype=output_dtype)
    sm_scale = q.shape[-1] ** -0.5

    batch, nheads_q, seqlen_q, head_size = q.shape
    _, nheads_k, seqlen_k, _ = k.shape

    max_seqlens_q = seqlen_q

    q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
    qs_strides = (qs.stride(0), qs.stride(1), qs.stride(2))
    k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
    ks_strides = (ks.stride(0), ks.stride(1), ks.stride(2))
    v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
    # vs_strides = (vs.stride(0), vs.stride(1), vs.stride(2))
    o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))

    # Get closest power of 2 over or equal to 32.
    unpadded_head_dims = {32, 64, 128, 256}
    if head_size not in unpadded_head_dims:
        padded_d_model = None
        for i in unpadded_head_dims:
            if i > head_size:
                padded_d_model = i
                break
        assert padded_d_model is not None
    else:
        padded_d_model = head_size

    # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'PRE_LOAD_V': False}, num_stages=1, num_warps=4)
    BLOCK_M = 128
    BLOCK_N = 128
    PRE_LOAD_V = False

    grid = (triton.cdiv(max_seqlens_q, BLOCK_M), nheads_q, batch)

    # encoded_softmax is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only.  This return holds no useful output aside from debugging.
    
    encoded_softmax = None
    M = torch.empty((batch, nheads_q, max_seqlens_q), device=q.device, dtype=torch.float32)

    # Seed the RNG so we get reproducible results for testing.
    philox_seed = 0x1BF52
    philox_offset = 0x1D4B42
    
    bias_strides = (0,0,0,0)
    alibi_strides = (0, 0)

    max_fp8 = torch.finfo(torch.float8_e4m3fn).max

    attn_fwd[grid](
        q, k, v, qs, ks, vs, None, sm_scale, M, o,
        *q_strides, *qs_strides, *k_strides, *ks_strides, *v_strides, *o_strides, *bias_strides, *alibi_strides,
        None, None,
        BLOCK_M=BLOCK_M,
        PRE_LOAD_V=PRE_LOAD_V,
        BLOCK_N=BLOCK_N,
        dropout_p=0.0,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        encoded_softmax=encoded_softmax,
        hq=nheads_q, hk=nheads_k,
        alibi_slopes = None,
        ACTUAL_BLOCK_DMODEL=head_size,
        MAX_SEQLENS_Q=seqlen_q, 
        MAX_SEQLENS_K=seqlen_k,
        IS_CAUSAL=True, ########################
        VARLEN=False,
        BLOCK_DMODEL=padded_d_model,
        BIAS_TYPE=0,
        USE_ALIBI=0,
        ENABLE_DROPOUT=False,
        RETURN_ENCODED_SOFTMAX=False,
        BATCH_SIZE= q.shape[0],
        MAX_FP8=max_fp8,
    )
    return o

@flash.register_fake
def _(q, k, v):
    return torch.empty_like(q, dtype=v.dtype)
            

if __name__ == "__main__":

    b, nh, s, hd = 1, 32, 128, 128

    q = torch.randn(b, nh, s, hd, dtype=torch.float16, device='cuda')
    k = torch.randn(b, nh, s, hd, dtype=torch.float16, device='cuda')
    v = torch.randn(b, nh, s, hd, dtype=torch.float16, device='cuda')

    sm_scale = q.shape[-1] ** -0.5

    @torch.compile(fullgraph=True)
    def f(q, k, v):
        return flash(q, k, v)
    
    o = f(q, k, v)
