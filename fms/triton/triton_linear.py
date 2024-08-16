import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from torch import nn



@triton.jit()
def column_major(pid,
              m, n,
              block_m: tl.constexpr, block_n: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m) 

    pid_m = pid % grid_m
    pid_n = pid // grid_m

    return pid_m, pid_n


@triton.jit
def gemm_split_k_kernel(a_ptr, b_ptr, c_ptr,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            m, n, k,
            block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
            split_k: tl.constexpr):
    
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k*split_k)

    pid_m, pid_n = column_major(pid,
                                m, n,
                                block_m, block_n)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    offs_k = pid_k*block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)


    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, grid_k):
        
        k_remaining = k - k_ * (block_k * split_k)

        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk
    
    
    acc.to(tl.float16)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]
    
    tl.atomic_add(c_ptrs, acc, mask=mask)


def get_config(m, n, k):

    # Granite-8B and Llama-8B Configs (Decoding)
    # fused_qkv_proj
    if (n == 6144 and k == 4096):
        block_m = 16
        block_n = 256
        block_k = 32
        split_k = 8
        num_warps = 4
        num_stages = 4

    # o_proj
    elif (n == 4096 and k == 4096):
        block_m = 16
        block_n = 512
        block_k = 64
        split_k = 32
        num_warps = 16
        num_stages = 3
        
    # fused gate_up_proj
    elif (n == 28672 and k == 4096):
        block_m = 16
        block_n = 256
        block_k = 64
        split_k = 8
        num_warps = 16
        num_stages = 3
        
    # down_proj
    elif (n == 4096 and k == 14336):
        block_m = 16
        block_n = 64
        block_k = 32
        split_k = 32
        num_warps = 8
        num_stages = 5

    # Default
    else:
        block_m = 16
        block_n = 64
        block_k = 32
        split_k = 16
        num_warps = 4
        num_stages = 4

    return block_m, block_n, block_k, split_k, num_warps, num_stages

def matmul(a, b):
    

    assert a.shape[1] == b.shape[0]

    m, k = a.shape
    _, n = b.shape

    # print(f"{m=}, {n=}, {k=}")
    block_m, block_n, block_k, split_k, num_warps, num_stages = get_config(m, n, k)

 
    # config = {
    #     "block_m" : block_m,
    #     "block_n" : block_n,
    #     "block_k" : block_k,
    #     "split_k" : split_k,
    #     "num_warps" : num_warps,
    #     "num_stages" : num_stages,
    # }

    # print(f"{config=}")

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k
    
    grid = (total_programs_mn, total_programs_k)
    
    c = torch.zeros((m, n), device=a.device, dtype=torch.float16)
    k = gemm_split_k_kernel[grid](a, b, c,
                            a.stride(0), a.stride(1),
                            b.stride(0), b.stride(1),
                            c.stride(0), c.stride(1),                        
                            m, n, k,
                            block_m, block_n, block_k,
                            split_k, num_warps=num_warps, num_stages=num_stages)

    return c


# @torch.library.custom_op("triton::matmul", mutates_args=())
# def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

#     assert a.shape[1] == b.shape[0]
#     m, k = a.shape
#     _, n = b.shape
    
#     block_m = 64
#     block_n = 64
#     block_k = 64
#     num_stages = 3
#     num_warps = 8
#     split_k = 4
#     group_m = 8

#     total_blocks_m = triton.cdiv(m, block_m)
#     total_blocks_n = triton.cdiv(n, block_n)
#     total_programs_mn = total_blocks_m * total_blocks_n
#     total_programs_k = split_k
    
#     grid = (total_programs_mn, total_programs_k)
    
#     c = torch.zeros((m, n), device=a.device, dtype=torch.float16)
#     k = gemm_split_k_kernel[grid](a, b, c,
#                             a.stride(0), a.stride(1),
#                             b.stride(0), b.stride(1),
#                             c.stride(0), c.stride(1),                        
#                             m, n, k,
#                             block_m, block_n, block_k,
#                             split_k, group_m, num_stages=num_stages, num_warps=num_warps)
    
#     return c 

# @matmul.register_fake
# def _(a, b):
#     m = a.shape[0]
#     n = b.shape[0]
#     return torch.zeros((m, n), device=a.device, dtype=torch.float16)

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super(TritonLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16, device='cuda'))

        # Granite
        # if bias:
        # self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16, device='cuda'))


    def forward(self, input):
        ishape= list(input.shape)
        if len(ishape) == 3:
            input = input.view(-1,ishape[-1])

        y = matmul(input, self.weight.T)

        # Granite
        # if self.bias is not None:
        #     y = y + self.bias

        if len(ishape) == 3:            
            y = y.view(ishape[0],ishape[1],-1)

        return y
    
       

if __name__ == '__main__':

    m, k, n = 512, 512, 512
    
    a = torch.randn(m, k, dtype=torch.float16, device='cuda')
    b = torch.randn(k, n, dtype=torch.float16, device='cuda')

    @torch.compile(fullgraph=True)
    def f(a, b):
        return matmul(a, b)

    c = f(a, b)

    c2 = torch.matmul(a, b)

    print(c)
    print(c2)


