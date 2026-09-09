import torch
import triton
import triton.langauge as tl

@triton.autotune(
    configs=[
        triton.Config({"BM": 32,  "BN": 64,  "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64,  "BN": 64,  "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64,  "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 64,  "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],)

device = 'cuda'
sizes = [256, 512, 1024, 2048, 4096, 8192]
warmup = 20
iters = 100

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M:tl.constexpr, N:tl.constexpr, K:tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bm: tl.constexpr, stride_bk: tl.constexpr,
    stride_cm: tl.constexpr, stride_ck: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis = 0)

    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)

    



