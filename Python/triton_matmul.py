import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BM": 32,  "BN": 64,  "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64,  "BN": 64,  "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64,  "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 64,  "BK": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)

    # Group neighboring program IDs along M for better L2 reuse of B.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    
    # each thread computes a BM * BN sized block
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BK)):
        k_offsets = k_block * BK + offs_k

        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # BF16 Tensor Core MMA; accumulation remains FP32.
        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.bfloat16)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    grid = lambda meta: (
        triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


device = "cuda"
sizes = [256, 512, 1024, 2048, 4096, 8192]
warmup = 20
iters = 100

for n in sizes:
    a = torch.randn((n, n), device=device, dtype=torch.bfloat16)
    b = torch.randn((n, n), device=device, dtype=torch.bfloat16)

    # Autotuning and JIT compilation happen during warmup.
    for _ in range(warmup):
        c = triton_matmul(a, b)
    torch.cuda.synchronize()

    # Optional correctness check; excluded from timing.
    torch.testing.assert_close(
        c,
        torch.matmul(a, b),
        rtol=2e-2,
        atol=1.0,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        triton_matmul(a, b)
    end.record()
    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / iters
    tflops = (2 * n**3) / (latency_ms * 1e9)

    print(
        f"N={n:5d} | latency={latency_ms:8.3f} ms "
        f"| throughput={tflops:8.2f} TFLOP/s"
    )

'''
N=  256 | latency=   0.022 ms | throughput=    1.52 TFLOP/s
N=  512 | latency=   0.021 ms | throughput=   12.68 TFLOP/s
N= 1024 | latency=   0.021 ms | throughput=  101.36 TFLOP/s
N= 2048 | latency=   0.105 ms | throughput=  163.12 TFLOP/s
N= 4096 | latency=   0.809 ms | throughput=  169.78 TFLOP/s
N= 8192 | latency=   6.486 ms | throughput=  169.52 TFLOP/s
'''
