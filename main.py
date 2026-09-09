import torch
torch.set_float32_matmul_precision("high")
device = 'cuda'
dtype = torch.bfloat16

def matmul(a,b):
    return torch.matmul(a,b)

compiled_matmul = torch.compile(matmul, mode="max-autotune", fullgraph = True)

sizes = [256, 512, 1024, 2048, 4096, 8192]
warmup = 20
iters = 100

for n in sizes:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    for _ in range(warmup):
        compiled_matmul(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        compiled_matmul(a, b)
    end.record()
    torch.cuda.synchronize()

    latency_ms = start.elapsed_time(end) / iters
    tflops = (2 * n**3) / (latency_ms * 1e9)

    print(f"N={n:5d} | latency={latency_ms:8.3f} ms | throughput={tflops:8.2f} TFLOP/s")
