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

'''
N=  512 | latency=   0.068 ms | throughput=    3.95 TFLOP/s
N= 1024 | latency=   0.063 ms | throughput=   34.34 TFLOP/s
N= 2048 | latency=   0.114 ms | throughput=  150.86 TFLOP/s
N= 4096 | latency=   1.036 ms | throughput=  132.72 TFLOP/s
N= 8192 | latency=   7.302 ms | throughput=  150.57 TFLOP/s
'''
