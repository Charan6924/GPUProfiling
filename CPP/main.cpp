#include <torch/torch.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA GPU required");

    torch::NoGradGuard no_grad;

    // enables TF32 Tensor Cores on Ampere+
    at::globalContext().setAllowTF32CuBLAS(true);

    const auto options = torch::TensorOptions()
        .device(torch::kCUDA, 0)
        .dtype(torch::kBFloat16); // Use Tensor Cores; use torch::kFloat for TF32

    const std::vector<int64_t> sizes = {256, 512, 1024, 2048, 4096, 8192};
    constexpr int warmup = 20;
    constexpr int iters = 100;

    std::cout << std::fixed << std::setprecision(3);

    for (int64_t n : sizes) {
        auto a = torch::randn({n, n}, options);
        auto b = torch::randn({n, n}, options);

        // warm up
        for (int i = 0; i < warmup; ++i) {
            auto c = torch::matmul(a, b);
        }
        torch::cuda::synchronize();

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) {
            auto c = torch::matmul(a, b);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float total_ms;
        cudaEventElapsedTime(&total_ms, start, end);

        float latency_ms = total_ms / iters;
        double tflops = (2.0 * n * n * n) / (latency_ms * 1e9);

        std::cout << "N=" << std::setw(5) << n
                  << " | latency=" << std::setw(8) << latency_ms << " ms"
                  << " | throughput=" << std::setw(8) << tflops << " TFLOP/s\n";

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
}


/*
N=  256 | latency=   0.006 ms | throughput=   5.363 TFLOP/s
N=  512 | latency=   0.006 ms | throughput=  41.339 TFLOP/s
N= 1024 | latency=   0.017 ms | throughput= 126.182 TFLOP/s
N= 2048 | latency=   0.110 ms | throughput= 156.009 TFLOP/s
N= 4096 | latency=   0.942 ms | throughput= 145.924 TFLOP/s
N= 8192 | latency=   6.767 ms | throughput= 162.471 TFLOP/s
*/
