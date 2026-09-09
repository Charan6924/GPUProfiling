#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                          \
        if (error != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)         \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t status = (call);                                     \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            std::cerr << "cuBLASLt error: " << status                         \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

__global__ void fill_ones(__nv_bfloat16* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < count) {
        data[index] = __float2bfloat16(1.0f);
    }
}

int main() {
    const std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192};
    constexpr int warmup = 20;
    constexpr int iters = 100;
    constexpr size_t workspace_size = 64ULL * 1024 * 1024; // 64 MiB

    cublasLtHandle_t lt_handle;
    CUBLAS_CHECK(cublasLtCreate(&lt_handle));

    void* workspace;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::cout << std::fixed << std::setprecision(3);

    for (int N : sizes) {
        const size_t elements = static_cast<size_t>(N) * N;
        const size_t bytes = elements * sizeof(__nv_bfloat16);

        __nv_bfloat16 *A, *B, *C;
        CUDA_CHECK(cudaMalloc(&A, bytes));
        CUDA_CHECK(cudaMalloc(&B, bytes));
        CUDA_CHECK(cudaMalloc(&C, bytes));

        constexpr int fill_threads = 256;
        const int fill_blocks = (elements + fill_threads - 1) / fill_threads;

        fill_ones<<<fill_blocks, fill_threads>>>(A, elements);
        fill_ones<<<fill_blocks, fill_threads>>>(B, elements);
        CUDA_CHECK(cudaGetLastError());

        // FP32 accumulation, BF16 input/output.
        cublasLtMatmulDesc_t operation_desc;
        CUBLAS_CHECK(cublasLtMatmulDescCreate(
            &operation_desc,
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F
        ));

        cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &A_desc, CUDA_R_16BF, N, N, N
        ));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &B_desc, CUDA_R_16BF, N, N, N
        ));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &C_desc, CUDA_R_16BF, N, N, N
        ));

        // Our matrices are row-major, as in PyTorch.
        cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &row_major, sizeof(row_major)
        ));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &row_major, sizeof(row_major)
        ));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            C_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &row_major, sizeof(row_major)
        ));

        // Ask cuBLASLt to choose its best available GEMM algorithm.
        cublasLtMatmulPreference_t preference;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size,
            sizeof(workspace_size)
        ));

        cublasLtMatmulHeuristicResult_t heuristic;
        int algorithms_found = 0;

        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
            lt_handle,
            operation_desc,
            A_desc,
            B_desc,
            C_desc,
            C_desc,
            preference,
            1,
            &heuristic,
            &algorithms_found
        ));

        if (algorithms_found == 0) {
            std::cerr << "No cuBLASLt algorithm found for N=" << N << '\n';
            return EXIT_FAILURE;
        }

        // Warmup: cuBLASLt chooses and initializes the optimized GEMM path.
        for (int i = 0; i < warmup; ++i) {
            CUBLAS_CHECK(cublasLtMatmul(
                lt_handle,
                operation_desc,
                &alpha,
                A, A_desc,
                B, B_desc,
                &beta,
                C, C_desc,
                C, C_desc,
                &heuristic.algo,
                workspace,
                workspace_size,
                0
            ));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < iters; ++i) {
            CUBLAS_CHECK(cublasLtMatmul(
                lt_handle,
                operation_desc,
                &alpha,
                A, A_desc,
                B, B_desc,
                &beta,
                C, C_desc,
                C, C_desc,
                &heuristic.algo,
                workspace,
                workspace_size,
                0
            ));
        }

        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, end));

        const float latency_ms = total_ms / iters;
        const double tflops = (2.0 * N * N * N) / (latency_ms * 1e9);

        // A and B are all ones, so every C element should equal N.
        __nv_bfloat16 value;
        CUDA_CHECK(cudaMemcpy(
            &value,
            C + elements / 2,
            sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToHost
        ));

        const float result = __bfloat162float(value);
        const bool correct = std::abs(result - static_cast<float>(N)) < 1.0f;

        std::cout << "N=" << std::setw(5) << N
                  << " | latency=" << std::setw(8) << latency_ms << " ms"
                  << " | throughput=" << std::setw(8) << tflops << " TFLOP/s"
                  << " | check=" << (correct ? "PASS" : "FAIL")
                  << '\n';

        CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(A_desc));
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(B_desc));
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(C_desc));
        CUBLAS_CHECK(cublasLtMatmulDescDestroy(operation_desc));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
        CUDA_CHECK(cudaFree(A));
        CUDA_CHECK(cudaFree(B));
        CUDA_CHECK(cudaFree(C));
    }

    CUDA_CHECK(cudaFree(workspace));
    CUBLAS_CHECK(cublasLtDestroy(lt_handle));
}

/*
N=  256 | latency=   0.004 ms | throughput=   8.424 TFLOP/s | check=PASS
N=  512 | latency=   0.006 ms | throughput=  43.764 TFLOP/s | check=PASS
N= 1024 | latency=   0.017 ms | throughput= 127.953 TFLOP/s | check=PASS
N= 2048 | latency=   0.110 ms | throughput= 156.533 TFLOP/s | check=PASS
N= 4096 | latency=   0.937 ms | throughput= 146.718 TFLOP/s | check=PASS
N= 8192 | latency=   6.505 ms | throughput= 169.014 TFLOP/s | check=PASS
*/
