#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                          \
        if (error != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)         \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

// Four warps compute four 16x16 tiles = one 32x32 output tile.
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;

__global__ void fill_ones(__nv_bfloat16* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < count) {
        data[index] = __float2bfloat16(1.0f);
    }
}

// A: row-major N x N BF16
// B_col: column-major N x N BF16
// C: row-major N x N FP32
__global__ void wmma_bf16_matmul(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B_col,
    float* __restrict__ C,
    int N
) {
    // A_shared[row][k]: row-major 32x16 tile.
    __shared__ __nv_bfloat16 A_shared[BLOCK_M][WMMA_K];

    // B_shared[col][k]: column-major logical 16x32 B tile.
    // This layout is required by matrix_b + wmma::col_major.
    __shared__ __nv_bfloat16 B_shared[BLOCK_N][WMMA_K];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    // Warp layout within this 32x32 output tile:
    // warp 0: top-left,  warp 1: top-right
    // warp 2: bottom-left, warp 3: bottom-right
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    wmma::fragment<
        wmma::matrix_a,
        WMMA_M, WMMA_N, WMMA_K,
        __nv_bfloat16,
        wmma::row_major
    > a_frag;

    wmma::fragment<
        wmma::matrix_b,
        WMMA_M, WMMA_N, WMMA_K,
        __nv_bfloat16,
        wmma::col_major
    > b_frag;

    wmma::fragment<
        wmma::accumulator,
        WMMA_M, WMMA_N, WMMA_K,
        float
    > acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k_start = 0; k_start < N; k_start += WMMA_K) {
        // Cooperatively stage A's 32x16 tile.
        for (int index = tid; index < BLOCK_M * WMMA_K; index += THREADS_PER_BLOCK) {
            const int row = index / WMMA_K;
            const int k = index % WMMA_K;

            A_shared[row][k] = A[(block_row + row) * N + k_start + k];
        }

        // Cooperatively stage B's 16x32 tile in column-major layout.
        for (int index = tid; index < BLOCK_N * WMMA_K; index += THREADS_PER_BLOCK) {
            const int col = index / WMMA_K;
            const int k = index % WMMA_K;

            B_shared[col][k] = B_col[(block_col + col) * N + k_start + k];
        }

        __syncthreads();

        // Each warp loads its 16x16 A and B tiles and executes Tensor Core MMA.
        wmma::load_matrix_sync(
            a_frag,
            &A_shared[warp_m * WMMA_M][0],
            WMMA_K
        );

        wmma::load_matrix_sync(
            b_frag,
            &B_shared[warp_n * WMMA_N][0],
            WMMA_K
        );

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        __syncthreads();
    }

    // Each warp writes its own 16x16 FP32 result tile.
    wmma::store_matrix_sync(
        C + (block_row + warp_m * WMMA_M) * N + block_col + warp_n * WMMA_N,
        acc_frag,
        N,
        wmma::mem_row_major
    );
}

int main() {
    const std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192};
    constexpr int warmup = 20;
    constexpr int iters = 100;

    std::cout << std::fixed << std::setprecision(3);

    for (int N : sizes) {
        // This version expects dimensions divisible by 32.
        if (N % BLOCK_M != 0 || N % BLOCK_N != 0) {
            std::cerr << "N must be divisible by 32\n";
            return EXIT_FAILURE;
        }

        const size_t elements = static_cast<size_t>(N) * N;
        const size_t bf16_bytes = elements * sizeof(__nv_bfloat16);
        const size_t fp32_bytes = elements * sizeof(float);

        __nv_bfloat16 *A, *B_col;
        float* C;

        CUDA_CHECK(cudaMalloc(&A, bf16_bytes));
        CUDA_CHECK(cudaMalloc(&B_col, bf16_bytes));
        CUDA_CHECK(cudaMalloc(&C, fp32_bytes));

        constexpr int fill_threads = 256;
        const int fill_blocks = (elements + fill_threads - 1) / fill_threads;

        fill_ones<<<fill_blocks, fill_threads>>>(A, elements);
        fill_ones<<<fill_blocks, fill_threads>>>(B_col, elements);
        CUDA_CHECK(cudaGetLastError());

        const dim3 block(THREADS_PER_BLOCK);
        const dim3 grid(N / BLOCK_N, N / BLOCK_M);

        for (int i = 0; i < warmup; ++i) {
            wmma_bf16_matmul<<<grid, block>>>(A, B_col, C, N);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < iters; ++i) {
            wmma_bf16_matmul<<<grid, block>>>(A, B_col, C, N);
        }

        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaGetLastError());

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, end));

        const float latency_ms = total_ms / iters;
        const double tflops = (2.0 * N * N * N) / (latency_ms * 1e9);

        float value;
        CUDA_CHECK(cudaMemcpy(
            &value,
            C + elements / 2,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        const bool correct = std::abs(value - static_cast<float>(N)) < 1.0f;

        std::cout << "N=" << std::setw(5) << N
                  << " | latency=" << std::setw(8) << latency_ms << " ms"
                  << " | throughput=" << std::setw(8) << tflops << " TFLOP/s"
                  << " | check=" << (correct ? "PASS" : "FAIL")
                  << '\n';

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
        CUDA_CHECK(cudaFree(A));
        CUDA_CHECK(cudaFree(B_col));
        CUDA_CHECK(cudaFree(C));
    }
}
