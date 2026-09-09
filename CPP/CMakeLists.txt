cmake_minimum_required(VERSION 3.18)
project(GPUProfiling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(gemm_bench main.cpp)

target_compile_options(gemm_bench PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(gemm_bench PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
