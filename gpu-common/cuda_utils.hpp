#ifndef GPU_COMMON_CUDA_UTILS_HPP
#define GPU_COMMON_CUDA_UTILS_HPP

#include <iostream>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

inline void check_cuda(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " — " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

inline void check_cuda(cublasStatus_t err, const char *file, int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error at " << file << ":" << line
                  << " — status " << err << std::endl;
        std::exit(1);
    }
}

#define CUDA_CHECK(x) check_cuda((x), __FILE__, __LINE__)

#endif
