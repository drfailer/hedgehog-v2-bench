#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cublas_v2.h>

#include "graph.hpp"

template <class Type>
bool run_test(size_t n, size_t m, size_t p, size_t block_size, double tol,
              std::vector<int> const &device_ids) {
    constexpr Order Ord = Order::Column;
    printf("  test n=%zu m=%zu p=%zu bs=%zu gpus=%zu ... ",
           n, m, p, block_size, device_ids.size());

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<Type> dist(-5.0, 5.0);

    auto *data_a = new Type[n * m];
    auto *data_b = new Type[m * p];
    auto *data_c = new Type[n * p];
    auto *data_ref = new Type[n * p];

    for (size_t i = 0; i < n * m; ++i) data_a[i] = dist(rng);
    for (size_t i = 0; i < m * p; ++i) data_b[i] = dist(rng);
    for (size_t i = 0; i < n * p; ++i) data_c[i] = dist(rng);

    std::memcpy(data_ref, data_c, n * p * sizeof(Type));

    // Reference computation via cublas
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    Type alpha = 1, beta = 1;
    if constexpr (std::is_same_v<Type, float>) {
        float *d_a, *d_b, *d_ref;
        cudaMalloc(reinterpret_cast<void **>(&d_a), n * m * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&d_b), m * p * sizeof(float));
        cudaMalloc(reinterpret_cast<void **>(&d_ref), n * p * sizeof(float));
        cudaMemcpy(d_a, data_a, n * m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, data_b, m * p * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ref, data_ref, n * p * sizeof(float), cudaMemcpyHostToDevice);
        cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, p, m, &alpha, d_a, n, d_b, m, &beta, d_ref, n);
        cudaMemcpy(data_ref, d_ref, n * p * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_ref);
    } else {
        double *d_a, *d_b, *d_ref;
        cudaMalloc(reinterpret_cast<void **>(&d_a), n * m * sizeof(double));
        cudaMalloc(reinterpret_cast<void **>(&d_b), m * p * sizeof(double));
        cudaMalloc(reinterpret_cast<void **>(&d_ref), n * p * sizeof(double));
        cudaMemcpy(d_a, data_a, n * m * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, data_b, m * p * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ref, data_ref, n * p * sizeof(double), cudaMemcpyHostToDevice);
        cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, p, m, &alpha, d_a, n, d_b, m, &beta, d_ref, n);
        cudaMemcpy(data_ref, d_ref, n * p * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_ref);
    }
    cublasDestroy_v2(handle);

    auto mat_a = std::make_shared<MatrixData<Type, 'a', Ord>>(n, m, block_size, data_a);
    auto mat_b = std::make_shared<MatrixData<Type, 'b', Ord>>(m, p, block_size, data_b);
    auto mat_c = std::make_shared<MatrixData<Type, 'c', Ord>>(n, p, block_size, data_c);

    auto graph = make_gpu_pipeline_dgemm_graph<Type, Ord>(
        n, m, p, block_size, 2, 2, device_ids);

    graph->start();
    graph->push_data(mat_a);
    graph->push_data(mat_b);
    graph->push_data(mat_c);

    auto result = std::get<0>(graph->get_result());
    graph->stop();

    double max_err = 0.0;
    for (size_t i = 0; i < n * p; ++i) {
        double err = std::abs(static_cast<double>(data_c[i]) - static_cast<double>(data_ref[i]));
        if (err > max_err) max_err = err;
    }

    bool pass = max_err < tol;
    printf("%s (max_err=%.2e)\n", pass ? "PASS" : "FAIL", max_err);

    delete[] data_a;
    delete[] data_b;
    delete[] data_c;
    delete[] data_ref;

    return pass;
}

int main() {
    std::vector<int> device_ids = {0};

    int count = 0;
    cudaGetDeviceCount(&count);
    if (count > 1) {
        device_ids.resize(count);
        std::iota(device_ids.begin(), device_ids.end(), 0);
    }

    int failures = 0;

    printf("GPU Pipeline DGEMM tests (Column-major, float, %zu GPU(s)):\n", device_ids.size());
    if (!run_test<float>(16, 16, 16, 4, 1e-3, device_ids)) ++failures;
    if (!run_test<float>(16, 16, 16, 16, 1e-3, device_ids)) ++failures;
    if (!run_test<float>(20, 12, 8, 4, 1e-3, device_ids)) ++failures;
    if (!run_test<float>(64, 64, 64, 16, 1e-3, device_ids)) ++failures;

    printf("GPU Pipeline DGEMM tests (Column-major, double, %zu GPU(s)):\n", device_ids.size());
    if (!run_test<double>(16, 16, 16, 4, 1e-10, device_ids)) ++failures;
    if (!run_test<double>(64, 64, 64, 16, 1e-10, device_ids)) ++failures;

    printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
