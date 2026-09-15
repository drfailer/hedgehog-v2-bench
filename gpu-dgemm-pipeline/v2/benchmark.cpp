#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "graph.hpp"

int main(int argc, char **argv) {
    using Type = float;
    constexpr Order Ord = Order::Column;

    size_t n = 1024;
    size_t block_size = 256;
    size_t product_threads = 3;
    size_t addition_threads = 3;
    int num_gpus = 0;

    if (argc > 1) n = std::atol(argv[1]);
    if (argc > 2) block_size = std::atol(argv[2]);
    if (argc > 3) product_threads = std::atol(argv[3]);
    if (argc > 4) addition_threads = std::atol(argv[4]);
    if (argc > 5) num_gpus = std::atoi(argv[5]);

    std::vector<int> device_ids;
    if (num_gpus <= 0) {
        int count = 0;
        cudaGetDeviceCount(&count);
        if (count <= 0) count = 1;
        device_ids.resize(count);
        std::iota(device_ids.begin(), device_ids.end(), 0);
    } else {
        device_ids.resize(num_gpus);
        std::iota(device_ids.begin(), device_ids.end(), 0);
    }

    printf("hedgehog-v2 GPU Pipeline DGEMM: n=%zu bs=%zu prod_threads=%zu add_threads=%zu gpus=%zu\n",
           n, block_size, product_threads, addition_threads, device_ids.size());

    uint64_t seed = 42;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Type> dist(0, 10);

    auto *data_a = new Type[n * n];
    auto *data_b = new Type[n * n];
    auto *data_c = new Type[n * n];

    for (size_t i = 0; i < n * n; ++i) data_a[i] = dist(rng);
    for (size_t i = 0; i < n * n; ++i) data_b[i] = dist(rng);
    for (size_t i = 0; i < n * n; ++i) data_c[i] = dist(rng);

    auto mat_a = std::make_shared<MatrixData<Type, 'a', Ord>>(n, n, block_size, data_a);
    auto mat_b = std::make_shared<MatrixData<Type, 'b', Ord>>(n, n, block_size, data_b);
    auto mat_c = std::make_shared<MatrixData<Type, 'c', Ord>>(n, n, block_size, data_c);

    auto graph = make_gpu_pipeline_dgemm_graph<Type, Ord>(
        n, n, n, block_size, product_threads, addition_threads, device_ids);

    auto t0 = std::chrono::high_resolution_clock::now();

    graph->start();
    graph->push_data(mat_a);
    graph->push_data(mat_b);
    graph->push_data(mat_c);

    auto result = std::get<0>(graph->get_result());
    graph->stop();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double gflops = 2.0 * n * n * n / elapsed / 1e9;
    printf("time: %.3f s  |  %.2f GFLOPS\n", elapsed, gflops);
    graph->generate_dot_file("v2-gpu-pipeline-dgemm.dot");

    delete[] data_a;
    delete[] data_b;
    delete[] data_c;

    return 0;
}
