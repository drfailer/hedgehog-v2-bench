#include <hedgehog/hedgehog.h>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "data/matrix_data.h"
#include "data/matrix_block_data.h"

#include "task/addition_task.h"
#include "task/matrix_row_traversal_task.h"
#include "task/matrix_column_traversal_task.h"

#include "state/partial_computation_state.h"
#include "state/partial_computation_state_manager.h"

#include "graph/cuda_computation_graph.h"
#include "execution_pipeline/multi_gpu_exec_pipeline.h"

int main(int argc, char **argv) {
    using MatrixType = float;
    constexpr Order Ord = Order::Column;

    size_t n = 1024;
    size_t blockSize = 256;
    size_t numberThreadProduct = 3;
    size_t numberThreadAddition = 3;
    int numGPUs = 1;

    if (argc > 1) n = std::atol(argv[1]);
    if (argc > 2) blockSize = std::atol(argv[2]);
    if (argc > 3) numberThreadProduct = std::atol(argv[3]);
    if (argc > 4) numberThreadAddition = std::atol(argv[4]);
    if (argc > 5) numGPUs = std::atoi(argv[5]);

    std::vector<int> deviceIds;
    if (numGPUs <= 0) {
        int numberGPUSAvailable = 0;
        cudaGetDeviceCount(&numberGPUSAvailable);
        deviceIds.resize(numberGPUSAvailable);
        std::iota(deviceIds.begin(), deviceIds.end(), 0);
    } else {
        deviceIds.resize(numGPUs);
        std::iota(deviceIds.begin(), deviceIds.end(), 0);
    }

    printf("hedgehog-v1 GPU Pipeline DGEMM: n=%zu bs=%zu prod_threads=%zu add_threads=%zu gpus=%zu\n",
           n, blockSize, numberThreadProduct, numberThreadAddition, deviceIds.size());

    if (blockSize == 0) blockSize = 1;

    uint64_t seed = 42;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<MatrixType> dist(0, 10);

    auto *dataA = new MatrixType[n * n];
    auto *dataB = new MatrixType[n * n];
    auto *dataC = new MatrixType[n * n];

    for (size_t i = 0; i < n * n; ++i) dataA[i] = dist(rng);
    for (size_t i = 0; i < n * n; ++i) dataB[i] = dist(rng);
    for (size_t i = 0; i < n * n; ++i) dataC[i] = dist(rng);

    auto matrixA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(n, n, blockSize, dataA);
    auto matrixB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(n, n, blockSize, dataB);
    auto matrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(n, n, blockSize, dataC);

    size_t nBlocks = std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1);
    size_t mBlocks = nBlocks;
    size_t pBlocks = nBlocks;

    printf("blocks: %zu x %zu x %zu\n", nBlocks, mBlocks, pBlocks);

    // GPU inner graph
    auto cudaGraph = std::make_shared<CUDAComputationGraph<MatrixType>>(
        n, n, n, blockSize, numberThreadProduct);

    // Execution pipeline
    auto executionPipeline =
        std::make_shared<MultiGPUExecPipeline<MatrixType>>(cudaGraph, deviceIds);

    // CPU tasks
    auto taskTraversalA =
        std::make_shared<MatrixColumnTraversalTask<MatrixType, 'a', Ord>>();
    auto taskTraversalB =
        std::make_shared<MatrixRowTraversalTask<MatrixType, 'b', Ord>>();
    auto taskTraversalC =
        std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Ord>>();
    auto additionTask =
        std::make_shared<AdditionTask<MatrixType, Ord>>(numberThreadAddition);

    // State
    auto statePartialComputation =
        std::make_shared<PartialComputationState<MatrixType, Ord>>(
            nBlocks, pBlocks, nBlocks * mBlocks * pBlocks + 1);
    auto stateManagerPartialComputation =
        std::make_shared<PartialComputationStateManager<MatrixType, Ord>>(statePartialComputation);

    // Build outer graph
    hh::Graph<3,
        MatrixData<MatrixType, 'a', Ord>, MatrixData<MatrixType, 'b', Ord>, MatrixData<MatrixType, 'c', Ord>,
        MatrixBlockData<MatrixType, 'c', Ord>>
        graph("GPU Pipeline DGEMM v1");

    graph.inputs(taskTraversalA);
    graph.inputs(taskTraversalB);
    graph.inputs(taskTraversalC);

    graph.edges(taskTraversalA, executionPipeline);
    graph.edges(taskTraversalB, executionPipeline);

    graph.edges(executionPipeline, stateManagerPartialComputation);

    graph.edges(taskTraversalC, stateManagerPartialComputation);
    graph.edges(stateManagerPartialComputation, additionTask);
    graph.edges(additionTask, stateManagerPartialComputation);
    graph.outputs(stateManagerPartialComputation);

    graph.executeGraph();

    auto t0 = std::chrono::high_resolution_clock::now();

    graph.pushData(matrixA);
    graph.pushData(matrixB);
    graph.pushData(matrixC);
    graph.finishPushingData();

    graph.waitForTermination();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double gflops = 2.0 * n * n * n / elapsed / 1e9;
    printf("time: %.3f s  |  %.2f GFLOPS\n", elapsed, gflops);

    graph.createDotFile("v1-gpu-pipeline-dgemm.dot",
                        hh::ColorScheme::EXECUTION, hh::StructureOptions::ALL);

    delete[] dataA;
    delete[] dataB;
    delete[] dataC;

    return 0;
}
