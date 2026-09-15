#include <hedgehog/hedgehog.h>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "data/matrix_data.h"
#include "data/matrix_block_data.h"
#include "data/cuda_matrix_block_data.h"

#include "task/addition_task.h"
#include "task/matrix_row_traversal_task.h"
#include "task/matrix_column_traversal_task.h"

#include "cuda_tasks/cuda_copy_in_gpu.h"
#include "cuda_tasks/cuda_copy_out_gpu.h"
#include "cuda_tasks/cuda_product_task.h"

#include "state/cuda_input_block_state.h"
#include "state/partial_computation_state.h"
#include "state/partial_computation_state_manager.h"

int main(int argc, char **argv) {
    using MatrixType = float;
    constexpr Order Ord = Order::Column;

    size_t n = 1024;
    size_t blockSize = 256;
    size_t numberThreadProduct = 3;
    size_t numberThreadAddition = 3;

    if (argc > 1) n = std::atol(argv[1]);
    if (argc > 2) blockSize = std::atol(argv[2]);
    if (argc > 3) numberThreadProduct = std::atol(argv[3]);
    if (argc > 4) numberThreadAddition = std::atol(argv[4]);

    if (blockSize == 0) blockSize = 1;

    printf("hedgehog-v1 GPU DGEMM: n=%zu bs=%zu prod_threads=%zu add_threads=%zu\n",
           n, blockSize, numberThreadProduct, numberThreadAddition);

    uint64_t seed = 42;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<MatrixType> dist(0, 10);

    auto *dataA = new MatrixType[n * n]();
    auto *dataB = new MatrixType[n * n]();
    auto *dataC = new MatrixType[n * n]();

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

    // CUDA tasks
    auto copyInATask = std::make_shared<CudaCopyInGpu<MatrixType, 'a'>>(pBlocks, blockSize, n);
    auto copyInBTask = std::make_shared<CudaCopyInGpu<MatrixType, 'b'>>(nBlocks, blockSize, n);
    auto productTask = std::make_shared<CudaProductTask<MatrixType>>(n, numberThreadProduct);
    auto copyOutTask = std::make_shared<CudaCopyOutGpu<MatrixType>>(blockSize);

    // Memory managers
    auto cudaMemoryManagerA =
        std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'a'>, size_t>>(nBlocks + 4, blockSize);
    auto cudaMemoryManagerB =
        std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'b'>, size_t>>(pBlocks + 4, blockSize);
    auto cudaMemoryManagerProduct =
        std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'p'>, size_t>>(8, blockSize);

    copyInATask->connectMemoryManager(cudaMemoryManagerA);
    copyInBTask->connectMemoryManager(cudaMemoryManagerB);
    productTask->connectMemoryManager(cudaMemoryManagerProduct);

    // CPU tasks
    auto taskTraversalA = std::make_shared<MatrixColumnTraversalTask<MatrixType, 'a', Ord>>();
    auto taskTraversalB = std::make_shared<MatrixRowTraversalTask<MatrixType, 'b', Ord>>();
    auto taskTraversalC = std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Ord>>();
    auto additionTask = std::make_shared<AdditionTask<MatrixType, Ord>>(numberThreadAddition);

    // States
    auto stateInputBlock =
        std::make_shared<CudaInputBlockState<MatrixType>>(nBlocks, mBlocks, pBlocks);
    auto statePartialComputation =
        std::make_shared<PartialComputationState<MatrixType, Ord>>(
            nBlocks, pBlocks, nBlocks * mBlocks * pBlocks + 1);

    // State managers
    auto stateManagerInputBlock =
        std::make_shared<hh::StateManager<
            2,
            CudaMatrixBlockData<MatrixType, 'a'>, CudaMatrixBlockData<MatrixType, 'b'>,
            std::pair<std::shared_ptr<CudaMatrixBlockData<MatrixType, 'a'>>,
                      std::shared_ptr<CudaMatrixBlockData<MatrixType, 'b'>>>
        >>(stateInputBlock, "Input State Manager");
    auto stateManagerPartialComputation =
        std::make_shared<PartialComputationStateManager<MatrixType, Ord>>(statePartialComputation);

    // Build graph
    hh::Graph<3,
        MatrixData<MatrixType, 'a', Ord>, MatrixData<MatrixType, 'b', Ord>, MatrixData<MatrixType, 'c', Ord>,
        MatrixBlockData<MatrixType, 'c', Ord>>
        graph("GPU DGEMM v1");

    graph.inputs(taskTraversalA);
    graph.inputs(taskTraversalB);
    graph.inputs(taskTraversalC);

    graph.edges(taskTraversalA, copyInATask);
    graph.edges(taskTraversalB, copyInBTask);

    graph.edges(copyInATask, stateManagerInputBlock);
    graph.edges(copyInBTask, stateManagerInputBlock);

    graph.edges(stateManagerInputBlock, productTask);

    graph.edges(productTask, copyOutTask);
    graph.edges(copyOutTask, stateManagerPartialComputation);

    graph.edges(taskTraversalC, stateManagerPartialComputation);
    graph.edges(stateManagerPartialComputation, additionTask);
    graph.edges(additionTask, stateManagerPartialComputation);
    graph.outputs(stateManagerPartialComputation);

    // Execute
    graph.executeGraph();

    auto t0 = std::chrono::high_resolution_clock::now();

    graph.pushData(matrixA);
    graph.pushData(matrixB);
    graph.pushData(matrixC);
    graph.finishPushingData();

    auto _ = graph.getBlockingResult();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double gflops = 2.0 * n * n * n / elapsed / 1e9;
    printf("time: %.3f s  |  %.2f GFLOPS\n", elapsed, gflops);

    graph.waitForTermination();

    graph.createDotFile("v1-gpu-dgemm.dot", hh::ColorScheme::EXECUTION, hh::StructureOptions::NONE);

    delete[] dataA;
    delete[] dataB;
    delete[] dataC;

    return 0;
}
