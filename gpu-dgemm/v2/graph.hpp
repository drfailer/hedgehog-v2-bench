#ifndef GPU_DGEMM_V2_GRAPH_HPP
#define GPU_DGEMM_V2_GRAPH_HPP

#include "../../gpu-common/gpu_tasks.hpp"

template <class Type, Order Ord>
auto make_gpu_dgemm_graph(size_t n, size_t m, size_t p, size_t block_size,
                          size_t product_threads, size_t addition_threads) {

    size_t n_blocks = (n + block_size - 1) / block_size;
    size_t m_blocks = (m + block_size - 1) / block_size;
    size_t p_blocks = (p + block_size - 1) / block_size;

    auto pool_a = std::make_shared<hh::Pool<CudaMatrixBlockData<Type, 'a'>>>();
    pool_a->fill(n_blocks + product_threads + 4, block_size);
    auto pool_b = std::make_shared<hh::Pool<CudaMatrixBlockData<Type, 'b'>>>();
    pool_b->fill(p_blocks + product_threads + 4, block_size);
    auto pool_p = std::make_shared<hh::Pool<CudaMatrixBlockData<Type, 'p'>>>();
    pool_p->fill(product_threads + 4, block_size);

    auto traversal_a = hh::make_task<ColumnTraversalTask<Type, 'a', Ord>>(1, "TraversalA");
    auto traversal_b = hh::make_task<RowTraversalTask<Type, 'b', Ord>>(1, "TraversalB");
    auto traversal_c = hh::make_task<RowTraversalTask<Type, 'c', Ord>>(1, "TraversalC");

    auto copy_in_a = hh::make_task(
        std::make_shared<CudaCopyInTask<Type, 'a'>>(block_size, n, pool_a), 1, "CopyInA");
    auto copy_in_b = hh::make_task(
        std::make_shared<CudaCopyInTask<Type, 'b'>>(block_size, m, pool_b), 1, "CopyInB");

    auto matcher = hh::make_state_manager(
        std::make_shared<CudaInputMatcherTask<Type>>(n_blocks, m_blocks, p_blocks),
        "InputMatcher");

    auto product = hh::make_task(
        std::make_shared<CudaProductTask<Type>>(pool_p), product_threads, "CudaProduct");

    auto copy_out = hh::make_task<CudaCopyOutTask<Type>>(1, "CopyOut");

    auto partial = hh::make_task(
        std::make_shared<PartialComputationTask<Type, Ord>>(n_blocks, p_blocks, m_blocks),
        1, "PartialComputation");

    auto addition = hh::make_task<AdditionTask<Type, Ord>>(addition_threads, "Addition");

    auto graph = hh::make_graph<3,
        MatrixData<Type, 'a', Ord>, MatrixData<Type, 'b', Ord>, MatrixData<Type, 'c', Ord>,
        MatrixData<Type, 'c', Ord>>("GPU DGEMM");

    graph->connect_inputs(traversal_a);
    graph->connect_inputs(traversal_b);
    graph->connect_inputs(traversal_c);
    graph->connect_inputs(partial);

    graph->draw_edges(traversal_a, copy_in_a);
    graph->draw_edges(traversal_b, copy_in_b);

    graph->draw_edges(copy_in_a, matcher);
    graph->draw_edges(copy_in_b, matcher);

    graph->draw_edges(matcher, product);

    graph->draw_edges(product, copy_out);
    graph->draw_edges(copy_out, partial);

    graph->draw_edges(traversal_c, partial);
    graph->draw_edges(partial, addition);
    graph->draw_edges(addition, partial);

    graph->connect_outputs(partial);

    return graph;
}

#endif
