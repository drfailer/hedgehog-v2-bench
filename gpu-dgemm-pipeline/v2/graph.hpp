#ifndef GPU_DGEMM_PIPELINE_V2_GRAPH_HPP
#define GPU_DGEMM_PIPELINE_V2_GRAPH_HPP

#include "../../gpu-common/gpu_tasks.hpp"

// GPU computation inner graph builder ////////////////////////////////////////

template <class Type>
auto make_gpu_inner_graph(size_t n, size_t m, size_t p, size_t block_size,
                          size_t product_threads, int device_id = 0) {
    size_t n_blocks = (n + block_size - 1) / block_size;
    size_t m_blocks = (m + block_size - 1) / block_size;
    size_t p_blocks = (p + block_size - 1) / block_size;

    auto pool_a = std::make_shared<hh::Pool<CudaMatrixBlockData<Type, 'a'>>>();
    pool_a->fill(n_blocks + product_threads + 4, block_size, device_id);
    auto pool_b = std::make_shared<hh::Pool<CudaMatrixBlockData<Type, 'b'>>>();
    pool_b->fill(p_blocks + product_threads + 4, block_size, device_id);
    auto pool_p = std::make_shared<hh::Pool<CudaMatrixBlockData<Type, 'p'>>>();
    pool_p->fill(product_threads + 4, block_size, device_id);

    auto copy_in_a = hh::make_task(
        std::make_shared<CudaCopyInTask<Type, 'a'>>(block_size, n, pool_a, device_id), 1, "CopyInA");
    auto copy_in_b = hh::make_task(
        std::make_shared<CudaCopyInTask<Type, 'b'>>(block_size, m, pool_b, device_id), 1, "CopyInB");

    auto matcher = hh::make_state_manager(
        std::make_shared<CudaInputMatcherTask<Type>>(n_blocks, m_blocks, p_blocks),
        "InputMatcher");

    auto product = hh::make_task(
        std::make_shared<CudaProductTask<Type>>(pool_p, device_id), product_threads, "CudaProduct");

    auto copy_out = hh::make_task(
        std::make_shared<CudaCopyOutTask<Type>>(device_id), 1, "CopyOut");

    auto graph = hh::make_graph<2,
        MatrixBlockData<Type, 'a', Order::Column>,
        MatrixBlockData<Type, 'b', Order::Column>,
        MatrixBlockData<Type, 'p', Order::Column>>("GPU Inner Graph");

    graph->connect_inputs(copy_in_a);
    graph->connect_inputs(copy_in_b);

    graph->draw_edges(copy_in_a, matcher);
    graph->draw_edges(copy_in_b, matcher);
    graph->draw_edges(matcher, product);
    graph->draw_edges(product, copy_out);

    graph->connect_outputs(copy_out);

    return graph;
}

// Multi-GPU pipeline //////////////////////////////////////////////////////////

template <class Type>
struct MultiGPUPipeline {
    size_t n_, m_, p_, block_size_, product_threads_;
    std::vector<int> device_ids_;

    MultiGPUPipeline(size_t n, size_t m, size_t p, size_t block_size,
                     size_t product_threads, std::vector<int> device_ids)
        : n_(n), m_(m), p_(p), block_size_(block_size),
          product_threads_(product_threads), device_ids_(std::move(device_ids)) {}

    auto make_graph(size_t index) {
        int dev_id = (index < device_ids_.size()) ? device_ids_[index] : 0;
        return make_gpu_inner_graph<Type>(n_, m_, p_, block_size_, product_threads_, dev_id);
    }

    size_t send_to(std::shared_ptr<MatrixBlockData<Type, 'a', Order::Column>> const &data) {
        return data->col_idx() % device_ids_.size();
    }

    size_t send_to(std::shared_ptr<MatrixBlockData<Type, 'b', Order::Column>> const &data) {
        return data->row_idx() % device_ids_.size();
    }
};

// Outer graph builder /////////////////////////////////////////////////////////

template <class Type, Order Ord>
auto make_gpu_pipeline_dgemm_graph(size_t n, size_t m, size_t p, size_t block_size,
                                   size_t product_threads, size_t addition_threads,
                                   std::vector<int> const &device_ids) {

    size_t n_blocks = (n + block_size - 1) / block_size;
    size_t m_blocks = (m + block_size - 1) / block_size;
    size_t p_blocks = (p + block_size - 1) / block_size;

    auto traversal_a = hh::make_task<ColumnTraversalTask<Type, 'a', Ord>>(1, "TraversalA");
    auto traversal_b = hh::make_task<RowTraversalTask<Type, 'b', Ord>>(1, "TraversalB");
    auto traversal_c = hh::make_task<RowTraversalTask<Type, 'c', Ord>>(1, "TraversalC");

    std::vector<hh::PipelineInfo> configs;
    for (auto dev_id : device_ids) {
        configs.push_back(hh::PipelineInfo{0, dev_id});
    }

    auto pipeline_impl = std::make_shared<MultiGPUPipeline<Type>>(
        n, m, p, block_size, product_threads, device_ids);

    auto pipeline = hh::make_pipeline(pipeline_impl, configs, "MultiGPU Pipeline");

    auto partial = hh::make_task(
        std::make_shared<PartialComputationTask<Type, Ord>>(n_blocks, p_blocks, m_blocks),
        1, "PartialComputation");

    auto addition = hh::make_task<AdditionTask<Type, Ord>>(addition_threads, "Addition");

    auto graph = hh::make_graph<3,
        MatrixData<Type, 'a', Ord>, MatrixData<Type, 'b', Ord>, MatrixData<Type, 'c', Ord>,
        MatrixData<Type, 'c', Ord>>("GPU Pipeline DGEMM");

    graph->connect_inputs(traversal_a);
    graph->connect_inputs(traversal_b);
    graph->connect_inputs(traversal_c);
    graph->connect_inputs(partial);

    graph->draw_edges(traversal_a, pipeline);
    graph->draw_edges(traversal_b, pipeline);

    graph->draw_edges(pipeline, partial);

    graph->draw_edges(traversal_c, partial);
    graph->draw_edges(partial, addition);
    graph->draw_edges(addition, partial);

    graph->connect_outputs(partial);

    return graph;
}

#endif
