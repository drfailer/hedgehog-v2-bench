#ifndef GPU_DGEMM_V2_GRAPH_HPP
#define GPU_DGEMM_V2_GRAPH_HPP

#include <cublas_v2.h>
#include <hedgehog/hedgehog.h>
#include "data.hpp"

// Type aliases ////////////////////////////////////////////////////////////////

template <class Type>
using CudaBlockPair = std::pair<
    std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>,
    std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>>;

template <class Type, Order Ord>
using AccumulationPair = std::pair<
    std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>,
    std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>;

// Traversal tasks /////////////////////////////////////////////////////////////

template <class Type, char Id, Order Ord>
struct RowTraversalTask {
    using inputs = hh::type_list<MatrixData<Type, Id, Ord>>;
    using outputs = hh::type_list<MatrixBlockData<Type, Id, Ord>>;

    void execute(auto ctx, std::shared_ptr<MatrixData<Type, Id, Ord>> matrix) {
        for (size_t i = 0; i < matrix->num_blocks_rows(); ++i)
            for (size_t j = 0; j < matrix->num_blocks_cols(); ++j)
                ctx->push_result(
                    std::make_shared<MatrixBlockData<Type, Id, Ord>>(i, j, *matrix));
    }
};

template <class Type, char Id, Order Ord>
struct ColumnTraversalTask {
    using inputs = hh::type_list<MatrixData<Type, Id, Ord>>;
    using outputs = hh::type_list<MatrixBlockData<Type, Id, Ord>>;

    void execute(auto ctx, std::shared_ptr<MatrixData<Type, Id, Ord>> matrix) {
        for (size_t j = 0; j < matrix->num_blocks_cols(); ++j)
            for (size_t i = 0; i < matrix->num_blocks_rows(); ++i)
                ctx->push_result(
                    std::make_shared<MatrixBlockData<Type, Id, Ord>>(i, j, *matrix));
    }
};

// CUDA copy-in task ///////////////////////////////////////////////////////////

template <class Type, char Id>
struct CudaCopyInTask {
    using inputs = hh::type_list<MatrixBlockData<Type, Id, Order::Column>>;
    using outputs = hh::type_list<CudaMatrixBlockData<Type, Id>>;

    size_t block_size_;
    size_t matrix_ld_;
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<hh::Pool<CudaMatrixBlockData<Type, Id>>> pool_;

    CudaCopyInTask(size_t block_size, size_t matrix_ld,
                   std::shared_ptr<hh::Pool<CudaMatrixBlockData<Type, Id>>> pool)
        : block_size_(block_size), matrix_ld_(matrix_ld), pool_(std::move(pool)) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaCopyInTask() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    std::shared_ptr<CudaCopyInTask> copy() {
        return std::make_shared<CudaCopyInTask>(block_size_, matrix_ld_, pool_);
    }

    void execute(auto ctx, std::shared_ptr<MatrixBlockData<Type, Id, Order::Column>> ptr) {
        auto *raw = pool_->allocate(true);
        raw->row_idx_ = ptr->row_idx();
        raw->col_idx_ = ptr->col_idx();
        raw->block_height_ = ptr->block_height();
        raw->block_width_ = ptr->block_width();
        raw->ld_ = ptr->block_height();
        raw->host_full_data_ = ptr->full_data();
        auto block = pool_wrap(raw, pool_);

        if (ptr->ld() == block->ld()) {
            CUDA_CHECK(cudaMemcpyAsync(
                block->device_data(), ptr->block_data(),
                sizeof(Type) * block->block_height() * block->block_width(),
                cudaMemcpyHostToDevice, stream_));
        } else {
            CUDA_CHECK(cublasSetMatrixAsync(
                (int)block->block_height(), (int)block->block_width(), sizeof(Type),
                ptr->full_data()
                    + IDX2C(ptr->row_idx() * block_size_,
                            ptr->col_idx() * block_size_, matrix_ld_),
                (int)matrix_ld_,
                block->device_data(), (int)block->ld(), stream_));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_));
        ctx->push_result(std::move(block));
    }
};

// CUDA input matcher (pairs A,B GPU blocks) ///////////////////////////////////

template <class Type>
struct CudaInputMatcherTask {
    using inputs = hh::type_list<
        CudaMatrixBlockData<Type, 'a'>,
        CudaMatrixBlockData<Type, 'b'>>;
    using outputs = hh::type_list<CudaBlockPair<Type>>;

    size_t n_blocks_, m_blocks_, p_blocks_;

    std::vector<std::shared_ptr<CudaMatrixBlockData<Type, 'a'>>> grid_a_;
    std::vector<std::shared_ptr<CudaMatrixBlockData<Type, 'b'>>> grid_b_;
    std::vector<size_t> ttl_a_, ttl_b_;

    CudaInputMatcherTask(size_t n_blocks, size_t m_blocks, size_t p_blocks)
        : n_blocks_(n_blocks), m_blocks_(m_blocks), p_blocks_(p_blocks),
          grid_a_(n_blocks * m_blocks, nullptr),
          grid_b_(m_blocks * p_blocks, nullptr),
          ttl_a_(n_blocks * m_blocks, p_blocks),
          ttl_b_(m_blocks * p_blocks, n_blocks) {}

    void execute(auto ctx, std::shared_ptr<CudaMatrixBlockData<Type, 'a'>> a) {
        size_t i = a->row_idx(), k = a->col_idx();
        grid_a_[i * m_blocks_ + k] = a;

        for (size_t j = 0; j < p_blocks_; ++j) {
            if (auto b = get_b(k, j)) {
                --ttl_a_[i * m_blocks_ + k];
                if (ttl_a_[i * m_blocks_ + k] == 0)
                    grid_a_[i * m_blocks_ + k] = nullptr;
                ctx->push_result(std::make_shared<CudaBlockPair<Type>>(a, b));
            }
        }
    }

    void execute(auto ctx, std::shared_ptr<CudaMatrixBlockData<Type, 'b'>> b) {
        size_t k = b->row_idx(), j = b->col_idx();
        grid_b_[k * p_blocks_ + j] = b;

        for (size_t i = 0; i < n_blocks_; ++i) {
            if (auto a = get_a(i, k)) {
                --ttl_b_[k * p_blocks_ + j];
                if (ttl_b_[k * p_blocks_ + j] == 0)
                    grid_b_[k * p_blocks_ + j] = nullptr;
                ctx->push_result(std::make_shared<CudaBlockPair<Type>>(a, b));
            }
        }
    }

  private:
    std::shared_ptr<CudaMatrixBlockData<Type, 'a'>> get_a(size_t i, size_t k) {
        auto ptr = grid_a_[i * m_blocks_ + k];
        if (ptr) {
            --ttl_a_[i * m_blocks_ + k];
            if (ttl_a_[i * m_blocks_ + k] == 0)
                grid_a_[i * m_blocks_ + k] = nullptr;
        }
        return ptr;
    }

    std::shared_ptr<CudaMatrixBlockData<Type, 'b'>> get_b(size_t k, size_t j) {
        auto ptr = grid_b_[k * p_blocks_ + j];
        if (ptr) {
            --ttl_b_[k * p_blocks_ + j];
            if (ttl_b_[k * p_blocks_ + j] == 0)
                grid_b_[k * p_blocks_ + j] = nullptr;
        }
        return ptr;
    }
};

// CUDA product task ///////////////////////////////////////////////////////////

template <class Type>
struct CudaProductTask {
    using inputs = hh::type_list<CudaBlockPair<Type>>;
    using outputs = hh::type_list<CudaMatrixBlockData<Type, 'p'>>;

    cudaStream_t stream_ = nullptr;
    cublasHandle_t handle_ = nullptr;
    std::shared_ptr<hh::Pool<CudaMatrixBlockData<Type, 'p'>>> pool_;

    explicit CudaProductTask(std::shared_ptr<hh::Pool<CudaMatrixBlockData<Type, 'p'>>> pool)
        : pool_(std::move(pool)) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cublasCreate_v2(&handle_));
        CUDA_CHECK(cublasSetStream_v2(handle_, stream_));
    }

    ~CudaProductTask() {
        if (handle_) cublasDestroy_v2(handle_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    std::shared_ptr<CudaProductTask> copy() {
        return std::make_shared<CudaProductTask>(pool_);
    }

    void execute(auto ctx, std::shared_ptr<CudaBlockPair<Type>> pair) {
        auto &a = pair->first;
        auto &b = pair->second;

        auto *raw = pool_->allocate(true);
        raw->row_idx_ = a->row_idx();
        raw->col_idx_ = b->col_idx();
        raw->block_height_ = a->block_height();
        raw->block_width_ = b->block_width();
        raw->ld_ = a->block_height();
        auto res = pool_wrap(raw, pool_);

        Type alpha = 1, beta = 0;

        if constexpr (std::is_same_v<Type, float>) {
            CUDA_CHECK(cublasSgemm_v2(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)a->block_height(), (int)b->block_width(), (int)a->block_width(),
                &alpha,
                a->device_data(), (int)a->ld(),
                b->device_data(), (int)b->ld(),
                &beta,
                res->device_data(), (int)res->ld()));
        } else {
            CUDA_CHECK(cublasDgemm_v2(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)a->block_height(), (int)b->block_width(), (int)a->block_width(),
                &alpha,
                a->device_data(), (int)a->ld(),
                b->device_data(), (int)b->ld(),
                &beta,
                res->device_data(), (int)res->ld()));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_));
        ctx->push_result(std::move(res));
    }
};

// CUDA copy-out task //////////////////////////////////////////////////////////

template <class Type>
struct CudaCopyOutTask {
    using inputs = hh::type_list<CudaMatrixBlockData<Type, 'p'>>;
    using outputs = hh::type_list<MatrixBlockData<Type, 'p', Order::Column>>;

    cudaStream_t stream_ = nullptr;

    CudaCopyOutTask() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaCopyOutTask() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    std::shared_ptr<CudaCopyOutTask> copy() {
        return std::make_shared<CudaCopyOutTask>();
    }

    void execute(auto ctx, std::shared_ptr<CudaMatrixBlockData<Type, 'p'>> ptr) {
        auto result = ptr->copy_to_host(stream_);
        ctx->push_result(std::move(result));
    }
};

// Partial computation task (accumulates C += P) ///////////////////////////////

template <class Type, Order Ord>
struct PartialComputationTask {
    using inputs = hh::type_list<
        MatrixData<Type, 'c', Ord>,
        MatrixBlockData<Type, 'c', Ord>,
        MatrixBlockData<Type, 'p', Ord>>;
    using outputs = hh::type_list<
        AccumulationPair<Type, Ord>,
        MatrixData<Type, 'c', Ord>>;

    size_t n_blocks_, p_blocks_, m_blocks_;
    size_t total_done_ = 0;
    std::shared_ptr<MatrixData<Type, 'c', Ord>> mat_c_;
    std::vector<std::shared_ptr<MatrixBlockData<Type, 'c', Ord>>> grid_c_;
    std::vector<std::vector<std::shared_ptr<MatrixBlockData<Type, 'p', Ord>>>> grid_p_;
    std::vector<size_t> done_count_;

    PartialComputationTask(size_t n_blocks, size_t p_blocks, size_t m_blocks)
        : n_blocks_(n_blocks), p_blocks_(p_blocks), m_blocks_(m_blocks),
          grid_c_(n_blocks * p_blocks, nullptr),
          grid_p_(n_blocks * p_blocks),
          done_count_(n_blocks * p_blocks, 0) {}

    void execute([[maybe_unused]] auto ctx, std::shared_ptr<MatrixData<Type, 'c', Ord>> mat_c) {
        mat_c_ = std::move(mat_c);
    }

    void execute(auto ctx, std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> c) {
        size_t idx = c->row_idx() * p_blocks_ + c->col_idx();

        if (done_count_[idx] == m_blocks_) {
            ++total_done_;
            if (total_done_ == n_blocks_ * p_blocks_)
                ctx->push_result(mat_c_);
        } else if (!grid_p_[idx].empty()) {
            auto p = grid_p_[idx].back();
            grid_p_[idx].pop_back();
            ++done_count_[idx];
            ctx->push_result(std::make_shared<AccumulationPair<Type, Ord>>(c, p));
        } else {
            grid_c_[idx] = c;
        }
    }

    void execute(auto ctx, std::shared_ptr<MatrixBlockData<Type, 'p', Ord>> p) {
        size_t idx = p->row_idx() * p_blocks_ + p->col_idx();

        if (grid_c_[idx]) {
            auto c = grid_c_[idx];
            grid_c_[idx] = nullptr;
            ++done_count_[idx];
            ctx->push_result(std::make_shared<AccumulationPair<Type, Ord>>(c, p));
        } else {
            grid_p_[idx].push_back(p);
        }
    }
};

// Addition task ///////////////////////////////////////////////////////////////

template <class Type, Order Ord>
struct AdditionTask {
    using inputs = hh::type_list<AccumulationPair<Type, Ord>>;
    using outputs = hh::type_list<MatrixBlockData<Type, 'c', Ord>>;

    std::shared_ptr<AdditionTask> copy() { return std::make_shared<AdditionTask>(); }

    void execute(auto ctx, std::shared_ptr<AccumulationPair<Type, Ord>> pair) {
        auto &c = pair->first;
        auto &p = pair->second;

        if constexpr (Ord == Order::Row) {
            for (size_t i = 0; i < c->block_height(); ++i)
                for (size_t j = 0; j < c->block_width(); ++j)
                    c->block_data()[i * c->ld() + j] += p->block_data()[i * p->ld() + j];
        } else {
            for (size_t j = 0; j < c->block_width(); ++j)
                for (size_t i = 0; i < c->block_height(); ++i)
                    c->block_data()[j * c->ld() + i] += p->block_data()[j * p->ld() + i];
        }

        delete[] p->block_data();
        ctx->push_result(c);
    }
};

// Graph builder ///////////////////////////////////////////////////////////////

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
