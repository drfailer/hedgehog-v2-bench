#ifndef GPU_COMMON_GPU_TASKS_HPP
#define GPU_COMMON_GPU_TASKS_HPP

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
    int device_id_;
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<hh::Pool<CudaMatrixBlockData<Type, Id>>> pool_;

    CudaCopyInTask(size_t block_size, size_t matrix_ld,
                   std::shared_ptr<hh::Pool<CudaMatrixBlockData<Type, Id>>> pool,
                   int device_id = 0)
        : block_size_(block_size), matrix_ld_(matrix_ld),
          device_id_(device_id), pool_(std::move(pool)) {}

    ~CudaCopyInTask() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    void initialize() {
        CUDA_CHECK(cudaSetDevice(device_id_));
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    void finalize() {
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }

    std::shared_ptr<CudaCopyInTask> copy() {
        return std::make_shared<CudaCopyInTask>(block_size_, matrix_ld_, pool_, device_id_);
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

    int device_id_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t handle_ = nullptr;
    std::shared_ptr<hh::Pool<CudaMatrixBlockData<Type, 'p'>>> pool_;

    CudaProductTask(std::shared_ptr<hh::Pool<CudaMatrixBlockData<Type, 'p'>>> pool,
                    int device_id = 0)
        : device_id_(device_id), pool_(std::move(pool)) {}

    ~CudaProductTask() {
        if (handle_) cublasDestroy_v2(handle_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    void initialize() {
        CUDA_CHECK(cudaSetDevice(device_id_));
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cublasCreate_v2(&handle_));
        CUDA_CHECK(cublasSetStream_v2(handle_, stream_));
    }

    void finalize() {
        if (handle_) { cublasDestroy_v2(handle_); handle_ = nullptr; }
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }

    std::shared_ptr<CudaProductTask> copy() {
        return std::make_shared<CudaProductTask>(pool_, device_id_);
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

    int device_id_;
    cudaStream_t stream_ = nullptr;

    explicit CudaCopyOutTask(int device_id = 0) : device_id_(device_id) {}

    ~CudaCopyOutTask() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    void initialize() {
        CUDA_CHECK(cudaSetDevice(device_id_));
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    void finalize() {
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }

    std::shared_ptr<CudaCopyOutTask> copy() {
        return std::make_shared<CudaCopyOutTask>(device_id_);
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

#endif
