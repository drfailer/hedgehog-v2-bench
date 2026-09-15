#ifndef GPU_DGEMM_PIPELINE_V2_DATA_HPP
#define GPU_DGEMM_PIPELINE_V2_DATA_HPP

#include <iostream>
#include <memory>
#include <cmath>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <hedgehog/impl/memory/pool.hpp>
#include "cuda_utils.hpp"

enum class Order { Row, Column };

// Full matrix wrapper ////////////////////////////////////////////////////////

template <class Type, char Id = '0', Order Ord = Order::Row>
class MatrixData {
    size_t height_ = 0;
    size_t width_ = 0;
    size_t block_size_ = 0;
    size_t num_blocks_rows_ = 0;
    size_t num_blocks_cols_ = 0;
    size_t ld_ = 0;
    Type *data_ = nullptr;

  public:
    MatrixData(size_t height, size_t width, size_t block_size, Type *data)
        : height_(height), width_(width), block_size_(std::max(block_size, size_t(1))), data_(data) {
        num_blocks_rows_ = (height_ + block_size_ - 1) / block_size_;
        num_blocks_cols_ = (width_ + block_size_ - 1) / block_size_;
        ld_ = (Ord == Order::Row) ? width_ : height_;
    }

    size_t height() const { return height_; }
    size_t width() const { return width_; }
    size_t block_size() const { return block_size_; }
    size_t num_blocks_rows() const { return num_blocks_rows_; }
    size_t num_blocks_cols() const { return num_blocks_cols_; }
    size_t ld() const { return ld_; }
    Type *data() const { return data_; }

    friend std::ostream &operator<<(std::ostream &os, MatrixData const &m) {
        os << "MatrixData " << Id
           << " (" << m.height_ << " x " << m.width_ << ")"
           << " blocks (" << m.num_blocks_rows_ << " x " << m.num_blocks_cols_ << ")"
           << " bs=" << m.block_size_ << " ld=" << m.ld_ << "\n";
        return os;
    }
};

// CPU matrix block (view into full matrix) ///////////////////////////////////

template <class Type, char Id, Order Ord>
class MatrixBlockData {
  protected:
    size_t row_idx_ = 0;
    size_t col_idx_ = 0;
    size_t block_height_ = 0;
    size_t block_width_ = 0;
    size_t ld_ = 0;
    Type *full_data_ = nullptr;
    Type *block_data_ = nullptr;

  public:
    MatrixBlockData() = default;

    MatrixBlockData(size_t row_idx, size_t col_idx,
                    size_t block_height, size_t block_width,
                    size_t ld, Type *full_data, Type *block_data)
        : row_idx_(row_idx), col_idx_(col_idx),
          block_height_(block_height), block_width_(block_width),
          ld_(ld), full_data_(full_data), block_data_(block_data) {}

    MatrixBlockData(size_t row_idx, size_t col_idx,
                    size_t block_height, size_t block_width,
                    Type *full_data, Type *block_data)
        : row_idx_(row_idx), col_idx_(col_idx),
          block_height_(block_height), block_width_(block_width),
          full_data_(full_data), block_data_(block_data) {
        ld_ = (Ord == Order::Row) ? block_width_ : block_height_;
    }

    MatrixBlockData(size_t row_idx, size_t col_idx, MatrixData<Type, Id, Ord> &matrix)
        : row_idx_(row_idx), col_idx_(col_idx), ld_(matrix.ld()), full_data_(matrix.data()) {
        block_height_ = std::min(matrix.block_size(), matrix.height() - row_idx * matrix.block_size());
        block_width_ = std::min(matrix.block_size(), matrix.width() - col_idx * matrix.block_size());
        if constexpr (Ord == Order::Row) {
            block_data_ = matrix.data()
                + (row_idx * matrix.block_size()) * matrix.ld()
                + col_idx * matrix.block_size();
        } else {
            block_data_ = matrix.data()
                + (col_idx * matrix.block_size()) * matrix.ld()
                + row_idx * matrix.block_size();
        }
    }

    template <char OldId>
    explicit MatrixBlockData(std::shared_ptr<MatrixBlockData<Type, OldId, Ord>> const &o)
        : row_idx_(o->row_idx()), col_idx_(o->col_idx()),
          block_height_(o->block_height()), block_width_(o->block_width()),
          ld_(o->ld()), full_data_(o->full_data()), block_data_(o->block_data()) {}

    size_t row_idx() const { return row_idx_; }
    size_t col_idx() const { return col_idx_; }
    size_t block_height() const { return block_height_; }
    size_t block_width() const { return block_width_; }
    size_t ld() const { return ld_; }
    Type *full_data() const { return full_data_; }
    Type *block_data() const { return block_data_; }
};

// GPU matrix block (pool-allocated device memory) ////////////////////////////

template <class Type, char Id>
struct CudaMatrixBlockData {
    size_t row_idx_ = 0;
    size_t col_idx_ = 0;
    size_t block_height_ = 0;
    size_t block_width_ = 0;
    size_t ld_ = 0;
    Type *device_data_ = nullptr;
    Type *host_full_data_ = nullptr;
    int device_id_ = 0;

    CudaMatrixBlockData() = default;

    CudaMatrixBlockData(size_t block_size, int device_id)
        : device_id_(device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&device_data_), sizeof(Type) * block_size * block_size));
    }

    ~CudaMatrixBlockData() {
        if (device_data_) {
            cudaSetDevice(device_id_);
            cudaFree(device_data_);
        }
    }

    CudaMatrixBlockData(CudaMatrixBlockData const &) = delete;
    CudaMatrixBlockData &operator=(CudaMatrixBlockData const &) = delete;

    void clean_memory() {
        row_idx_ = 0;
        col_idx_ = 0;
        block_height_ = 0;
        block_width_ = 0;
        ld_ = 0;
        host_full_data_ = nullptr;
    }

    size_t row_idx() const { return row_idx_; }
    size_t col_idx() const { return col_idx_; }
    size_t block_height() const { return block_height_; }
    size_t block_width() const { return block_width_; }
    size_t ld() const { return ld_; }
    Type *device_data() const { return device_data_; }

    std::shared_ptr<MatrixBlockData<Type, Id, Order::Column>> copy_to_host(cudaStream_t stream) const {
        auto *buf = new Type[block_height_ * block_width_]();
        CUDA_CHECK(cudaMemcpyAsync(buf, device_data_,
            block_height_ * block_width_ * sizeof(Type),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        return std::make_shared<MatrixBlockData<Type, Id, Order::Column>>(
            row_idx_, col_idx_, block_height_, block_width_, buf, buf);
    }
};

template <class Type, char Id>
auto pool_wrap(CudaMatrixBlockData<Type, Id> *raw,
               std::shared_ptr<hh::Pool<CudaMatrixBlockData<Type, Id>>> pool) {
    return std::shared_ptr<CudaMatrixBlockData<Type, Id>>(
        raw, [pool](CudaMatrixBlockData<Type, Id> *p) { pool->release(p); });
}

#endif
