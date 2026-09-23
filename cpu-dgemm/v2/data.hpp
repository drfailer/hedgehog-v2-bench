#ifndef TUTORIAL4_DATA_HPP
#define TUTORIAL4_DATA_HPP

#include <iostream>
#include <memory>
#include <cmath>
#include <iomanip>
#include <limits>
#include <algorithm>

enum class Order { Row, Column };

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
    MatrixData() = default;

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
        for (size_t i = 0; i < m.height_; ++i) {
            for (size_t j = 0; j < m.width_; ++j) {
                size_t idx = (Ord == Order::Row) ? i * m.ld_ + j : j * m.ld_ + i;
                os << std::setprecision(std::numeric_limits<Type>::digits10 + 1)
                   << m.data_[idx] << " ";
            }
            os << "\n";
        }
        return os;
    }
};

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
        : row_idx_(row_idx), col_idx_(col_idx), ld_(matrix.ld()) , full_data_(matrix.data()) {
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

    friend std::ostream &operator<<(std::ostream &os, MatrixBlockData const &b) {
        os << "Block " << Id << " (" << b.row_idx_ << "," << b.col_idx_ << ")"
           << " [" << b.block_height_ << " x " << b.block_width_ << "] ld=" << b.ld_ << "\n";
        for (size_t i = 0; i < b.block_height_; ++i) {
            for (size_t j = 0; j < b.block_width_; ++j) {
                size_t idx = (Ord == Order::Row) ? i * b.ld_ + j : j * b.ld_ + i;
                os << b.block_data_[idx] << " ";
            }
            os << "\n";
        }
        return os;
    }
};

#endif // TUTORIAL4_DATA_HPP
