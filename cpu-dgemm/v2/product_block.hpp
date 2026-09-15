#ifndef TUTORIAL4_PRODUCT_BLOCK_HPP
#define TUTORIAL4_PRODUCT_BLOCK_HPP

#include <cstring>

template <class Type, Order Ord>
struct ProductBlock {
    size_t row_idx_ = 0;
    size_t col_idx_ = 0;
    size_t block_height_ = 0;
    size_t block_width_ = 0;
    size_t ld_ = 0;
    Type *data_ = nullptr;
    size_t capacity_ = 0;

    ProductBlock() = default;

    explicit ProductBlock(size_t block_size)
        : capacity_(block_size * block_size) {
        data_ = new Type[capacity_]();
    }

    ~ProductBlock() { delete[] data_; }

    ProductBlock(ProductBlock const &) = delete;
    ProductBlock &operator=(ProductBlock const &) = delete;

    void clean_memory() {
        row_idx_ = col_idx_ = 0;
        block_height_ = block_width_ = 0;
        ld_ = 0;
        std::memset(data_, 0, capacity_ * sizeof(Type));
    }

    void setup(size_t row, size_t col, size_t height, size_t width) {
        row_idx_ = row;
        col_idx_ = col;
        block_height_ = height;
        block_width_ = width;
        ld_ = (Ord == Order::Row) ? width : height;
    }

    size_t row_idx() const { return row_idx_; }
    size_t col_idx() const { return col_idx_; }
    size_t block_height() const { return block_height_; }
    size_t block_width() const { return block_width_; }
    size_t ld() const { return ld_; }
    Type *block_data() const { return data_; }
};

#endif
