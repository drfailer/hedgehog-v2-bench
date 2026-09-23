#ifndef TUTORIAL4_GRAPH_PTR_HPP
#define TUTORIAL4_GRAPH_PTR_HPP

#include <cblas.h>
#include <hedgehog/hedgehog.h>
#include "data.hpp"
#include "product_block.hpp"

template <class Type, Order Ord>
using BlockPair = std::pair<
    MatrixBlockData<Type, 'a', Ord>*,
    MatrixBlockData<Type, 'b', Ord>*>;

template <class Type, Order Ord>
using AccumulationPair = std::pair<
    MatrixBlockData<Type, 'c', Ord>*,
    ProductBlock<Type, Ord>*>;

// Traversal tasks /////////////////////////////////////////////////////////////

template <class Type, char Id, Order Ord>
struct RowTraversalTask {
    using inputs = hh::type_list<MatrixData<Type, Id, Ord>>;
    using outputs = hh::type_list<MatrixBlockData<Type, Id, Ord>>;

    void execute(auto ctx, MatrixData<Type, Id, Ord>* matrix) {
        for (size_t i = 0; i < matrix->num_blocks_rows(); ++i)
            for (size_t j = 0; j < matrix->num_blocks_cols(); ++j)
                ctx->push_result(new MatrixBlockData<Type, Id, Ord>(i, j, *matrix));
    }
};

template <class Type, char Id, Order Ord>
struct ColumnTraversalTask {
    using inputs = hh::type_list<MatrixData<Type, Id, Ord>>;
    using outputs = hh::type_list<MatrixBlockData<Type, Id, Ord>>;

    void execute(auto ctx, MatrixData<Type, Id, Ord>* matrix) {
        for (size_t j = 0; j < matrix->num_blocks_cols(); ++j)
            for (size_t i = 0; i < matrix->num_blocks_rows(); ++i)
                ctx->push_result(new MatrixBlockData<Type, Id, Ord>(i, j, *matrix));
    }
};

// Input matcher ///////////////////////////////////////////////////////////////

template <class Type, Order Ord>
struct InputMatcherTask {
    using inputs = hh::type_list<
        MatrixBlockData<Type, 'a', Ord>,
        MatrixBlockData<Type, 'b', Ord>>;
    using outputs = hh::type_list<BlockPair<Type, Ord>>;

    size_t n_blocks_, m_blocks_, p_blocks_;
    std::vector<MatrixBlockData<Type, 'a', Ord>*> grid_a_;
    std::vector<MatrixBlockData<Type, 'b', Ord>*> grid_b_;
    std::vector<size_t> ttl_a_, ttl_b_;

    InputMatcherTask(size_t n_blocks, size_t m_blocks, size_t p_blocks)
        : n_blocks_(n_blocks), m_blocks_(m_blocks), p_blocks_(p_blocks),
          grid_a_(n_blocks * m_blocks, nullptr),
          grid_b_(m_blocks * p_blocks, nullptr),
          ttl_a_(n_blocks * m_blocks, p_blocks),
          ttl_b_(m_blocks * p_blocks, n_blocks) {}

    std::shared_ptr<InputMatcherTask> copy() {
        return std::make_shared<InputMatcherTask>(n_blocks_, m_blocks_, p_blocks_);
    }

    void execute(auto ctx, MatrixBlockData<Type, 'a', Ord>* a) {
        size_t i = a->row_idx(), k = a->col_idx();
        grid_a_[i * m_blocks_ + k] = a;

        for (size_t j = 0; j < p_blocks_; ++j) {
            if (auto *b = get_b(k, j)) {
                --ttl_a_[i * m_blocks_ + k];
                if (ttl_a_[i * m_blocks_ + k] == 0)
                    grid_a_[i * m_blocks_ + k] = nullptr;
                ctx->push_result(new BlockPair<Type, Ord>(a, b));
            }
        }
    }

    void execute(auto ctx, MatrixBlockData<Type, 'b', Ord>* b) {
        size_t k = b->row_idx(), j = b->col_idx();
        grid_b_[k * p_blocks_ + j] = b;

        for (size_t i = 0; i < n_blocks_; ++i) {
            if (auto *a = get_a(i, k)) {
                --ttl_b_[k * p_blocks_ + j];
                if (ttl_b_[k * p_blocks_ + j] == 0)
                    grid_b_[k * p_blocks_ + j] = nullptr;
                ctx->push_result(new BlockPair<Type, Ord>(a, b));
            }
        }
    }

  private:
    MatrixBlockData<Type, 'a', Ord>* get_a(size_t i, size_t k) {
        auto *ptr = grid_a_[i * m_blocks_ + k];
        if (ptr) {
            --ttl_a_[i * m_blocks_ + k];
            if (ttl_a_[i * m_blocks_ + k] == 0)
                grid_a_[i * m_blocks_ + k] = nullptr;
        }
        return ptr;
    }

    MatrixBlockData<Type, 'b', Ord>* get_b(size_t k, size_t j) {
        auto *ptr = grid_b_[k * p_blocks_ + j];
        if (ptr) {
            --ttl_b_[k * p_blocks_ + j];
            if (ttl_b_[k * p_blocks_ + j] == 0)
                grid_b_[k * p_blocks_ + j] = nullptr;
        }
        return ptr;
    }
};

// Product task (pool-allocated) ///////////////////////////////////////////////

template <class Type, Order Ord>
struct ProductTask {
    using inputs = hh::type_list<BlockPair<Type, Ord>>;
    using outputs = hh::type_list<ProductBlock<Type, Ord>>;

    std::shared_ptr<hh::Pool<ProductBlock<Type, Ord>>> pool_;

    ProductTask(std::shared_ptr<hh::Pool<ProductBlock<Type, Ord>>> pool)
        : pool_(std::move(pool)) {}

    std::shared_ptr<ProductTask> copy() {
        return std::make_shared<ProductTask>(pool_);
    }

    void execute(auto ctx, BlockPair<Type, Ord>* pair) {
        auto *a = pair->first;
        auto *b = pair->second;

        size_t M = a->block_height(), N = b->block_width(), K = a->block_width();

        auto *product = pool_->allocate(true);
        product->setup(a->row_idx(), b->col_idx(), M, N);

        if constexpr (std::is_same_v<Type, float>) {
            cblas_sgemm(Ord == Order::Column ? CblasColMajor : CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f,
                a->block_data(), a->ld(),
                b->block_data(), b->ld(),
                1.0f,
                product->block_data(), product->ld());
        } else {
            cblas_dgemm(Ord == Order::Column ? CblasColMajor : CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0,
                a->block_data(), a->ld(),
                b->block_data(), b->ld(),
                1.0,
                product->block_data(), product->ld());
        }

        delete pair;
        ctx->push_result(product);
    }
};

// Partial computation task ////////////////////////////////////////////////////

template <class Type, Order Ord>
struct PartialComputationTask {
    using inputs = hh::type_list<
        MatrixData<Type, 'c', Ord>,
        MatrixBlockData<Type, 'c', Ord>,
        ProductBlock<Type, Ord>>;
    using outputs = hh::type_list<
        AccumulationPair<Type, Ord>,
        MatrixData<Type, 'c', Ord>>;

    size_t n_blocks_, p_blocks_, m_blocks_;
    size_t total_done_ = 0;
    MatrixData<Type, 'c', Ord>* mat_c_ = nullptr;
    std::vector<MatrixBlockData<Type, 'c', Ord>*> grid_c_;
    std::vector<std::vector<ProductBlock<Type, Ord>*>> grid_p_;
    std::vector<size_t> done_count_;

    PartialComputationTask(size_t n_blocks, size_t p_blocks, size_t m_blocks)
        : n_blocks_(n_blocks), p_blocks_(p_blocks), m_blocks_(m_blocks),
          grid_c_(n_blocks * p_blocks, nullptr),
          grid_p_(n_blocks * p_blocks),
          done_count_(n_blocks * p_blocks, 0) {}

    std::shared_ptr<PartialComputationTask> copy() {
        return std::make_shared<PartialComputationTask>(n_blocks_, p_blocks_, m_blocks_);
    }

    void execute([[maybe_unused]] auto ctx, MatrixData<Type, 'c', Ord>* mat_c) {
        mat_c_ = mat_c;
    }

    void execute(auto ctx, MatrixBlockData<Type, 'c', Ord>* c) {
        size_t idx = c->row_idx() * p_blocks_ + c->col_idx();

        if (done_count_[idx] == m_blocks_) {
            ++total_done_;
            delete c;
            if (total_done_ == n_blocks_ * p_blocks_)
                ctx->push_result(mat_c_);
        } else if (!grid_p_[idx].empty()) {
            auto *p = grid_p_[idx].back();
            grid_p_[idx].pop_back();
            ++done_count_[idx];
            ctx->push_result(new AccumulationPair<Type, Ord>(c, p));
        } else {
            grid_c_[idx] = c;
        }
    }

    void execute(auto ctx, ProductBlock<Type, Ord>* p) {
        size_t idx = p->row_idx() * p_blocks_ + p->col_idx();

        if (grid_c_[idx]) {
            auto *c = grid_c_[idx];
            grid_c_[idx] = nullptr;
            ++done_count_[idx];
            ctx->push_result(new AccumulationPair<Type, Ord>(c, p));
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

    std::shared_ptr<hh::Pool<ProductBlock<Type, Ord>>> pool_;

    AdditionTask(std::shared_ptr<hh::Pool<ProductBlock<Type, Ord>>> pool)
        : pool_(std::move(pool)) {}

    std::shared_ptr<AdditionTask> copy() {
        return std::make_shared<AdditionTask>(pool_);
    }

    void execute(auto ctx, AccumulationPair<Type, Ord>* pair) {
        auto *c = pair->first;
        auto *p = pair->second;

        if constexpr (Ord == Order::Row) {
            for (size_t i = 0; i < c->block_height(); ++i)
                for (size_t j = 0; j < c->block_width(); ++j)
                    c->block_data()[i * c->ld() + j] += p->block_data()[i * p->ld() + j];
        } else {
            for (size_t j = 0; j < c->block_width(); ++j)
                for (size_t i = 0; i < c->block_height(); ++i)
                    c->block_data()[j * c->ld() + i] += p->block_data()[j * p->ld() + i];
        }

        pool_->release(p);
        delete pair;
        ctx->push_result(c);
    }
};

// Graph builder ///////////////////////////////////////////////////////////////

template <class Type, Order Ord>
auto make_dgemm_graph(size_t n_blocks, size_t m_blocks, size_t p_blocks,
                      size_t block_size, size_t product_threads, size_t addition_threads) {

    auto pool = std::make_shared<hh::Pool<ProductBlock<Type, Ord>>>();
    pool->fill(product_threads + addition_threads + 4, block_size);

    auto traversal_a = hh::make_task<RowTraversalTask<Type, 'a', Ord>>(1, "TraversalA");
    auto traversal_b = hh::make_task<ColumnTraversalTask<Type, 'b', Ord>>(1, "TraversalB");
    auto traversal_c = hh::make_task<RowTraversalTask<Type, 'c', Ord>>(1, "TraversalC");

    auto matcher = hh::make_task(
        std::make_shared<InputMatcherTask<Type, Ord>>(n_blocks, m_blocks, p_blocks),
        1, "InputMatcher");

    auto product = hh::make_task(
        std::make_shared<ProductTask<Type, Ord>>(pool),
        product_threads, "Product");

    auto partial = hh::make_task(
        std::make_shared<PartialComputationTask<Type, Ord>>(n_blocks, p_blocks, m_blocks),
        1, "PartialComputation");

    auto addition = hh::make_task(
        std::make_shared<AdditionTask<Type, Ord>>(pool),
        addition_threads, "Addition");

    auto graph = hh::make_graph<3,
        MatrixData<Type, 'a', Ord>, MatrixData<Type, 'b', Ord>, MatrixData<Type, 'c', Ord>,
        MatrixData<Type, 'c', Ord>>("DGEMM-ptr");

    graph->connect_inputs(traversal_a);
    graph->connect_inputs(traversal_b);
    graph->connect_inputs(traversal_c);
    graph->connect_inputs(partial);

    graph->draw_edges(traversal_a, matcher);
    graph->draw_edges(traversal_b, matcher);
    graph->draw_edges(traversal_c, partial);
    graph->draw_edges(matcher, product);
    graph->draw_edges(product, partial);
    graph->draw_edges(partial, addition);
    graph->draw_edges(addition, partial);

    graph->connect_outputs(partial);

    return graph;
}

#endif
