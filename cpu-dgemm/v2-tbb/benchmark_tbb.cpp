#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "../v2/graph.hpp"

template <class Type, Order Ord>
auto make_dgemm_graph_tbb(int tbb_threads,
                          size_t n_blocks, size_t m_blocks, size_t p_blocks,
                          size_t product_threads, size_t addition_threads) {

    auto traversal_a = hh::make_task<RowTraversalTask<Type, 'a', Ord>>(1, "TraversalA");
    auto traversal_b = hh::make_task<ColumnTraversalTask<Type, 'b', Ord>>(1, "TraversalB");
    auto traversal_c = hh::make_task<RowTraversalTask<Type, 'c', Ord>>(1, "TraversalC");

    auto matcher = hh::make_task(
        std::make_shared<InputMatcherTask<Type, Ord>>(n_blocks, m_blocks, p_blocks),
        1, "InputMatcher");

    auto product = hh::make_task<ProductTask<Type, Ord>>(product_threads, "Product");

    auto partial = hh::make_task(
        std::make_shared<PartialComputationTask<Type, Ord>>(n_blocks, p_blocks, m_blocks),
        1, "PartialComputation");

    auto addition = hh::make_task<AdditionTask<Type, Ord>>(addition_threads, "Addition");

    auto runner = std::make_shared<hh::TBBRunner>(tbb_threads);
    auto graph = hh::make_graph<hh::TBBRunner, 3,
        MatrixData<Type, 'a', Ord>, MatrixData<Type, 'b', Ord>, MatrixData<Type, 'c', Ord>,
        MatrixData<Type, 'c', Ord>>(runner, "DGEMM-TBB");

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

int main(int argc, char **argv) {
    using Type = double;
    constexpr Order Ord = Order::Column;

    size_t n = 10000;
    size_t block_size = 1024;
    size_t product_threads = 40;
    size_t addition_threads = 10;
    int tbb_threads = tbb::task_arena::automatic;

    if (argc > 1) n = std::atol(argv[1]);
    if (argc > 2) block_size = std::atol(argv[2]);
    if (argc > 3) product_threads = std::atol(argv[3]);
    if (argc > 4) addition_threads = std::atol(argv[4]);
    if (argc > 5) tbb_threads = std::atoi(argv[5]);

    printf("hedgehog-v2 DGEMM (TBB): n=%zu bs=%zu prod_threads=%zu add_threads=%zu tbb_threads=%d\n",
           n, block_size, product_threads, addition_threads, tbb_threads);

    uint64_t seed = 42;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Type> dist(0, 10);

    auto *data_a = new Type[n * n];
    auto *data_b = new Type[n * n];
    auto *data_c = new Type[n * n];

    for (size_t i = 0; i < n * n; ++i) data_a[i] = dist(rng);
    for (size_t i = 0; i < n * n; ++i) data_b[i] = dist(rng);
    for (size_t i = 0; i < n * n; ++i) data_c[i] = dist(rng);

    auto mat_a = std::make_shared<MatrixData<Type, 'a', Ord>>(n, n, block_size, data_a);
    auto mat_b = std::make_shared<MatrixData<Type, 'b', Ord>>(n, n, block_size, data_b);
    auto mat_c = std::make_shared<MatrixData<Type, 'c', Ord>>(n, n, block_size, data_c);

    size_t n_blocks = mat_a->num_blocks_rows();
    size_t m_blocks = mat_a->num_blocks_cols();
    size_t p_blocks = mat_b->num_blocks_cols();

    printf("blocks: %zu x %zu x %zu\n", n_blocks, m_blocks, p_blocks);

    auto graph = make_dgemm_graph_tbb<Type, Ord>(
        tbb_threads, n_blocks, m_blocks, p_blocks, product_threads, addition_threads);

    graph->start();

    auto t0 = std::chrono::high_resolution_clock::now();

    graph->push_data(mat_a);
    graph->push_data(mat_b);
    graph->push_data(mat_c);

    auto result = std::get<0>(graph->get_result());

    auto t1 = std::chrono::high_resolution_clock::now();

    graph->stop();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double gflops = 2.0 * n * n * n / elapsed / 1e9;
    printf("time: %.3f s  |  %.2f GFLOPS\n", elapsed, gflops);
    graph->generate_dot_file("v2-cpu-dgemm-tbb.dot");

    delete[] data_a;
    delete[] data_b;
    delete[] data_c;

    return 0;
}
