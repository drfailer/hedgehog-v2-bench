#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cblas.h>

#include "graph.hpp"

template <class Type, Order Ord>
bool run_test(size_t n, size_t m, size_t p, size_t block_size, double tol) {
    constexpr char ord_char = (Ord == Order::Row) ? 'R' : 'C';
    printf("  test n=%zu m=%zu p=%zu bs=%zu order=%c ... ",
           n, m, p, block_size, ord_char);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<Type> dist(-5.0, 5.0);

    auto *data_a = new Type[n * m];
    auto *data_b = new Type[m * p];
    auto *data_c = new Type[n * p];
    auto *data_ref = new Type[n * p];

    for (size_t i = 0; i < n * m; ++i) data_a[i] = dist(rng);
    for (size_t i = 0; i < m * p; ++i) data_b[i] = dist(rng);
    for (size_t i = 0; i < n * p; ++i) data_c[i] = dist(rng);

    std::memcpy(data_ref, data_c, n * p * sizeof(Type));

    if constexpr (std::is_same_v<Type, double>) {
        cblas_dgemm(
            Ord == Order::Column ? CblasColMajor : CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            n, p, m, 1.0,
            data_a, Ord == Order::Row ? m : n,
            data_b, Ord == Order::Row ? p : m,
            1.0, data_ref, Ord == Order::Row ? p : n);
    } else {
        cblas_sgemm(
            Ord == Order::Column ? CblasColMajor : CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            n, p, m, 1.0f,
            data_a, Ord == Order::Row ? m : n,
            data_b, Ord == Order::Row ? p : m,
            1.0f, data_ref, Ord == Order::Row ? p : n);
    }

    auto mat_a = std::make_shared<MatrixData<Type, 'a', Ord>>(n, m, block_size, data_a);
    auto mat_b = std::make_shared<MatrixData<Type, 'b', Ord>>(m, p, block_size, data_b);
    auto mat_c = std::make_shared<MatrixData<Type, 'c', Ord>>(n, p, block_size, data_c);

    size_t n_blocks = mat_a->num_blocks_rows();
    size_t m_blocks = mat_a->num_blocks_cols();
    size_t p_blocks = mat_b->num_blocks_cols();

    auto graph = make_dgemm_graph<Type, Ord>(n_blocks, m_blocks, p_blocks, 2, 2);

    graph->start();
    graph->push_data(mat_a);
    graph->push_data(mat_b);
    graph->push_data(mat_c);

    auto result = std::get<0>(graph->get_result());
    graph->stop();

    double max_err = 0.0;
    for (size_t i = 0; i < n * p; ++i) {
        double err = std::abs(static_cast<double>(data_c[i]) - static_cast<double>(data_ref[i]));
        if (err > max_err) max_err = err;
    }

    bool pass = max_err < tol;
    printf("%s (max_err=%.2e)\n", pass ? "PASS" : "FAIL", max_err);

    delete[] data_a;
    delete[] data_b;
    delete[] data_c;
    delete[] data_ref;

    return pass;
}

int main() {
    int failures = 0;

    printf("Column-major tests:\n");
    if (!run_test<double, Order::Column>(16, 16, 16, 4, 1e-10)) ++failures;
    if (!run_test<double, Order::Column>(16, 16, 16, 16, 1e-10)) ++failures;
    if (!run_test<double, Order::Column>(20, 12, 8, 4, 1e-10)) ++failures;
    if (!run_test<double, Order::Column>(100, 100, 100, 32, 1e-10)) ++failures;
    if (!run_test<double, Order::Column>(63, 47, 51, 16, 1e-10)) ++failures;
    if (!run_test<float, Order::Column>(64, 64, 64, 16, 1e-3)) ++failures;

    printf("Row-major tests:\n");
    if (!run_test<double, Order::Row>(16, 16, 16, 4, 1e-10)) ++failures;
    if (!run_test<double, Order::Row>(20, 12, 8, 4, 1e-10)) ++failures;
    if (!run_test<double, Order::Row>(100, 100, 100, 32, 1e-10)) ++failures;
    if (!run_test<double, Order::Row>(63, 47, 51, 16, 1e-10)) ++failures;
    if (!run_test<float, Order::Row>(64, 64, 64, 16, 1e-3)) ++failures;

    printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
