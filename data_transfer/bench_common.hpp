#ifndef BENCH_COMMON_HPP
#define BENCH_COMMON_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstddef>

struct TransferData {
    size_t id;
};

struct Stats {
    double mean;
    double std;
    double min;
    double max;
};

struct BenchResult {
    double global_time_ns;
    std::vector<double> per_data_ns;
};

inline Stats compute_stats(std::vector<double> const &values) {
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / static_cast<double>(values.size());
    double sq_sum = 0.0;
    for (auto v : values) {
        double diff = v - mean;
        sq_sum += diff * diff;
    }
    double std_dev = std::sqrt(sq_sum / static_cast<double>(values.size()));
    double min_val = *std::min_element(values.begin(), values.end());
    double max_val = *std::max_element(values.begin(), values.end());
    return {mean, std_dev, min_val, max_val};
}

constexpr size_t NUM_REPS = 10;

BenchResult bench_v1_transfer(size_t n_tasks, size_t n_threads, size_t ndata);
BenchResult bench_v2_transfer(size_t n_tasks, size_t n_threads, size_t ndata);
BenchResult bench_v2_moodycamel_cond_transfer(size_t n_tasks, size_t n_threads, size_t ndata);
BenchResult bench_v2_moodycamel_atomic_transfer(size_t n_tasks, size_t n_threads, size_t ndata);

#endif
