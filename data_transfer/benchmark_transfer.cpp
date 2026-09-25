#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include "bench_common.hpp"

static std::string format_duration(double ns) {
    char buf[32];
    if (ns >= 1e9) {
        std::snprintf(buf, sizeof(buf), "%.3fs", ns / 1e9);
    } else if (ns >= 1e6) {
        std::snprintf(buf, sizeof(buf), "%.3fms", ns / 1e6);
    } else if (ns >= 1e3) {
        std::snprintf(buf, sizeof(buf), "%.3fus", ns / 1e3);
    } else {
        std::snprintf(buf, sizeof(buf), "%.0fns", ns);
    }
    return buf;
}

static void print_results(char const *label, BenchResult const &res,
                          size_t n_tasks) {
    std::cout << "--- " << label << " ---\n";
    std::cout << "  global:         " << format_duration(res.global_time_ns) << "\n";

    auto per_data = compute_stats(res.per_data_ns);
    std::cout << "  per_data:       "
              << format_duration(per_data.mean) << " +- "
              << format_duration(per_data.std) << "  [min="
              << format_duration(per_data.min) << ", max="
              << format_duration(per_data.max) << "]\n";

    if (n_tasks > 1) {
        std::vector<double> between;
        between.reserve(res.per_data_ns.size());
        for (auto v : res.per_data_ns)
            between.push_back(v / static_cast<double>(n_tasks - 1));
        auto bt = compute_stats(between);
        std::cout << "  between_tasks:  "
                  << format_duration(bt.mean) << " +- "
                  << format_duration(bt.std) << "  [min="
                  << format_duration(bt.min) << ", max="
                  << format_duration(bt.max) << "]\n";
    }

    std::cout << "\n";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <N> [n_threads=1] [ndata=1]\n", argv[0]);
        printf("  N         number of tasks in the chain\n");
        printf("  n_threads threads per task (default 1)\n");
        printf("  ndata     batch size per iteration (default 1)\n");
        return 1;
    }

    size_t n_tasks = std::atol(argv[1]);
    size_t n_threads = argc > 2 ? std::atol(argv[2]) : 1;
    size_t ndata = argc > 3 ? std::atol(argv[3]) : 1;

    if (n_tasks < 1) {
        printf("error: N must be >= 1\n");
        return 1;
    }

    printf("=== Data Transfer Benchmark ===\n");
    printf("N=%zu tasks, n_threads=%zu, ndata=%zu, reps=%zu\n\n",
           n_tasks, n_threads, ndata, NUM_REPS);

    auto res_v1 = bench_v1_transfer(n_tasks, n_threads, ndata);
    print_results("Hedgehog v1", res_v1, n_tasks);

    auto res_v2 = bench_v2_transfer(n_tasks, n_threads, ndata);
    print_results("Hedgehog v2", res_v2, n_tasks);

    return 0;
}
