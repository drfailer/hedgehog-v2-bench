#include <memory>
#include <optional>
#include <hedgehog/hedgehog.h>
#include <chrono>
#include <string>
#include <vector>
#include "bench_common.hpp"

namespace {

struct PassthroughTask {
    using inputs = hh::type_list<TransferData>;
    using outputs = hh::type_list<TransferData>;

    static void execute(auto ctx, std::shared_ptr<TransferData> data) {
        ctx->push_result(std::move(data));
    }
};

struct MoodycamelCondTask {
    using inputs = hh::type_list<TransferData>;
    using outputs = hh::type_list<TransferData>;
    using node_input = hh::MoodycamelMPMCInput<TransferData>;

    static void execute(auto ctx, std::shared_ptr<TransferData> data) {
        ctx->push_result(std::move(data));
    }
};

struct MoodycamelAtomicTask {
    using inputs = hh::type_list<TransferData>;
    using outputs = hh::type_list<TransferData>;
    using node_input = hh::MoodycamelAtomicInput<TransferData>;

    static void execute(auto ctx, std::shared_ptr<TransferData> data) {
        ctx->push_result(std::move(data));
    }
};

template <typename TaskImpl>
auto make_transfer_graph(size_t n_tasks, size_t n_threads, std::string name) {
    auto graph = hh::make_graph<1, TransferData, TransferData>(name);

    auto first_task = hh::make_task<TaskImpl>(n_threads, "Task_0");
    using TaskNodeT = decltype(first_task);

    std::vector<TaskNodeT> tasks;
    tasks.reserve(n_tasks);
    tasks.push_back(std::move(first_task));
    for (size_t i = 1; i < n_tasks; ++i) {
        tasks.push_back(
            hh::make_task<TaskImpl>(n_threads, "Task_" + std::to_string(i)));
    }

    graph->connect_inputs(tasks[0]);
    for (size_t i = 0; i + 1 < n_tasks; ++i) {
        graph->draw_edges(tasks[i], tasks[i + 1]);
    }
    graph->connect_outputs(tasks[n_tasks - 1]);

    return graph;
}

template <typename TaskImpl>
BenchResult run_bench(size_t n_tasks, size_t n_threads, size_t ndata, std::string name) {
    using clock = std::chrono::high_resolution_clock;

    auto graph = make_transfer_graph<TaskImpl>(n_tasks, n_threads, name);
    graph->start();

    std::vector<double> per_data_ns;
    per_data_ns.reserve(NUM_REPS);

    auto t_global_start = clock::now();

    for (size_t rep = 0; rep < NUM_REPS; ++rep) {
        auto t1 = clock::now();
        for (size_t d = 0; d < ndata; ++d) {
            graph->push_data(
                std::make_shared<TransferData>(TransferData{d}));
        }
        for (size_t d = 0; d < ndata; ++d) {
            graph->get_result();
        }
        auto t2 = clock::now();
        double batch_ns =
            std::chrono::duration<double, std::nano>(t2 - t1).count();
        per_data_ns.push_back(batch_ns / static_cast<double>(ndata));
    }

    auto t_global_end = clock::now();
    double global_ns =
        std::chrono::duration<double, std::nano>(t_global_end - t_global_start)
            .count();

    graph->stop();

    return {global_ns, per_data_ns};
}

} // anonymous namespace

BenchResult bench_v2_transfer(size_t n_tasks, size_t n_threads, size_t ndata) {
    return run_bench<PassthroughTask>(n_tasks, n_threads, ndata, "v2 default");
}

BenchResult bench_v2_moodycamel_cond_transfer(size_t n_tasks, size_t n_threads, size_t ndata) {
    return run_bench<MoodycamelCondTask>(n_tasks, n_threads, ndata, "v2 moodycamel+cond");
}

BenchResult bench_v2_moodycamel_atomic_transfer(size_t n_tasks, size_t n_threads, size_t ndata) {
    return run_bench<MoodycamelAtomicTask>(n_tasks, n_threads, ndata, "v2 moodycamel+atomic");
}
