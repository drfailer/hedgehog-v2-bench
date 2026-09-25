#include <hedgehog/hedgehog.h>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "bench_common.hpp"

namespace {

class PassthroughTask
    : public hh::AbstractTask<1, TransferData, TransferData> {
public:
    explicit PassthroughTask(size_t n_threads)
        : hh::AbstractTask<1, TransferData, TransferData>(
              "Passthrough", n_threads) {}

    void execute(std::shared_ptr<TransferData> data) override {
        this->addResult(std::move(data));
    }

    std::shared_ptr<hh::AbstractTask<1, TransferData, TransferData>>
    copy() override {
        return std::make_shared<PassthroughTask>(this->numberThreads());
    }
};

} // anonymous namespace

BenchResult bench_v1_transfer(size_t n_tasks, size_t n_threads, size_t ndata) {
    using clock = std::chrono::high_resolution_clock;

    std::vector<std::shared_ptr<PassthroughTask>> tasks;
    tasks.reserve(n_tasks);
    for (size_t i = 0; i < n_tasks; ++i) {
        tasks.push_back(std::make_shared<PassthroughTask>(n_threads));
    }

    hh::Graph<1, TransferData, TransferData> graph("Transfer v1");

    graph.inputs(tasks[0]);
    for (size_t i = 0; i + 1 < n_tasks; ++i) {
        graph.edges(tasks[i], tasks[i + 1]);
    }
    graph.outputs(tasks[n_tasks - 1]);

    graph.executeGraph(true);

    std::vector<double> per_data_ns;
    per_data_ns.reserve(NUM_REPS);

    auto t_global_start = clock::now();

    for (size_t rep = 0; rep < NUM_REPS; ++rep) {
        auto t1 = clock::now();
        for (size_t d = 0; d < ndata; ++d) {
            graph.pushData(std::make_shared<TransferData>(TransferData{d}));
        }
        for (size_t d = 0; d < ndata; ++d) {
            graph.getBlockingResult();
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

    graph.finishPushingData();
    graph.waitForTermination();

    return {global_ns, per_data_ns};
}
