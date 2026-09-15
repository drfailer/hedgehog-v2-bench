find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(benchmark_gpu_v2 benchmark.cpp)
target_include_directories(benchmark_gpu_v2 PRIVATE ${CMAKE_SOURCE_DIR}/../hedgehog-v2)
target_link_libraries(benchmark_gpu_v2 Threads::Threads CUDA::cudart CUDA::cublas)
target_compile_definitions(benchmark_gpu_v2 PRIVATE HH_ENABLE_PROFILING)

add_executable(test_gpu_v2 test.cpp)
target_include_directories(test_gpu_v2 PRIVATE ${CMAKE_SOURCE_DIR}/../hedgehog-v2)
target_link_libraries(test_gpu_v2 Threads::Threads CUDA::cudart CUDA::cublas)
