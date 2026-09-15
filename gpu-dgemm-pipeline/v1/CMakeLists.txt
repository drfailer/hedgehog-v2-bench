include(FetchContent)

FetchContent_Declare(
    hedgehog_v1
    GIT_REPOSITORY https://github.com/drfailer/hedgehog.git
    GIT_TAG master
)
FetchContent_MakeAvailable(hedgehog_v1)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(benchmark_gpu_pipeline_v1 benchmark_v1.cpp)
target_include_directories(benchmark_gpu_pipeline_v1 PRIVATE ${hedgehog_v1_SOURCE_DIR})
target_link_libraries(benchmark_gpu_pipeline_v1 Threads::Threads CUDA::cudart CUDA::cublas)
target_compile_definitions(benchmark_gpu_pipeline_v1 PRIVATE HH_USE_CUDA)
