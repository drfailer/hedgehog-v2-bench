include(FetchContent)

FetchContent_Declare(
    hedgehog_v1
    GIT_REPOSITORY https://github.com/drfailer/hedgehog.git
    GIT_TAG master
)
FetchContent_MakeAvailable(hedgehog_v1)

find_package(Threads REQUIRED)

add_library(bench_v1_obj OBJECT bench_v1.cpp)
target_include_directories(bench_v1_obj PRIVATE ${hedgehog_v1_SOURCE_DIR})
target_link_libraries(bench_v1_obj Threads::Threads)

add_library(bench_v2_obj OBJECT bench_v2.cpp)
target_include_directories(bench_v2_obj PRIVATE ${CMAKE_SOURCE_DIR}/../hedgehog-v2)
target_link_libraries(bench_v2_obj Threads::Threads)

add_executable(benchmark_transfer benchmark_transfer.cpp)
target_link_libraries(benchmark_transfer bench_v1_obj bench_v2_obj Threads::Threads)
