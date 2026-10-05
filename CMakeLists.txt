cmake_minimum_required(VERSION 3.16)

set(CMAKE_VERBOSE_MAKEFILE TRUE)

project(hedgehog_benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(cpu-dgemm/v1)
add_subdirectory(cpu-dgemm/v2)
add_subdirectory(cpu-dgemm/v2-tbb)
add_subdirectory(data_transfer)
add_subdirectory(gpu-dgemm/v1)
add_subdirectory(gpu-dgemm/v2)
add_subdirectory(gpu-dgemm-pipeline/v1)
add_subdirectory(gpu-dgemm-pipeline/v2)
