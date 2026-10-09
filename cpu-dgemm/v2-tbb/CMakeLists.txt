cmake_minimum_required(VERSION 3.20)
project(benchmark_v2_tbb CXX)

include(FetchContent)

find_package(Threads REQUIRED)
find_package(BLAS REQUIRED)

find_package(TBB QUIET)
if (NOT TBB_FOUND)
    message(STATUS "TBB not found on system, fetching via FetchContent...")
    set(BUILD_SHARED_LIBS ON)
    set(TBB_TEST OFF CACHE BOOL "" FORCE)
    set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(TBB_STRICT OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        onetbb
        GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB.git
        GIT_TAG        v2023.1.0
        GIT_SHALLOW    TRUE
        OVERRIDE_FIND_PACKAGE
    )
    FetchContent_MakeAvailable(onetbb)
    if (TARGET tbb AND NOT TARGET TBB::tbb)
        add_library(TBB::tbb ALIAS tbb)
    endif()
endif()

add_executable(benchmark_v2_tbb benchmark_tbb.cpp)
target_include_directories(benchmark_v2_tbb PRIVATE ${CMAKE_SOURCE_DIR}/../hedgehog-v2)
target_link_libraries(benchmark_v2_tbb Threads::Threads ${BLAS_LIBRARIES} TBB::tbb)
target_compile_definitions(benchmark_v2_tbb PRIVATE HH_ENABLE_TBB HH_ENABLE_PROFILING)
