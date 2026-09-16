find_package(Threads REQUIRED)
find_package(BLAS REQUIRED)

add_executable(benchmark_v2 benchmark.cpp)
target_include_directories(benchmark_v2 PRIVATE ${CMAKE_SOURCE_DIR}/../hedgehog-v2)
target_link_libraries(benchmark_v2 Threads::Threads ${BLAS_LIBRARIES})
target_compile_definitions(benchmark_v2 PRIVATE HH_ENABLE_PROFILING)

add_executable(test_v2 test.cpp)
target_include_directories(test_v2 PRIVATE ${CMAKE_SOURCE_DIR}/../hedgehog-v2)
target_link_libraries(test_v2 Threads::Threads ${BLAS_LIBRARIES})

add_executable(test_v2_ptr test_pointer.cpp)
target_include_directories(test_v2_ptr PRIVATE ${CMAKE_SOURCE_DIR}/../hedgehog-v2)
target_link_libraries(test_v2_ptr Threads::Threads ${BLAS_LIBRARIES})
target_compile_definitions(test_v2_ptr PRIVATE HH_POINTER_MODE)

add_executable(test_v2_val test_value.cpp)
target_include_directories(test_v2_val PRIVATE ${CMAKE_SOURCE_DIR}/../hedgehog-v2)
target_link_libraries(test_v2_val Threads::Threads ${BLAS_LIBRARIES})
target_compile_definitions(test_v2_val PRIVATE HH_VALUE_MODE)

add_executable(benchmark_v2_ptr benchmark_ptr.cpp)
target_include_directories(benchmark_v2_ptr PRIVATE ${CMAKE_SOURCE_DIR}/../hedgehog-v2)
target_link_libraries(benchmark_v2_ptr Threads::Threads ${BLAS_LIBRARIES})
target_compile_definitions(benchmark_v2_ptr PRIVATE HH_POINTER_MODE HH_ENABLE_PROFILING)

add_executable(benchmark_v2_val benchmark_val.cpp)
target_include_directories(benchmark_v2_val PRIVATE ${CMAKE_SOURCE_DIR}/../hedgehog-v2)
target_link_libraries(benchmark_v2_val Threads::Threads ${BLAS_LIBRARIES})
target_compile_definitions(benchmark_v2_val PRIVATE HH_VALUE_MODE HH_ENABLE_PROFILING)
