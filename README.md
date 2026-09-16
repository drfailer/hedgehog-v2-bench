# Hedgehog v2 Blocked DGEMM

Port of the hedgehog tutorial4/5/6 (blocked matrix multiplication, C += A*B) to hedgehog-v2.

- tutorial 4: CPU dgemm using OpenBlas.
- tutorial 5: GPU dgemm using cublas.
- tutorial 6: Multi-GPU dgemm using cublas and hedgehog pipelines.

## Prerequisites

- GCC 15 (`personal-g++` / `personal-gcc`)
- OpenBLAS
- hedgehog-v2 (expected at `../hedgehog-v2/`)

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
```

#4 Run

Both benchmark binaries accept the same arguments:

```bash
./build/cpu-dgemm/v1/benchmark_v1 [n] [block_size] [product_threads] [addition_threads]
./build/cpu-dgemm/v2/benchmark_v2 [n] [block_size] [product_threads] [addition_threads]
```

Defaults: `n=10000 block_size=1024 product_threads=40 addition_threads=10`
