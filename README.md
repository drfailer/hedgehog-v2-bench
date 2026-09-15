# Hedgehog v2 Tutorial 4 — Blocked DGEMM

Port of the hedgehog tutorial4 (blocked matrix multiplication, C += A*B) to hedgehog-v2.

## Prerequisites

- GCC 15 (`personal-g++` / `personal-gcc`)
- OpenBLAS (expected at `~/Programming/usr/`)
- hedgehog-v2 (expected at `../hedgehog-v2/`)
- hedgehog (v1 fork, expected at `../hedgehog-fork/`) — only needed for the v1 benchmark
- hedgehog-Tutorials (expected at `../hedgehog-Tutorials/`) — only needed for the v1 benchmark

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
```

This produces three targets:

| Binary | Description |
|---|---|
| `build/tutorial4` | Correctness test (small matrices, verification against naive matmul) |
| `build/benchmark` | Hedgehog-v2 benchmark |
| `build/benchmark_v1` | Hedgehog-v1 benchmark (same algorithm, original framework) |

## Running the correctness test

```bash
./build/tutorial4
```

Runs a small DGEMM (10x11 * 11x12) and verifies against a naive implementation. Prints `PASS` or `FAIL`.

## Benchmark

### Quick run

```bash
./bench.sh
```

Runs both v1 and v2 at 10k and 20k square matrices with block size 1024, 40 product threads, and 10 addition threads.

### Manual run

Both benchmark binaries accept the same arguments:

```bash
./build/benchmark   [n] [block_size] [product_threads] [addition_threads]
./build/benchmark_v1 [n] [block_size] [product_threads] [addition_threads]
```

Defaults: `n=10000 block_size=1024 product_threads=40 addition_threads=10`

Examples:

```bash
# 10k x 10k, block size 1024, 40 product / 10 addition threads
./build/benchmark 10000 1024 40 10

# 20k x 20k
./build/benchmark 20000 1024 40 10
```

### Tuning tips

- **Block size**: 1024 typically fits in L3 cache and gives good BLAS throughput. Smaller values (e.g. 256) increase scheduling overhead.
- **Product threads**: Set to core count (e.g. 40). Oversubscription works well with this algorithm.
- **Addition threads**: 10 is generally a good value; addition is cheap relative to the BLAS calls.
