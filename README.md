# Measuring the cost of `cuda::atomic` on the DGX Spark

The DGX Spark has unified memory with hardware coherence via ATS (Address Translation Services). The CPU (ARM, 20 cores) and GPU (NVIDIA GB10, sm_121) share a single 128 GB memory pool. This benchmark measures the performance characteristics of that shared memory, with a focus on the cost of GPU-side atomics.

## Benchmarks

### 1. Bandwidth
Baseline read/write throughput to unified memory from both CPU and GPU. The GPU kernel streams through a 512 MB buffer with 512 blocks x 256 threads. CPU tests are run both single-threaded and multi-threaded across all 20 ARM cores via [parlay](https://github.com/cmuparlay/parlaylib).

### 2. Latency (pointer chasing)
A random permutation cycle (Sattolo shuffle) over 16M entries. Both GPU (single-thread kernel) and CPU follow the chain for 1M hops.

### 3. Atomic throughput
A large grid (512 blocks x 256 threads) performs `fetch_add` on `uint64_t` via `cuda::atomic_ref` at block/device/system scope. Each thread loops 1000 iterations. Three contention levels:
- **Uncontended**: each thread hits a different element in a 1M-element array
- **Per-block**: all 256 threads in a block hit the same element
- **All-to-one**: all threads in the grid hit element 0

### 4. Atomic latency (single thread)
A single GPU thread performs 1M repeated operations on one location, measuring per-op latency for plain loads/stores, `fetch_add`, and `compare_exchange_strong` at each scope.

## Build & run

```
make        # builds bench
make run    # builds and runs, writes results.csv
make clean
```

Requires CUDA 13.0+ and sm_121 (GB10).

## Results

Measured on DGX Spark (NVIDIA GB10, CUDA 13.0, ARM 20-core CPU).

### Bandwidth

| Test | Throughput |
|---|---|
| GPU read | 147 GB/s |
| GPU write | 137 GB/s |
| CPU read (1 thread) | 29 GB/s |
| CPU write (1 thread) | 81 GB/s |
| CPU read (parlay, 20 threads) | 128 GB/s |
| CPU write (parlay, 20 threads) | 134 GB/s |


### Latency (pointer chase)

| Test | Latency |
|---|---|
| GPU | 379 ns/hop |
| CPU | 88 ns/hop |

### Atomic throughput

| Operation | Uncontended | Per-block | All-to-one |
|---|---|---|---|
| plain_store | 28171 Gops/s | — | — |
| plain_load | 1744 Gops/s | — | — |
| fetch_add (block) | 70 Gops/s | 3.4 Gops/s | 69 Gops/s |
| fetch_add (device) | 70 Gops/s | 3.4 Gops/s | 69 Gops/s |
| fetch_add (system) | 69 Gops/s | 3.5 Gops/s | 69 Gops/s |

### Atomic latency (single thread, no contention)

| Operation | Latency |
|---|---|
| plain_store | ~0 ns |
| plain_load | 0.8 ns |
| fetch_add (block) | 42 ns |
| fetch_add (device) | 42 ns |
| fetch_add (system) | 42 ns |
| CAS (block) | 176 ns |
| CAS (device) | 176 ns |
| CAS (system) | 175 ns |

### Observations

- **Scope has no effect**: block/device/system scopes produce identical results across all tests. With ATS hardware coherence, scope hints don't change the instruction path.
- **`fetch_add` costs ~42 ns**, roughly 50x slower than a plain load. CAS costs ~176 ns (4x more than `fetch_add`).
- **Per-block contention is the worst case** (~3.4 Gops/s vs ~70 Gops/s uncontended). All-to-one contention recovers to uncontended throughput, likely because the hardware atomic units can pipeline operations arriving from many SMs.
- **GPU memory latency is ~4x CPU latency** (379 vs 88 ns) for pointer chasing through system memory.
