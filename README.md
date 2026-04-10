# Measuring the cost of `cuda::atomic` on the DGX Spark

The DGX Spark has unified memory with hardware coherence via ATS (Address Translation Services). The CPU (ARM, 20 cores) and GPU (NVIDIA GB10, sm_121) share a single 128 GB memory pool. This benchmark measures the performance characteristics of that shared memory, with a focus on the cost of GPU-side atomics.

## Benchmarks

### 1. Bandwidth
Baseline read/write throughput to unified memory from both CPU and GPU. The GPU kernel streams through a 512 MB buffer with 512 blocks x 256 threads. CPU tests are run both single-threaded and multi-threaded across all 20 ARM cores via [parlay](https://github.com/cmuparlay/parlaylib). Also tests concurrent CPU+GPU access to the same shared buffer.

### 2. Latency (pointer chasing)
A random permutation cycle (Sattolo shuffle) over 16M entries. Both GPU (single-thread kernel) and CPU follow the chain for 1M hops.

### 3. Atomic throughput
A large grid (512 blocks x 256 threads) performs `fetch_add` on `uint64_t` via `cuda::atomic_ref` at block/device/system scope. Each thread loops 1000 iterations. Three contention levels:
- **Uncontended**: each thread hits a different element in a 1M-element array
- **Per-block**: all 256 threads in a block hit the same element
- **All-to-one**: all threads in the grid hit element 0

### 4. Atomic latency (single thread)
A single GPU thread performs 1M repeated operations on one location, measuring per-op latency for plain loads/stores, `fetch_add`, and `compare_exchange_strong` at each scope. Tested at two cache levels:
- **L2 (cache hit)**: repeated ops on a single address (hot in cache)
- **DRAM (cold cache lines)**: ops chasing through 16M cache-line-spaced elements via a random permutation, ensuring every op is a cache miss

## Build & run

```
make        # builds bench
make run    # builds and runs, writes results.csv
make clean
```

Requires CUDA 13.0+ and sm_121 (GB10).

## Results



### Bandwidth

| Test | Throughput |
|---|---|
| GPU read (malloc, ATS) | 153 GB/s |
| GPU write (malloc, ATS) | 138 GB/s |
| GPU read (cudaMalloc) | 243 GB/s |
| GPU write (cudaMalloc) | 196 GB/s |
| CPU read (1 thread) | 29 GB/s |
| CPU write (1 thread) | 81 GB/s |
| CPU read (parlay, 20 threads) | 128 GB/s |
| CPU write (parlay, 20 threads) | 128 GB/s |

#### Concurrent CPU+GPU (same shared malloc buffer)

| Scenario | GPU | CPU | Combined |
|---|---|---|---|
| GPU read + CPU read | 103 GB/s | 80 GB/s | 183 GB/s |
| GPU write + CPU write | 86 GB/s | 87 GB/s | 173 GB/s |
| GPU read + CPU write | 98 GB/s | 104 GB/s | 202 GB/s |
| GPU write + CPU read | 103 GB/s | 71 GB/s | 173 GB/s |

### Latency (pointer chase)

| Test | Latency |
|---|---|
| GPU (malloc, ATS) | 379 ns/hop |
| GPU (cudaMalloc) | 369 ns/hop |
| CPU | 92 ns/hop |

### Atomic throughput

| Operation | Uncontended | Per-block | All-to-one |
|---|---|---|---|
| plain_store | 27472 Gops/s | — | — |
| plain_load | 1662 Gops/s | — | — |
| fetch_add (block) | 72 Gops/s | 3.5 Gops/s | 69 Gops/s |
| fetch_add (device) | 72 Gops/s | 3.5 Gops/s | 69 Gops/s |
| fetch_add (system) | 73 Gops/s | 3.5 Gops/s | 69 Gops/s |

### Atomic latency (single thread)

#### L2 cache hit (single location)

| Operation | malloc (ATS) | cudaMalloc |
|---|---|---|
| plain_store | ~0 ns | ~0 ns |
| plain_load | 0.8 ns | 0.8 ns |
| fetch_add (block) | 42 ns | 42 ns |
| fetch_add (device) | 42 ns | 42 ns |
| fetch_add (system) | 42 ns | 42 ns |
| CAS (block) | 180 ns | 179 ns |
| CAS (device) | 180 ns | 179 ns |
| CAS (system) | 179 ns | 178 ns |

#### DRAM (cold cache lines, 16M-entry chase)

| Operation | Latency |
|---|---|
| plain_load | 447 ns |
| fetch_add (block) | 1135 ns |
| fetch_add (device) | 1137 ns |
| fetch_add (system) | 1137 ns |
| CAS (block) | 1526 ns |
| CAS (device) | 1528 ns |
| CAS (system) | 1524 ns |

<!-- ### Observations

- **ATS has a ~35% bandwidth penalty**: GPU read/write through ATS (malloc) is significantly slower than device-local memory (cudaMalloc). This is the main cost of hardware coherence.
- **ATS has negligible latency penalty**: pointer chase (~3%) and atomic ops (~0%) show no meaningful difference between malloc and cudaMalloc. The IOTLB translation cost is hidden by the memory access latency itself.
- **Concurrent CPU+GPU sharing costs ~30-45% per side**: when both CPU and GPU stream through the same buffer, each side loses significant bandwidth. Combined throughput (173-202 GB/s) exceeds either side alone, so the memory controller services both concurrently. GPU writes + CPU reads is the worst combination for the CPU (45% drop), likely due to coherence invalidations.
- **Scope has no effect**: block/device/system scopes produce identical results across all tests. With ATS hardware coherence, scope hints don't change the instruction path.
- **Cache-hit atomics are cheap, DRAM atomics are expensive**: `fetch_add` costs 42 ns in L2 but 1135 ns at DRAM (~27x). CAS costs 181 ns in L2 but 1526 ns at DRAM (~8.4x). A DRAM `fetch_add` is ~2.5x the raw DRAM read latency (447 ns); CAS is ~3.4x.
- **Per-block contention is the worst case** (~3.5 Gops/s vs ~72 Gops/s uncontended). All-to-one contention recovers to uncontended throughput, likely because the hardware atomic units can pipeline operations arriving from many SMs.
- **GPU memory latency is ~4.1x CPU latency** (379 vs 92 ns) for pointer chasing through system memory. -->
