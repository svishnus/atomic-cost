# Measuring the cost of `cuda::atomic` on the DGX Spark

The DGX Spark has unified memory with hardware coherence via ATS (Address Translation Services). The CPU (ARM, 20 cores) and GPU (NVIDIA GB10, sm_121) share a single 128 GB memory pool. This benchmark measures the performance characteristics of that shared memory, with a focus on the cost of GPU-side atomics.

## Benchmarks

### 1. Bandwidth
Baseline read/write throughput to unified memory from both CPU and GPU. The GPU kernel streams through a 512 MB buffer with 512 blocks × 256 threads. CPU tests are run both single-threaded and multi-threaded across all 20 ARM cores via [parlay](https://github.com/cmuparlay/parlaylib). Also tests concurrent CPU+GPU access to the same shared buffer.

### 2. Latency (pointer chasing)
A random permutation cycle (Sattolo shuffle) sized to 4x the GPU L2 cache to guarantee cold misses. Both GPU (single-thread kernel) and CPU follow the chain for 1M hops.

### 3. Atomic throughput
A large grid (512 blocks × 256 threads) performs `fetch_add` on `uint64_t` via `cuda::atomic_ref` at block/device/system scope. Each thread loops 1000 iterations. Two array sizes: L2-resident (1M elements, 8 MB) and DRAM-resident (4× L2, ~96 MB with strided access). Three contention levels (L2 only):
- **Uncontended**: each thread hits a different element
- **Per-block**: all 256 threads in a block hit the same element
- **All-to-one**: all threads in the grid hit element 0

### 4. Atomic latency (single thread)
A single GPU thread performs 1M repeated operations on one location, measuring per-op latency for plain loads/stores, `fetch_add`, and `compare_exchange_strong` at each scope. Tested at two cache levels:
- **L2 (cache hit)**: repeated ops on a single address (hot in cache)
- **DRAM (cold cache lines)**: ops chasing through 4× L2 cache-line-spaced elements via a random permutation, ensuring every op is a cache miss

## Build & run

```
make        # builds bench
make run    # builds and runs, writes results.csv
make clean
```

Requires CUDA 13.0+ and sm_121 (GB10).

## Results

NVIDIA GB10 (sm_121, 48 SMs, 24 MB L2), ARM Grace (20 cores), 128 GB unified memory.

### Bandwidth

| Test | Throughput |
|---|---|
| GPU read (malloc, ATS) | 157 GB/s |
| GPU write (malloc, ATS) | 133 GB/s |
| GPU read (cudaMalloc) | 240 GB/s |
| GPU write (cudaMalloc) | 195 GB/s |
| CPU read (1 thread, NEON) | 34 GB/s |
| CPU write (1 thread, memset) | 81 GB/s |
| CPU read (parlay, 20 threads) | 127 GB/s |
| CPU write (parlay, 20 threads) | 132 GB/s |

#### Concurrent CPU+GPU (split halves of shared malloc buffer)

| Scenario | GPU | CPU | Combined |
|---|---|---|---|
| GPU read + CPU read | 98 GB/s | 78 GB/s | 176 GB/s |
| GPU write + CPU write | 82 GB/s | 85 GB/s | 167 GB/s |
| GPU read + CPU write | 96 GB/s | 109 GB/s | 205 GB/s |
| GPU write + CPU read | 101 GB/s | 72 GB/s | 173 GB/s |

### Latency (pointer chase)

| Test | Latency |
|---|---|
| GPU (malloc, ATS) | 404 ns/hop |
| GPU (cudaMalloc) | 385 ns/hop |
| CPU | 81–102 ns/hop |

### Atomic throughput (L2-resident)

| Operation | Uncontended | Per-block | All-to-one |
|---|---|---|---|
| plain_store | 792 Gops/s | — | — |
| plain_load | 178 Gops/s | — | — |
| fetch_add (block) | 66 Gops/s | 3.4 Gops/s | 69 Gops/s |
| fetch_add (device) | 66 Gops/s | 3.4 Gops/s | 69 Gops/s |
| fetch_add (system) | 65 Gops/s | 3.5 Gops/s | 69 Gops/s |

### Atomic throughput (DRAM-resident, 4× L2)

| Operation | Throughput |
|---|---|
| plain_store | 17.3 Gops/s |
| plain_load | 19.9 Gops/s |
| fetch_add (block) | 12.9 Gops/s |
| fetch_add (device) | 13.1 Gops/s |
| fetch_add (system) | 13.0 Gops/s |

### Atomic latency (single thread)

#### L2 cache hit (single location)

| Operation | malloc (ATS) | cudaMalloc |
|---|---|---|
| plain_store | 3.7 ns | 3.7 ns |
| plain_load | 20 ns | 19 ns |
| fetch_add (block) | 42 ns | 42 ns |
| fetch_add (device) | 42 ns | 42 ns |
| fetch_add (system) | 42 ns | 42 ns |
| CAS (block) | 32 ns | 32 ns |
| CAS (device) | 32 ns | 32 ns |
| CAS (system) | 32 ns | 32 ns |

#### DRAM (cold cache lines, 4× L2 chase)

| Operation | Latency |
|---|---|
| plain_load | 184 ns |
| fetch_add (block) | 433 ns |
| fetch_add (device) | 433 ns |
| fetch_add (system) | 433 ns |
| CAS (block) | 813 ns |
| CAS (device) | 813 ns |
| CAS (system) | 813 ns |

### Observations

- **ATS has a ~35% bandwidth penalty**: GPU read through ATS (157 GB/s) vs cudaMalloc (240 GB/s). Write penalty is ~32% (133 vs 195 GB/s). This is the cost of address translation through the IOMMU.
- **ATS latency penalty is ~5%**: pointer chase 404 ns (ATS) vs 385 ns (cudaMalloc). The IOTLB translation cost is mostly hidden by DRAM latency.
- **Scope has no effect**: block/device/system scopes produce identical results across all tests. With ATS hardware coherence, scope hints don't change the instruction path on GB10.
- **fetch_add appears 10 ns slower than CAS at L2**: 42 ns vs 32 ns. This is **not** a difference in L2 atomic unit cost — it's warp-aggregation overhead. `ptxas` automatically inserts a warp-reduction preamble (~10 instructions: `VOTEU`, `UFLO`, `UPOPC`, etc.) around every `ATOM.ADD` at the SASS level to coalesce adds from active lanes in a warp. CAS cannot be aggregated (each has a unique expected value) so it skips this preamble. Confirmed by implementing fetch_add via a CAS loop, which measures 32 ns — identical to CAS. The true L2 atomic RMW cost is **32 ns** for both operations.
- **DRAM CAS costs 2× DRAM atomic round-trip (813 ns)**: a CAS-with-retry on a cold cache line requires two trips through the L2 atomic unit — the first attempt (expected=0) fails and fetches the line from DRAM (~407 ns), the retry succeeds but also costs a full round-trip because the L2 atomic unit does not cache data between operations. fetch_add needs only one round-trip (433 ns).
- **DRAM atomics cost ~2.4× raw DRAM read latency**: fetch_add at 433 ns vs plain_load at 184 ns. The overhead is the L2 atomic unit's read-modify-write processing.
- **Per-block contention is the worst case**: ~3.4 Gops/s vs ~66 Gops/s uncontended (19× slower). All-to-one recovers to ~69 Gops/s — the L2 atomic units can pipeline operations from many SMs to a single address more efficiently than serializing 256 threads within one SM.
- **L2 → DRAM throughput cliff**: plain_store drops 43× (792 → 17 Gops/s), plain_load drops 9× (178 → 20 Gops/s), fetch_add drops 5× (66 → 13 Gops/s). Stores suffer most because L2 can absorb fire-and-forget stores at register speed, but DRAM stores require write-allocate round-trips.
- **GPU memory latency is ~4–5× CPU latency**: 404 vs 81–102 ns for pointer chasing through unified memory.
- **Concurrent CPU+GPU saturates ~170–205 GB/s combined**: both sides lose bandwidth individually but total throughput exceeds either alone.
