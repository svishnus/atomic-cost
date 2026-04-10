#include "common.cuh"

// ---------------------------------------------------------------------------
// Benchmark 3a: Atomic Throughput
// ---------------------------------------------------------------------------

// Templated throughput kernel: each thread runs `op(data, index)` in a loop.
template <typename OpFunc>
__global__ void atomic_tput_kernel(uint64_t *data, size_t n_elements, int iters,
                                   OpFunc op) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < iters; i++) {
    op(data, tid, n_elements);
  }
}

void bench_atomic_throughput() {
  printf("\n=== Atomic Throughput ===\n");

  uint64_t *buf = (uint64_t *)malloc(ATOMIC_ARRAY_N * sizeof(uint64_t));
  if (!buf) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  memset(buf, 0, ATOMIC_ARRAY_N * sizeof(uint64_t));

  size_t total_threads = (size_t)ATOMIC_TPUT_BLOCKS * ATOMIC_TPUT_THREADS;
  double total_ops = (double)total_threads * ATOMIC_ITERS_PER_THREAD;

  // Helper to run one variant
  auto run = [&](const char *name, const char *contention, auto op) {
    // Reset buffer
    memset(buf, 0, ATOMIC_ARRAY_N * sizeof(uint64_t));

    for (int i = 0; i < ATOMIC_WARMUP; i++) {
      atomic_tput_kernel<<<ATOMIC_TPUT_BLOCKS, ATOMIC_TPUT_THREADS>>>(
          buf, ATOMIC_ARRAY_N, ATOMIC_ITERS_PER_THREAD, op);
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    GpuTimer timer;
    double total_ms = 0;
    for (int i = 0; i < ATOMIC_ITERS; i++) {
      timer.begin();
      atomic_tput_kernel<<<ATOMIC_TPUT_BLOCKS, ATOMIC_TPUT_THREADS>>>(
          buf, ATOMIC_ARRAY_N, ATOMIC_ITERS_PER_THREAD, op);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_s = (total_ms / ATOMIC_ITERS) / 1000.0;
    double gops = total_ops / avg_s / 1e9;
    printf("  %-30s [%-12s]: %8.2f Gops/s\n", name, contention, gops);
    std::string vstr = std::string(name) + "(" + contention + ")";
    g_results.push_back({"atomic_tput", vstr, gops, "Gops/s"});
  };

  // --- Uncontended: each thread hits a different element ---
  printf("  -- Uncontended --\n");

  run("plain_store", "uncontended",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        *reinterpret_cast<volatile uint64_t *>(&data[tid % n]) = tid;
      });

  run("plain_load", "uncontended",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        volatile uint64_t v =
            *reinterpret_cast<volatile uint64_t *>(&data[tid % n]);
        (void)v;
      });

  run("fetch_add_block", "uncontended",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(data[tid % n]);
        ref.fetch_add(1, cuda::memory_order_relaxed);
      });

  run("fetch_add_device", "uncontended",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(
            data[tid % n]);
        ref.fetch_add(1, cuda::memory_order_relaxed);
      });

  run("fetch_add_system", "uncontended",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(
            data[tid % n]);
        ref.fetch_add(1, cuda::memory_order_relaxed);
      });

  // ! Throw CPU into the mix with the cuda atomic

  // --- Per-block contended: all threads in a block hit the same element ---
  printf("  -- Per-block contended --\n");

  run("fetch_add_block", "per_block",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(
            data[blockIdx.x % n]);
        ref.fetch_add(1, cuda::memory_order_relaxed);
      });

  run("fetch_add_device", "per_block",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(
            data[blockIdx.x % n]);
        ref.fetch_add(1, cuda::memory_order_relaxed);
      });

  run("fetch_add_system", "per_block",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(
            data[blockIdx.x % n]);
        ref.fetch_add(1, cuda::memory_order_relaxed);
      });

  // --- All-to-one: maximum contention, all threads hit element 0 ---
  printf("  -- All-to-one contended --\n");

  run("fetch_add_block", "all_to_one",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(data[0]);
        ref.fetch_add(1, cuda::memory_order_relaxed);
      });

  run("fetch_add_device", "all_to_one",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(data[0]);
        ref.fetch_add(1, cuda::memory_order_relaxed);
      });

  run("fetch_add_system", "all_to_one",
      [] __device__(uint64_t * data, size_t tid, size_t n) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(data[0]);
        ref.fetch_add(1, cuda::memory_order_relaxed);
      });

  free(buf);
}

// ---------------------------------------------------------------------------
// Benchmark 3a': Atomic Throughput (DRAM, 4x L2)
// ---------------------------------------------------------------------------

// Strided kernel: each iteration hits a different element, spaced by
// total_threads, so successive ops go to different cache lines.
template <typename OpFunc>
__global__ void atomic_tput_strided_kernel(uint64_t *data, size_t n_elements,
                                           int iters, OpFunc op) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (int i = 0; i < iters; i++) {
    op(data, tid + (size_t)i * stride, n_elements);
  }
}

void bench_atomic_throughput_dram() {
  printf("\n=== Atomic Throughput (DRAM, 4x L2) ===\n");

  size_t dram_n = (size_t)g_l2CacheSize * 4 / sizeof(uint64_t);
  size_t dram_bytes = dram_n * sizeof(uint64_t);

  uint64_t *buf = (uint64_t *)malloc(dram_bytes);
  if (!buf) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  memset(buf, 0, dram_bytes);

  size_t total_threads = (size_t)ATOMIC_TPUT_BLOCKS * ATOMIC_TPUT_THREADS;
  double total_ops = (double)total_threads * ATOMIC_ITERS_PER_THREAD;

  auto run = [&](const char *name, auto op) {
    memset(buf, 0, dram_bytes);
    for (int i = 0; i < ATOMIC_WARMUP; i++) {
      atomic_tput_strided_kernel<<<ATOMIC_TPUT_BLOCKS, ATOMIC_TPUT_THREADS>>>(
          buf, dram_n, ATOMIC_ITERS_PER_THREAD, op);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    GpuTimer timer;
    double total_ms = 0;
    for (int i = 0; i < ATOMIC_ITERS; i++) {
      timer.begin();
      atomic_tput_strided_kernel<<<ATOMIC_TPUT_BLOCKS, ATOMIC_TPUT_THREADS>>>(
          buf, dram_n, ATOMIC_ITERS_PER_THREAD, op);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_s = (total_ms / ATOMIC_ITERS) / 1000.0;
    double gops = total_ops / avg_s / 1e9;
    printf("  %-30s: %8.2f Gops/s\n", name, gops);
    g_results.push_back({"atomic_tput_dram", name, gops, "Gops/s"});
  };

  run("plain_store", [] __device__(uint64_t * data, size_t idx, size_t n) {
    *reinterpret_cast<volatile uint64_t *>(&data[idx % n]) = idx;
  });

  run("plain_load", [] __device__(uint64_t * data, size_t idx, size_t n) {
    volatile uint64_t v =
        *reinterpret_cast<volatile uint64_t *>(&data[idx % n]);
    (void)v;
  });

  run("fetch_add_block", [] __device__(uint64_t * data, size_t idx, size_t n) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(data[idx % n]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run("fetch_add_device", [] __device__(uint64_t * data, size_t idx, size_t n) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(data[idx % n]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run("fetch_add_system", [] __device__(uint64_t * data, size_t idx, size_t n) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(data[idx % n]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  free(buf);
}
