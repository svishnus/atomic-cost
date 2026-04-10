#include "common.cuh"

// ---------------------------------------------------------------------------
// Benchmark 3b: Atomic Latency (single thread)
// ---------------------------------------------------------------------------

template <typename OpFunc>
__global__ void atomic_lat_kernel(uint64_t *data, int ops, OpFunc op) {
  for (int i = 0; i < ops; i++) {
    op(data);
  }
}

// CAS variant: passes mutable state through the loop so that each CAS
// succeeds (single-threaded, expected tracks the current value).
template <typename OpFunc>
__global__ void atomic_lat_cas_kernel(uint64_t *data, int ops, OpFunc op) {
  uint64_t state = 0;
  for (int i = 0; i < ops; i++) {
    op(data, state);
  }
}

// Chase-based atomic latency kernel: follows a pointer chain through
// cache-line-spaced elements, performing an atomic at each hop.
// chase[i] gives the index of the next element to visit.
// Each element is spaced by CACHE_LINE_ELEMS so every hop is a new cache line.
template <typename OpFunc>
__global__ void atomic_lat_chase_kernel(uint64_t *data,
                                        const uint32_t *__restrict__ chase,
                                        int hops, OpFunc op) {
  uint32_t idx = 0;
  for (int i = 0; i < hops; i++) {
    op(&data[idx * CACHE_LINE_ELEMS]);
    idx = chase[idx];
  }
}

void bench_atomic_latency() {
  printf("\n=== Atomic Latency (single thread) ===\n");

  uint64_t *buf = (uint64_t *)malloc(sizeof(uint64_t));
  if (!buf) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  *buf = 0;

  auto run = [&](const char *name, auto op) {
    *buf = 0;
    for (int i = 0; i < ATOMIC_WARMUP; i++) {
      atomic_lat_kernel<<<1, 1>>>(buf, ATOMIC_LAT_OPS, op);
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    GpuTimer timer;
    double total_ms = 0;
    for (int i = 0; i < ATOMIC_ITERS; i++) {
      timer.begin();
      atomic_lat_kernel<<<1, 1>>>(buf, ATOMIC_LAT_OPS, op);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_ns = (total_ms / ATOMIC_ITERS) * 1e6 / ATOMIC_LAT_OPS;
    printf("  %-30s: %8.1f ns/op\n", name, avg_ns);
    g_results.push_back({"atomic_lat", name, avg_ns, "ns"});
  };

  auto run_cas = [&](const char *name, auto op) {
    *buf = 0;
    for (int i = 0; i < ATOMIC_WARMUP; i++) {
      atomic_lat_cas_kernel<<<1, 1>>>(buf, ATOMIC_LAT_OPS, op);
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    GpuTimer timer;
    double total_ms = 0;
    for (int i = 0; i < ATOMIC_ITERS; i++) {
      timer.begin();
      atomic_lat_cas_kernel<<<1, 1>>>(buf, ATOMIC_LAT_OPS, op);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_ns = (total_ms / ATOMIC_ITERS) * 1e6 / ATOMIC_LAT_OPS;
    printf("  %-30s: %8.1f ns/op\n", name, avg_ns);
    g_results.push_back({"atomic_lat", name, avg_ns, "ns"});
  };

  run("plain_store", [] __device__(uint64_t * data) {
    *reinterpret_cast<volatile uint64_t *>(data) = 1;
  });

  run("plain_load", [] __device__(uint64_t * data) {
    volatile uint64_t v = *reinterpret_cast<volatile uint64_t *>(data);
    (void)v;
  });

  run("fetch_add_block", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(data[0]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run("fetch_add_device", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(data[0]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run("fetch_add_system", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(data[0]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run_cas("cas_block", [] __device__(uint64_t * data, uint64_t & state) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(data[0]);
    uint64_t desired = state + 1;
    ref.compare_exchange_strong(state, desired, cuda::memory_order_relaxed);
    state = desired;
  });

  run_cas("cas_device", [] __device__(uint64_t * data, uint64_t & state) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(data[0]);
    uint64_t desired = state + 1;
    ref.compare_exchange_strong(state, desired, cuda::memory_order_relaxed);
    state = desired;
  });

  run_cas("cas_system", [] __device__(uint64_t * data, uint64_t & state) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(data[0]);
    uint64_t desired = state + 1;
    ref.compare_exchange_strong(state, desired, cuda::memory_order_relaxed);
    state = desired;
  });

  free(buf);

  // --- Atomic latency with cudaMalloc (no ATS) ---
  printf("\n=== Atomic Latency single thread (cudaMalloc, no ATS) ===\n");

  uint64_t *dbuf;
  CHECK_CUDA(cudaMalloc(&dbuf, sizeof(uint64_t)));
  CHECK_CUDA(cudaMemset(dbuf, 0, sizeof(uint64_t)));

  auto run_dev = [&](const char *name, auto op) {
    CHECK_CUDA(cudaMemset(dbuf, 0, sizeof(uint64_t)));
    for (int i = 0; i < ATOMIC_WARMUP; i++) {
      atomic_lat_kernel<<<1, 1>>>(dbuf, ATOMIC_LAT_OPS, op);
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    GpuTimer timer;
    double total_ms = 0;
    for (int i = 0; i < ATOMIC_ITERS; i++) {
      timer.begin();
      atomic_lat_kernel<<<1, 1>>>(dbuf, ATOMIC_LAT_OPS, op);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_ns = (total_ms / ATOMIC_ITERS) * 1e6 / ATOMIC_LAT_OPS;
    printf("  %-30s: %8.1f ns/op\n", name, avg_ns);
    g_results.push_back({"atomic_lat_dev", name, avg_ns, "ns"});
  };

  auto run_dev_cas = [&](const char *name, auto op) {
    CHECK_CUDA(cudaMemset(dbuf, 0, sizeof(uint64_t)));
    for (int i = 0; i < ATOMIC_WARMUP; i++) {
      atomic_lat_cas_kernel<<<1, 1>>>(dbuf, ATOMIC_LAT_OPS, op);
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    GpuTimer timer;
    double total_ms = 0;
    for (int i = 0; i < ATOMIC_ITERS; i++) {
      timer.begin();
      atomic_lat_cas_kernel<<<1, 1>>>(dbuf, ATOMIC_LAT_OPS, op);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_ns = (total_ms / ATOMIC_ITERS) * 1e6 / ATOMIC_LAT_OPS;
    printf("  %-30s: %8.1f ns/op\n", name, avg_ns);
    g_results.push_back({"atomic_lat_dev", name, avg_ns, "ns"});
  };

  run_dev("plain_store", [] __device__(uint64_t * data) {
    *reinterpret_cast<volatile uint64_t *>(data) = 1;
  });

  run_dev("plain_load", [] __device__(uint64_t * data) {
    volatile uint64_t v = *reinterpret_cast<volatile uint64_t *>(data);
    (void)v;
  });

  run_dev("fetch_add_block", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(data[0]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run_dev("fetch_add_device", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(data[0]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run_dev("fetch_add_system", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(data[0]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run_dev_cas("cas_block", [] __device__(uint64_t * data, uint64_t & state) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(data[0]);
    uint64_t desired = state + 1;
    ref.compare_exchange_strong(state, desired, cuda::memory_order_relaxed);
    state = desired;
  });

  run_dev_cas("cas_device", [] __device__(uint64_t * data, uint64_t & state) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(data[0]);
    uint64_t desired = state + 1;
    ref.compare_exchange_strong(state, desired, cuda::memory_order_relaxed);
    state = desired;
  });

  run_dev_cas("cas_system", [] __device__(uint64_t * data, uint64_t & state) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(data[0]);
    uint64_t desired = state + 1;
    ref.compare_exchange_strong(state, desired, cuda::memory_order_relaxed);
    state = desired;
  });

  CHECK_CUDA(cudaFree(dbuf));

  // --- Atomic latency at DRAM (chase through cold cache lines) ---
  printf("\n=== Atomic Latency single thread (DRAM, cold cache lines) ===\n");

  size_t dram_n_lines =
      (size_t)g_l2CacheSize * 4 / CACHE_LINE_BYTES; // 4x L2 in cache lines
  size_t dram_buf_elems = dram_n_lines * CACHE_LINE_ELEMS;
  size_t dram_buf_bytes = dram_buf_elems * sizeof(uint64_t);

  uint64_t *dram_buf = (uint64_t *)malloc(dram_buf_bytes);
  if (!dram_buf) {
    fprintf(stderr, "malloc failed for DRAM atomic buffer\n");
    exit(1);
  }
  memset(dram_buf, 0, dram_buf_bytes);

  // Build a chase permutation over cache-line indices
  uint32_t *chase = (uint32_t *)malloc(dram_n_lines * sizeof(uint32_t));
  if (!chase) {
    fprintf(stderr, "malloc failed for chase array\n");
    exit(1);
  }
  generate_chase(chase, dram_n_lines);

  // Copy chase to device memory so chasing the index array itself doesn't
  // pollute the measurement (chase fits in L2, data buffer doesn't).
  uint32_t *d_chase;
  CHECK_CUDA(cudaMalloc(&d_chase, dram_n_lines * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemcpy(d_chase, chase, dram_n_lines * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  int dram_hops = LAT_HOPS; // 1M hops

  auto run_dram = [&](const char *name, auto op) {
    memset(dram_buf, 0, dram_buf_bytes);

    for (int i = 0; i < ATOMIC_WARMUP; i++) {
      atomic_lat_chase_kernel<<<1, 1>>>(dram_buf, d_chase, dram_hops, op);
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    GpuTimer timer;
    double total_ms = 0;
    for (int i = 0; i < ATOMIC_ITERS; i++) {
      timer.begin();
      atomic_lat_chase_kernel<<<1, 1>>>(dram_buf, d_chase, dram_hops, op);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_ns = (total_ms / ATOMIC_ITERS) * 1e6 / dram_hops;
    printf("  %-30s: %8.1f ns/op\n", name, avg_ns);
    g_results.push_back({"atomic_lat_dram", name, avg_ns, "ns"});
  };

  run_dram("plain_load", [] __device__(uint64_t * data) {
    volatile uint64_t v = *reinterpret_cast<volatile uint64_t *>(data);
    (void)v;
  });

  run_dram("fetch_add_block", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(data[0]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run_dram("fetch_add_device", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(data[0]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run_dram("fetch_add_system", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(data[0]);
    ref.fetch_add(1, cuda::memory_order_relaxed);
  });

  run_dram("cas_block", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(data[0]);
    uint64_t expected = 0;
    while (!ref.compare_exchange_strong(expected, expected + 1,
                                        cuda::memory_order_relaxed)) {
    }
  });

  run_dram("cas_device", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(data[0]);
    uint64_t expected = 0;
    while (!ref.compare_exchange_strong(expected, expected + 1,
                                        cuda::memory_order_relaxed)) {
    }
  });

  run_dram("cas_system", [] __device__(uint64_t * data) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(data[0]);
    uint64_t expected = 0;
    while (!ref.compare_exchange_strong(expected, expected + 1,
                                        cuda::memory_order_relaxed)) {
    }
  });

  CHECK_CUDA(cudaFree(d_chase));
  free(chase);
  free(dram_buf);
}
