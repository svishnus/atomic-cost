#include "common.cuh"

// ---------------------------------------------------------------------------
// Benchmark 2: Latency (pointer chasing)
// ---------------------------------------------------------------------------

__global__ void gpu_chase_kernel(const uint32_t *__restrict__ arr, int hops,
                                 uint32_t *__restrict__ out) {
  uint32_t idx = 0;
  for (int i = 0; i < hops; i++) {
    idx = arr[idx];
  }
  *out = idx; // prevent DCE
}

void bench_latency() {
  printf("\n=== Latency (pointer chase) ===\n");

  uint32_t *arr = (uint32_t *)malloc(LAT_N * sizeof(uint32_t));
  if (!arr) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }

  printf("  Generating random permutation (%zu entries)...\n", LAT_N);
  generate_chase(arr, LAT_N);

  uint32_t *dce_out;
  CHECK_CUDA(cudaMalloc(&dce_out, sizeof(uint32_t)));

  // --- GPU latency ---
  {
    GpuTimer timer;
    for (int i = 0; i < LAT_WARMUP; i++) {
      gpu_chase_kernel<<<1, 1>>>(arr, LAT_HOPS, dce_out);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    double total_ms = 0;
    for (int i = 0; i < LAT_ITERS; i++) {
      timer.begin();
      gpu_chase_kernel<<<1, 1>>>(arr, LAT_HOPS, dce_out);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_ns = (total_ms / LAT_ITERS) * 1e6 / LAT_HOPS;
    printf("  GPU: %.1f ns/hop\n", avg_ns);
    g_results.push_back({"lat_gpu", "pointer_chase", avg_ns, "ns"});
  }

  // --- CPU latency ---
  {
    CpuTimer timer;
    volatile uint32_t sink = 0;
    for (int i = 0; i < LAT_WARMUP; i++) {
      uint32_t idx = 0;
      for (int j = 0; j < LAT_HOPS; j++)
        idx = arr[idx];
      sink = idx;
    }
    double total_ms = 0;
    for (int i = 0; i < LAT_ITERS; i++) {
      uint32_t idx = 0;
      timer.begin();
      for (int j = 0; j < LAT_HOPS; j++)
        idx = arr[idx];
      timer.end();
      sink = idx;
      total_ms += timer.elapsed_ms();
    }
    double avg_ns = (total_ms / LAT_ITERS) * 1e6 / LAT_HOPS;
    printf("  CPU: %.1f ns/hop\n", avg_ns);
    g_results.push_back({"lat_cpu", "pointer_chase", avg_ns, "ns"});
    (void)sink;
  }

  // --- GPU latency (cudaMalloc, no ATS) ---
  {
    uint32_t *darr;
    CHECK_CUDA(cudaMalloc(&darr, LAT_N * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(darr, arr, LAT_N * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    GpuTimer timer;
    for (int i = 0; i < LAT_WARMUP; i++) {
      gpu_chase_kernel<<<1, 1>>>(darr, LAT_HOPS, dce_out);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    double total_ms = 0;
    for (int i = 0; i < LAT_ITERS; i++) {
      timer.begin();
      gpu_chase_kernel<<<1, 1>>>(darr, LAT_HOPS, dce_out);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_ns = (total_ms / LAT_ITERS) * 1e6 / LAT_HOPS;
    printf("  GPU (cudaMalloc): %.1f ns/hop\n", avg_ns);
    g_results.push_back({"lat_gpu_dev", "pointer_chase", avg_ns, "ns"});

    CHECK_CUDA(cudaFree(darr));
  }

  CHECK_CUDA(cudaFree(dce_out));
  free(arr);
}
