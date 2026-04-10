#include "common.cuh"
#include "cpu_bw.h"
#include <parlay/parallel.h>
#include <parlay/primitives.h>

// ---------------------------------------------------------------------------
// Benchmark 1: Bandwidth
// ---------------------------------------------------------------------------

__global__ void gpu_read_kernel(const uint64_t *__restrict__ data, size_t n,
                                uint64_t *__restrict__ out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  uint64_t acc = 0;
  for (size_t i = idx; i < n; i += stride) {
    acc += data[i];
  }
  if (acc == 0xDEAD)
    out[idx % 32] = acc;
}

__global__ void gpu_write_kernel(uint64_t *__restrict__ data, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < n; i += stride) {
    data[i] = i;
  }
}

void bench_bandwidth() {
  printf("=== Bandwidth ===\n");

  uint64_t *buf = (uint64_t *)malloc(BW_BUFFER_BYTES);
  if (!buf) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
  }
  // First-touch from CPU so pages are populated
  memset(buf, 0x01, BW_BUFFER_BYTES);

  uint64_t *dce_out;
  CHECK_CUDA(cudaMalloc(&dce_out, 32 * sizeof(uint64_t)));

  int threads = 256;
  int blocks = 512;

  // --- GPU Read ---
  {
    GpuTimer timer;
    for (int i = 0; i < BW_WARMUP; i++) {
      gpu_read_kernel<<<blocks, threads>>>(buf, BW_N, dce_out);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    double total_ms = 0;
    for (int i = 0; i < BW_ITERS; i++) {
      timer.begin();
      gpu_read_kernel<<<blocks, threads>>>(buf, BW_N, dce_out);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_s = (total_ms / BW_ITERS) / 1000.0;
    double gbps = BW_BUFFER_BYTES / avg_s / 1e9;
    printf("  GPU read:  %.2f GB/s\n", gbps);
    g_results.push_back({"bw_gpu", "read", gbps, "GB/s"});
  }

  // --- GPU Write ---
  {
    GpuTimer timer;
    for (int i = 0; i < BW_WARMUP; i++) {
      gpu_write_kernel<<<blocks, threads>>>(buf, BW_N);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    double total_ms = 0;
    for (int i = 0; i < BW_ITERS; i++) {
      timer.begin();
      gpu_write_kernel<<<blocks, threads>>>(buf, BW_N);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_s = (total_ms / BW_ITERS) / 1000.0;
    double gbps = BW_BUFFER_BYTES / avg_s / 1e9;
    printf("  GPU write: %.2f GB/s\n", gbps);
    g_results.push_back({"bw_gpu", "write", gbps, "GB/s"});
  }

  // --- CPU Read ---
  {
    CpuTimer timer;
    volatile uint64_t sink = 0;
    auto cpu_read_sum = [&]() -> uint64_t { return cpu_read_neon(buf, BW_N); };
    for (int i = 0; i < BW_WARMUP; i++)
      sink = cpu_read_sum();
    double total_ms = 0;
    for (int i = 0; i < BW_ITERS; i++) {
      timer.begin();
      uint64_t acc = cpu_read_sum();
      timer.end();
      sink = acc;
      total_ms += timer.elapsed_ms();
    }
    double avg_s = (total_ms / BW_ITERS) / 1000.0;
    double gbps = BW_BUFFER_BYTES / avg_s / 1e9;
    printf("  CPU read:  %.2f GB/s\n", gbps);
    g_results.push_back({"bw_cpu", "read", gbps, "GB/s"});
    (void)sink;
  }

  // --- CPU Write ---
  {
    CpuTimer timer;
    for (int i = 0; i < BW_WARMUP; i++) {
      memset(buf, 0x02, BW_BUFFER_BYTES);
    }
    double total_ms = 0;
    for (int i = 0; i < BW_ITERS; i++) {
      timer.begin();
      memset(buf, 0x03, BW_BUFFER_BYTES);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_s = (total_ms / BW_ITERS) / 1000.0;
    double gbps = BW_BUFFER_BYTES / avg_s / 1e9;
    printf("  CPU write: %.2f GB/s\n", gbps);
    g_results.push_back({"bw_cpu", "write", gbps, "GB/s"});
  }

  // --- CPU Parallel Read (parlay, all cores) ---
  {
    size_t nworkers = parlay::num_workers();
    // Warmup
    for (int i = 0; i < BW_WARMUP; i++) {
      volatile uint64_t s = parlay::reduce(parlay::delayed_seq<uint64_t>(
          BW_N, [&](size_t j) { return buf[j]; }));
      (void)s;
    }
    CpuTimer timer;
    double total_ms = 0;
    volatile uint64_t sink = 0;
    for (int i = 0; i < BW_ITERS; i++) {
      timer.begin();
      uint64_t s = parlay::reduce(parlay::delayed_seq<uint64_t>(
          BW_N, [&](size_t j) { return buf[j]; }));
      timer.end();
      sink = s;
      total_ms += timer.elapsed_ms();
    }
    double avg_s = (total_ms / BW_ITERS) / 1000.0;
    double gbps = BW_BUFFER_BYTES / avg_s / 1e9;
    printf("  CPU read  (parlay, %zu workers): %.2f GB/s\n", nworkers, gbps);
    g_results.push_back({"bw_cpu_par", "read", gbps, "GB/s"});
    (void)sink;
  }

  // --- CPU Parallel Write (parlay, all cores) ---
  {
    size_t nworkers = parlay::num_workers();
    for (int i = 0; i < BW_WARMUP; i++) {
      parlay::parallel_for(0, BW_N, [&](size_t j) { buf[j] = j; }, 8192);
    }
    CpuTimer timer;
    double total_ms = 0;
    for (int i = 0; i < BW_ITERS; i++) {
      timer.begin();
      parlay::parallel_for(0, BW_N, [&](size_t j) { buf[j] = j; }, 8192);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_s = (total_ms / BW_ITERS) / 1000.0;
    double gbps = BW_BUFFER_BYTES / avg_s / 1e9;
    printf("  CPU write (parlay, %zu workers): %.2f GB/s\n", nworkers, gbps);
    g_results.push_back({"bw_cpu_par", "write", gbps, "GB/s"});
  }

  // --- GPU Read (cudaMalloc, no ATS) ---
  uint64_t *dbuf;
  CHECK_CUDA(cudaMalloc(&dbuf, BW_BUFFER_BYTES));
  CHECK_CUDA(cudaMemset(dbuf, 0x01, BW_BUFFER_BYTES));

  {
    GpuTimer timer;
    for (int i = 0; i < BW_WARMUP; i++) {
      gpu_read_kernel<<<blocks, threads>>>(dbuf, BW_N, dce_out);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    double total_ms = 0;
    for (int i = 0; i < BW_ITERS; i++) {
      timer.begin();
      gpu_read_kernel<<<blocks, threads>>>(dbuf, BW_N, dce_out);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_s = (total_ms / BW_ITERS) / 1000.0;
    double gbps = BW_BUFFER_BYTES / avg_s / 1e9;
    printf("  GPU read  (cudaMalloc): %.2f GB/s\n", gbps);
    g_results.push_back({"bw_gpu_dev", "read", gbps, "GB/s"});
  }

  // --- GPU Write (cudaMalloc, no ATS) ---
  {
    GpuTimer timer;
    for (int i = 0; i < BW_WARMUP; i++) {
      gpu_write_kernel<<<blocks, threads>>>(dbuf, BW_N);
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    double total_ms = 0;
    for (int i = 0; i < BW_ITERS; i++) {
      timer.begin();
      gpu_write_kernel<<<blocks, threads>>>(dbuf, BW_N);
      timer.end();
      total_ms += timer.elapsed_ms();
    }
    double avg_s = (total_ms / BW_ITERS) / 1000.0;
    double gbps = BW_BUFFER_BYTES / avg_s / 1e9;
    printf("  GPU write (cudaMalloc): %.2f GB/s\n", gbps);
    g_results.push_back({"bw_gpu_dev", "write", gbps, "GB/s"});
  }

  CHECK_CUDA(cudaFree(dbuf));

  // --- Concurrent CPU+GPU on the same shared buffer ---
  printf("  -- Concurrent CPU+GPU (same shared buffer) --\n");

  // GPU and CPU each operate on their own half of the 512 MB buffer
  // to avoid data races while still contending for memory bandwidth.

  size_t nworkers = parlay::num_workers();

  // Helper: run GPU + CPU concurrently, report both bandwidths
  size_t half_n = BW_N / 2;
  size_t half_bytes = BW_BUFFER_BYTES / 2;
  uint64_t *gpu_buf = buf;
  uint64_t *cpu_buf = buf + half_n;

  auto run_concurrent = [&](const char *gpu_op, const char *cpu_op,
                            bool gpu_write, bool cpu_write) {
    // Warmup
    for (int i = 0; i < BW_WARMUP; i++) {
      if (gpu_write)
        gpu_write_kernel<<<blocks, threads>>>(gpu_buf, half_n);
      else
        gpu_read_kernel<<<blocks, threads>>>(gpu_buf, half_n, dce_out);
      if (cpu_write)
        parlay::parallel_for(
            0, half_n, [&](size_t j) { cpu_buf[j] = j; }, 8192);
      else {
        volatile uint64_t s = parlay::reduce(parlay::delayed_seq<uint64_t>(
            half_n, [&](size_t j) { return cpu_buf[j]; }));
        (void)s;
      }
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    double gpu_total_ms = 0;
    double cpu_total_ms = 0;

    for (int i = 0; i < BW_ITERS; i++) {
      GpuTimer gtimer;
      gtimer.begin();
      if (gpu_write)
        gpu_write_kernel<<<blocks, threads>>>(gpu_buf, half_n);
      else
        gpu_read_kernel<<<blocks, threads>>>(gpu_buf, half_n, dce_out);
      gtimer.end();

      CpuTimer ctimer;
      ctimer.begin();
      if (cpu_write) {
        parlay::parallel_for(
            0, half_n, [&](size_t j) { cpu_buf[j] = j; }, 8192);
      } else {
        volatile uint64_t s = parlay::reduce(parlay::delayed_seq<uint64_t>(
            half_n, [&](size_t j) { return cpu_buf[j]; }));
        (void)s;
      }
      ctimer.end();

      gpu_total_ms += gtimer.elapsed_ms();
      cpu_total_ms += ctimer.elapsed_ms();
    }

    double gpu_avg_s = (gpu_total_ms / BW_ITERS) / 1000.0;
    double cpu_avg_s = (cpu_total_ms / BW_ITERS) / 1000.0;
    double gpu_gbps = half_bytes / gpu_avg_s / 1e9;
    double cpu_gbps = half_bytes / cpu_avg_s / 1e9;

    printf("  GPU %-5s + CPU %-5s:  GPU %.2f GB/s, CPU %.2f GB/s\n", gpu_op,
           cpu_op, gpu_gbps, cpu_gbps);

    std::string gv = std::string("gpu_") + gpu_op + "+cpu_" + cpu_op + "(gpu)";
    std::string cv = std::string("gpu_") + gpu_op + "+cpu_" + cpu_op + "(cpu)";
    g_results.push_back({"bw_concurrent", gv, gpu_gbps, "GB/s"});
    g_results.push_back({"bw_concurrent", cv, cpu_gbps, "GB/s"});
  };

  run_concurrent("read", "read", false, false);
  run_concurrent("write", "write", true, true);
  run_concurrent("read", "write", false, true);
  run_concurrent("write", "read", true, false);

  CHECK_CUDA(cudaFree(dce_out));
  free(buf);
}
