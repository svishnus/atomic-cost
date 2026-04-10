#include "cpu_bw.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <numeric>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

struct Result {
  std::string benchmark;
  std::string variant;
  double value;
  std::string unit;
};

static std::vector<Result> g_results;

// ! use nsight
// ? Try using CPU with cudaMalloc

static void print_results() {
  printf("\n%-16s %-20s %12s %s\n", "benchmark", "variant", "value", "unit");
  printf("%-16s %-20s %12s %s\n", "--------", "-------", "-----", "----");
  for (auto &r : g_results) {
    printf("%-16s %-20s %12.2f %s\n", r.benchmark.c_str(), r.variant.c_str(),
           r.value, r.unit.c_str());
  }

  FILE *csv = fopen("results.csv", "w");
  if (csv) {
    fprintf(csv, "benchmark,variant,value,unit\n");
    for (auto &r : g_results) {
      fprintf(csv, "%s,%s,%.4f,%s\n", r.benchmark.c_str(), r.variant.c_str(),
              r.value, r.unit.c_str());
    }
    fclose(csv);
    printf("\nResults written to results.csv\n");
  } else {
    fprintf(stderr, "Warning: could not write results.csv\n");
  }
}

// GPU timer using CUDA events
struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() {
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void begin() { CHECK_CUDA(cudaEventRecord(start)); }
  void end() { CHECK_CUDA(cudaEventRecord(stop)); }
  float elapsed_ms() {
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// CPU timer
struct CpuTimer {
  std::chrono::high_resolution_clock::time_point t0, t1;
  void begin() { t0 = std::chrono::high_resolution_clock::now(); }
  void end() { t1 = std::chrono::high_resolution_clock::now(); }
  double elapsed_ms() {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
};

// Constants
static int g_l2CacheSize = 0; // set from cudaDeviceProp in main()

static constexpr size_t BW_BUFFER_BYTES = 512ULL * 1024 * 1024; // 512 MB
static constexpr size_t BW_N = BW_BUFFER_BYTES / sizeof(uint64_t);
static constexpr int BW_WARMUP = 2;
static constexpr int BW_ITERS = 10;

static size_t LAT_N =
    16 * 1024 * 1024; // default; main() may raise to 4x L2 bytes / 4B per entry
static constexpr int LAT_HOPS = 1 * 1024 * 1024; // 1M hops
static constexpr int LAT_WARMUP = 2;
static constexpr int LAT_ITERS = 10;

static constexpr size_t ATOMIC_ARRAY_N = 1024 * 1024; // 1M elements
static constexpr int ATOMIC_ITERS_PER_THREAD = 1000;
static constexpr int ATOMIC_WARMUP = 2;
static constexpr int ATOMIC_ITERS = 10;
static constexpr int ATOMIC_TPUT_THREADS = 256;
static constexpr int ATOMIC_TPUT_BLOCKS = 512;

static constexpr int ATOMIC_LAT_OPS =
    1000000; // 1M ops for single-thread latency

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

static void bench_bandwidth() {
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

// ---------------------------------------------------------------------------
// Benchmark 2: Latency (pointer chasing)
// ---------------------------------------------------------------------------

// Generate a single random cycle of length n using Sattolo's algorithm
static void generate_chase(uint32_t *arr, size_t n) {
  for (size_t i = 0; i < n; i++)
    arr[i] = (uint32_t)i;
  std::mt19937_64 rng(42);
  for (size_t i = n - 1; i > 0; i--) {
    std::uniform_int_distribution<size_t> dist(0, i - 1);
    size_t j = dist(rng);
    std::swap(arr[i], arr[j]);
  }
}

__global__ void gpu_chase_kernel(const uint32_t *__restrict__ arr, int hops,
                                 uint32_t *__restrict__ out) {
  uint32_t idx = 0;
  for (int i = 0; i < hops; i++) {
    idx = arr[idx];
  }
  *out = idx; // prevent DCE
}

static void bench_latency() {
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

static void bench_atomic_throughput() {
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

static void bench_atomic_throughput_dram() {
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
static constexpr int CACHE_LINE_BYTES = 128; // GPU cache line
static constexpr int CACHE_LINE_ELEMS = CACHE_LINE_BYTES / sizeof(uint64_t);

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

static void bench_atomic_latency() {
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
  // Use enough cache lines to blow past L2.  We want ~4x L2 worth of
  // cache lines so every hop is a guaranteed L2 miss.  The device's L2
  // size is queried at runtime (e.g. GB10 = 24 MB).
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("Device: %s\n", prop.name);
  printf("Compute: %d.%d\n", prop.major, prop.minor);
  printf("SMs: %d\n", prop.multiProcessorCount);
  printf("L2 cache: %d KB (%d MB)\n\n", prop.l2CacheSize / 1024,
         prop.l2CacheSize / (1024 * 1024));

  g_l2CacheSize = prop.l2CacheSize;

  size_t min_entries = (size_t)prop.l2CacheSize * 4 / sizeof(uint32_t);
  if (min_entries > LAT_N)
    LAT_N = min_entries;

  bench_bandwidth();
  bench_latency();
  bench_atomic_throughput();
  bench_atomic_throughput_dram();
  bench_atomic_latency();

  print_results();
  return 0;
}
