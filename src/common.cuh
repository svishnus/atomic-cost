#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <numeric>
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

extern std::vector<Result> g_results;

// GPU timer using CUDA events
struct GpuTimer {
  cudaEvent_t start, stop;
  inline GpuTimer() {
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
  }
  inline ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  inline void begin() { CHECK_CUDA(cudaEventRecord(start)); }
  inline void end() { CHECK_CUDA(cudaEventRecord(stop)); }
  inline float elapsed_ms() {
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// CPU timer
struct CpuTimer {
  std::chrono::high_resolution_clock::time_point t0, t1;
  inline void begin() { t0 = std::chrono::high_resolution_clock::now(); }
  inline void end() { t1 = std::chrono::high_resolution_clock::now(); }
  inline double elapsed_ms() {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

extern int g_l2CacheSize; // set from cudaDeviceProp in main()

static constexpr size_t BW_BUFFER_BYTES = 512ULL * 1024 * 1024; // 512 MB
static constexpr size_t BW_N = BW_BUFFER_BYTES / sizeof(uint64_t);
static constexpr int BW_WARMUP = 2;
static constexpr int BW_ITERS = 10;

extern size_t LAT_N; // default 16M; main() may raise to 4x L2
static constexpr int LAT_HOPS = 1 * 1024 * 1024; // 1M hops
static constexpr int LAT_WARMUP = 2;
static constexpr int LAT_ITERS = 10;

static constexpr size_t ATOMIC_ARRAY_N = 1024 * 1024; // 1M elements
static constexpr int ATOMIC_ITERS_PER_THREAD = 1000;
static constexpr int ATOMIC_WARMUP = 2;
static constexpr int ATOMIC_ITERS = 10;
static constexpr int ATOMIC_TPUT_THREADS = 256;
static constexpr int ATOMIC_TPUT_BLOCKS = 512;

static constexpr int ATOMIC_LAT_OPS = 1000000; // 1M ops for single-thread latency

static constexpr int CACHE_LINE_BYTES = 128; // GPU cache line
static constexpr int CACHE_LINE_ELEMS = CACHE_LINE_BYTES / sizeof(uint64_t);

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Generate a single random cycle of length n using Sattolo's algorithm
inline void generate_chase(uint32_t *arr, size_t n) {
  for (size_t i = 0; i < n; i++)
    arr[i] = (uint32_t)i;
  std::mt19937_64 rng(42);
  for (size_t i = n - 1; i > 0; i--) {
    std::uniform_int_distribution<size_t> dist(0, i - 1);
    size_t j = dist(rng);
    std::swap(arr[i], arr[j]);
  }
}

// ---------------------------------------------------------------------------
// Benchmark declarations
// ---------------------------------------------------------------------------

void bench_bandwidth();
void bench_latency();
void bench_atomic_throughput();
void bench_atomic_throughput_dram();
void bench_atomic_latency();
