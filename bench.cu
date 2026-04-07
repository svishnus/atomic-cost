#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

struct Result {
    const char* benchmark;
    const char* variant;
    double value;
    const char* unit;
};

static std::vector<Result> g_results;

static void print_results() {
    printf("\n%-16s %-20s %12s %s\n", "benchmark", "variant", "value", "unit");
    printf("%-16s %-20s %12s %s\n", "--------", "-------", "-----", "----");
    for (auto& r : g_results) {
        printf("%-16s %-20s %12.2f %s\n", r.benchmark, r.variant, r.value,
               r.unit);
    }

    FILE* csv = fopen("results.csv", "w");
    if (csv) {
        fprintf(csv, "benchmark,variant,value,unit\n");
        for (auto& r : g_results) {
            fprintf(csv, "%s,%s,%.4f,%s\n", r.benchmark, r.variant, r.value,
                    r.unit);
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
static constexpr size_t BW_BUFFER_BYTES = 512ULL * 1024 * 1024; // 512 MB
static constexpr size_t BW_N = BW_BUFFER_BYTES / sizeof(uint64_t);
static constexpr int BW_WARMUP = 2;
static constexpr int BW_ITERS = 10;

static constexpr size_t LAT_N = 16 * 1024 * 1024; // 16M entries
static constexpr int LAT_HOPS = 1 * 1024 * 1024;  // 1M hops
static constexpr int LAT_WARMUP = 2;
static constexpr int LAT_ITERS = 10;

static constexpr size_t ATOMIC_ARRAY_N = 1024 * 1024; // 1M elements
static constexpr int ATOMIC_ITERS_PER_THREAD = 1000;
static constexpr int ATOMIC_WARMUP = 2;
static constexpr int ATOMIC_ITERS = 10;
static constexpr int ATOMIC_TPUT_THREADS = 256;
static constexpr int ATOMIC_TPUT_BLOCKS = 512;

static constexpr int ATOMIC_LAT_OPS = 1000000; // 1M ops for single-thread latency

// ---------------------------------------------------------------------------
// Benchmark 1: Bandwidth
// ---------------------------------------------------------------------------

__global__ void gpu_read_kernel(const uint64_t* __restrict__ data, size_t n,
                                uint64_t* __restrict__ out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    uint64_t acc = 0;
    for (size_t i = idx; i < n; i += stride) {
        acc += data[i];
    }
    if (acc == 0xDEAD) out[idx % 32] = acc; // prevent DCE
}

__global__ void gpu_write_kernel(uint64_t* __restrict__ data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        data[i] = i;
    }
}

static void bench_bandwidth() {
    printf("=== Bandwidth ===\n");

    uint64_t* buf = (uint64_t*)malloc(BW_BUFFER_BYTES);
    if (!buf) { fprintf(stderr, "malloc failed\n"); exit(1); }
    // First-touch from CPU so pages are populated
    memset(buf, 0x01, BW_BUFFER_BYTES);

    uint64_t* dce_out;
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
        for (int i = 0; i < BW_WARMUP; i++) {
            uint64_t acc = 0;
            for (size_t j = 0; j < BW_N; j++) acc += buf[j];
            sink = acc;
        }
        double total_ms = 0;
        for (int i = 0; i < BW_ITERS; i++) {
            uint64_t acc = 0;
            timer.begin();
            for (size_t j = 0; j < BW_N; j++) acc += buf[j];
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
            volatile uint64_t s = parlay::reduce(
                parlay::delayed_seq<uint64_t>(BW_N, [&](size_t j) { return buf[j]; }));
            (void)s;
        }
        CpuTimer timer;
        double total_ms = 0;
        volatile uint64_t sink = 0;
        for (int i = 0; i < BW_ITERS; i++) {
            timer.begin();
            uint64_t s = parlay::reduce(
                parlay::delayed_seq<uint64_t>(BW_N, [&](size_t j) { return buf[j]; }));
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

    CHECK_CUDA(cudaFree(dce_out));
    free(buf);
}

// ---------------------------------------------------------------------------
// Benchmark 2: Latency (pointer chasing)
// ---------------------------------------------------------------------------

// Generate a single random cycle of length n using Sattolo's algorithm
static void generate_chase(uint32_t* arr, size_t n) {
    for (size_t i = 0; i < n; i++) arr[i] = (uint32_t)i;
    std::mt19937_64 rng(42);
    for (size_t i = n - 1; i > 0; i--) {
        std::uniform_int_distribution<size_t> dist(0, i - 1);
        size_t j = dist(rng);
        std::swap(arr[i], arr[j]);
    }
}

__global__ void gpu_chase_kernel(const uint32_t* __restrict__ arr, int hops,
                                 uint32_t* __restrict__ out) {
    uint32_t idx = 0;
    for (int i = 0; i < hops; i++) {
        idx = arr[idx];
    }
    *out = idx; // prevent DCE
}

static void bench_latency() {
    printf("\n=== Latency (pointer chase) ===\n");

    uint32_t* arr = (uint32_t*)malloc(LAT_N * sizeof(uint32_t));
    if (!arr) { fprintf(stderr, "malloc failed\n"); exit(1); }

    printf("  Generating random permutation (%zu entries)...\n", LAT_N);
    generate_chase(arr, LAT_N);

    uint32_t* dce_out;
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
            for (int j = 0; j < LAT_HOPS; j++) idx = arr[idx];
            sink = idx;
        }
        double total_ms = 0;
        for (int i = 0; i < LAT_ITERS; i++) {
            uint32_t idx = 0;
            timer.begin();
            for (int j = 0; j < LAT_HOPS; j++) idx = arr[idx];
            timer.end();
            sink = idx;
            total_ms += timer.elapsed_ms();
        }
        double avg_ns = (total_ms / LAT_ITERS) * 1e6 / LAT_HOPS;
        printf("  CPU: %.1f ns/hop\n", avg_ns);
        g_results.push_back({"lat_cpu", "pointer_chase", avg_ns, "ns"});
        (void)sink;
    }

    CHECK_CUDA(cudaFree(dce_out));
    free(arr);
}

// ---------------------------------------------------------------------------
// Benchmark 3a: Atomic Throughput
// ---------------------------------------------------------------------------

// Templated throughput kernel: each thread runs `op(data, index)` in a loop.
template <typename OpFunc>
__global__ void atomic_tput_kernel(uint64_t* data, size_t n_elements,
                                   int iters, OpFunc op) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iters; i++) {
        op(data, tid, n_elements);
    }
}

static void bench_atomic_throughput() {
    printf("\n=== Atomic Throughput ===\n");

    uint64_t* buf = (uint64_t*)malloc(ATOMIC_ARRAY_N * sizeof(uint64_t));
    if (!buf) { fprintf(stderr, "malloc failed\n"); exit(1); }
    memset(buf, 0, ATOMIC_ARRAY_N * sizeof(uint64_t));

    size_t total_threads = (size_t)ATOMIC_TPUT_BLOCKS * ATOMIC_TPUT_THREADS;
    double total_ops = (double)total_threads * ATOMIC_ITERS_PER_THREAD;

    // Helper to run one variant
    auto run = [&](const char* name, const char* contention, auto op) {
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
        // Build variant string: "name(contention)"
        static char vbuf[64];
        snprintf(vbuf, sizeof(vbuf), "%s(%s)", name, contention);
        // Need to strdup since vbuf is static and reused
        g_results.push_back({"atomic_tput", strdup(vbuf), gops, "Gops/s"});
    };

    // --- Uncontended: each thread hits a different element ---
    printf("  -- Uncontended --\n");

    run("plain_store", "uncontended",
        [] __device__(uint64_t * data, size_t tid, size_t n) {
            data[tid % n] = tid;
        });

    run("plain_load", "uncontended",
        [] __device__(uint64_t * data, size_t tid, size_t n) {
            volatile uint64_t v = data[tid % n];
            (void)v;
        });

    run("fetch_add_block", "uncontended",
        [] __device__(uint64_t * data, size_t tid, size_t n) {
            cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(
                data[tid % n]);
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
// Benchmark 3b: Atomic Latency (single thread)
// ---------------------------------------------------------------------------

template <typename OpFunc>
__global__ void atomic_lat_kernel(uint64_t* data, int ops, OpFunc op) {
    for (int i = 0; i < ops; i++) {
        op(data);
    }
}

static void bench_atomic_latency() {
    printf("\n=== Atomic Latency (single thread) ===\n");

    uint64_t* buf = (uint64_t*)malloc(sizeof(uint64_t));
    if (!buf) { fprintf(stderr, "malloc failed\n"); exit(1); }
    *buf = 0;

    auto run = [&](const char* name, auto op) {
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

    run("plain_store", [] __device__(uint64_t * data) {
        data[0] = 1;
    });

    run("plain_load", [] __device__(uint64_t * data) {
        volatile uint64_t v = data[0];
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

    run("cas_block", [] __device__(uint64_t * data) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_block> ref(data[0]);
        uint64_t expected = ref.load(cuda::memory_order_relaxed);
        ref.compare_exchange_strong(expected, expected + 1,
                                    cuda::memory_order_relaxed);
    });

    run("cas_device", [] __device__(uint64_t * data) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(data[0]);
        uint64_t expected = ref.load(cuda::memory_order_relaxed);
        ref.compare_exchange_strong(expected, expected + 1,
                                    cuda::memory_order_relaxed);
    });

    run("cas_system", [] __device__(uint64_t * data) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref(data[0]);
        uint64_t expected = ref.load(cuda::memory_order_relaxed);
        ref.compare_exchange_strong(expected, expected + 1,
                                    cuda::memory_order_relaxed);
    });

    free(buf);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n\n", prop.major, prop.minor);

    bench_bandwidth();
    bench_latency();
    bench_atomic_throughput();
    bench_atomic_latency();

    print_results();
    return 0;
}
