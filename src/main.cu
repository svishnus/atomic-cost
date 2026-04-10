#include "common.cuh"

// ---------------------------------------------------------------------------
// Global definitions (declared extern in common.cuh)
// ---------------------------------------------------------------------------

std::vector<Result> g_results;
int g_l2CacheSize = 0;
size_t LAT_N = 16 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Results output
// ---------------------------------------------------------------------------

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
