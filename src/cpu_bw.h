#pragma once
#include <cstddef>
#include <cstdint>

// NEON-vectorized CPU read bandwidth (compiled separately by g++ to get
// proper auto-vectorization, since nvcc's host pass can't include arm_neon.h).
uint64_t cpu_read_neon(const uint64_t *buf, size_t n);
