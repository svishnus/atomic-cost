#include "cpu_bw.h"
#include <arm_neon.h>

uint64_t cpu_read_neon(const uint64_t *buf, size_t n) {
  uint64x2_t acc0 = vdupq_n_u64(0), acc1 = vdupq_n_u64(0);
  uint64x2_t acc2 = vdupq_n_u64(0), acc3 = vdupq_n_u64(0);
  for (size_t j = 0; j < n; j += 8) {
    acc0 = vaddq_u64(acc0, vld1q_u64(buf + j));
    acc1 = vaddq_u64(acc1, vld1q_u64(buf + j + 2));
    acc2 = vaddq_u64(acc2, vld1q_u64(buf + j + 4));
    acc3 = vaddq_u64(acc3, vld1q_u64(buf + j + 6));
  }
  acc0 = vaddq_u64(acc0, acc1);
  acc2 = vaddq_u64(acc2, acc3);
  acc0 = vaddq_u64(acc0, acc2);
  return vgetq_lane_u64(acc0, 0) ^ vgetq_lane_u64(acc0, 1);
}
