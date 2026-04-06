# Measuring cost of `cuda::atomics` on the DGX Spark

- Rough idea: first get a baseline on read / write bandwith of both CPU and GPU from the unified memory pool
- Then, measure cost of both system-wide and block-wide atomics, and compare to the baseline
