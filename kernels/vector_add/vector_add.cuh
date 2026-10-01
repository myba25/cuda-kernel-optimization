// Stage 0 warm-up: vector add, c[i] = a[i] + b[i] for i in [0, n).
//
// TODO:
//   1. Write a __global__ kernel in this file.
//   2. Implement launch_vector_add(): choose a block size, compute the grid size,
//      launch on `stream`, then CUDA_CHECK_LAST().
//   3. Any n must work (1000, 1023, ...), not only multiples of the block size.
//
// Check: build/bin/test_vector_add
#pragma once

#include "common.cuh"

inline void launch_vector_add(const float* a, const float* b, float* c, size_t n, cudaStream_t stream) {
    // TODO
}
