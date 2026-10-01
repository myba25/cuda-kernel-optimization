// Stage 0: copy kernel, out[i] = in[i] for i in [0, n).
//
// A copy reads every byte once and writes it once, which is the minimum traffic of any
// memory-bound kernel. Its bandwidth is the practical peak that reduction and softmax are
// compared against (bench/bench_copy.cu also measures cudaMemcpy as a baseline).
//
// TODO:
//   1. Write a __global__ kernel in this file.
//   2. Implement launch_copy(): choose a block size, compute the grid size,
//      launch on `stream`, then CUDA_CHECK_LAST().
//   3. Any n must work (1000, 1023, ...), not only multiples of the block size.
//   Later versions are welcome (grid-stride loop, float4 loads): add them to bench_copy.cu.
//
// Check:   build/bin/test_copy
// Measure: build/bin/bench_copy
#pragma once

#include "common.cuh"

inline void launch_copy(const float* in, float* out, size_t n, cudaStream_t stream) {
    // TODO
}
