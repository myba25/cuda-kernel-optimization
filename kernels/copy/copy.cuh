// Stage 0: copy kernel, out[i] = in[i] for i in [0, n).
//
// A copy reads every byte once and writes it once, which is the minimum traffic of any
// memory-bound kernel. Its bandwidth is the practical peak that reduction and softmax are
// compared against (bench/bench_copy.cu also measures cudaMemcpy as a baseline).
//
// Version 1: one thread per element, like vector add. Ideas for later versions, each in its own
// file: a grid-stride loop with a few blocks per SM, float4 loads and stores.
//
// Check:   build/bin/test_copy
// Measure: build/bin/bench_copy
#pragma once

#include "common.cuh"

__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    // Widen to size_t before multiplying: blockIdx.x * blockDim.x is 32-bit and overflows at 2^32.
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

inline void launch_copy(const float* in, float* out, size_t n, cudaStream_t stream) {
    constexpr unsigned kBlockSize = 256;  // a multiple of the 32-thread warp
    if (n == 0) return;                   // a grid of 0 blocks is an invalid launch
    const size_t grid = ceil_div(n, kBlockSize);
    REQUIRE(grid <= 0x7FFFFFFFu, "n is too large for a 1D grid");
    copy_kernel<<<static_cast<unsigned>(grid), kBlockSize, 0, stream>>>(in, out, n);
    CUDA_CHECK_LAST();
}
