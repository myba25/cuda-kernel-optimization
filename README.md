# CUDA Kernel Optimization

Reduction, SGEMM and softmax written from scratch in CUDA, one optimization at a time: every version
is a separate file, measured against CUB / cuBLAS and explained with Nsight Compute metrics.

> **Status:** stage 0, infrastructure. No results yet.

## Results

_Coming with stage 1: performance per version and % of the library reference (CUB, cuBLAS)._

## Hardware and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB, Ada (sm_89), 20 SMs |
| Driver | 610.78 |
| CUDA Toolkit | 13.4 |
| Nsight Compute | 2026.3.0 |
| Host compiler | MSVC, Visual Studio 2026 (18.10) |
| CMake | 4.4.3 |
| OS | Windows |

## Build and run

Requirements: CUDA Toolkit 12.x or 13.x, CMake ≥ 3.24, a C++17 host compiler, Python 3 with
`pandas` and `matplotlib` (`pip install -r scripts/requirements.txt`).

**Windows** (PowerShell):

```powershell
. .\scripts\dev_shell.ps1                 # x64 MSVC environment, so nvcc can find cl.exe
cmake -S . -B build -G Ninja              # Release, sm_89 by default
cmake --build build
ctest --test-dir build --output-on-failure
.\build\bin\bench_copy.exe
python scripts\plot.py results\rtx_4050_laptop_gpu\bench_copy.csv
```

**Linux:**

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
ctest --test-dir build --output-on-failure
./build/bin/bench_copy
python3 scripts/plot.py results/<gpu>/bench_copy.csv
```

Other GPUs: `-DCMAKE_CUDA_ARCHITECTURES=75` (RTX 20xx, T4), `86` (RTX 30xx), `89` (RTX 40xx), `120` (RTX 50xx).

### Profiling and checking

```bash
ncu --set full -k <kernel_name> -c 1 -o results/<gpu>/ncu/<name> build/bin/bench_copy
compute-sanitizer --tool memcheck build/bin/test_copy     # also: racecheck, synccheck
cmake -S . -B build -DPTXAS_VERBOSE=ON                    # registers and spills per kernel
```

`ncu` locks GPU clocks to base by default, so its durations are never mixed with benchmark numbers.

## Methodology

- Kernel time comes from `cudaEvent`s around each launch, without host↔device copies:
  10 warm-up runs, then 100 timed runs, median reported.
- SGEMM: GFLOPS = 2·M·N·K / t, against `cublasSgemm` in strict FP32 (no TF32).
- Reduction and softmax: GB/s counted over the minimal traffic (input read once, output written once),
  so extra passes over memory show up as lower GB/s. They are compared with the practical peak
  (`bench_copy`) and the theoretical peak.
- Correctness: fixed seed, results compared with a reference (CPU in double, CUB, cuBLAS) by relative
  error. Results are not bit-identical because the summation order differs.

## Roadmap

- [ ] **Stage 0, infrastructure:** build, timing, CSV, plots, copy kernel for the practical bandwidth peak
- [ ] **Stage 1, reduction:** R1 interleaved → R6 grid-stride + `float4`, vs `cub::DeviceReduce::Sum`
- [ ] **Stage 2, SGEMM:** G1 naive → G6 vectorized 2D blocktiling, vs `cublasSgemm`
- [ ] **Stage 3, softmax:** S1 naive → S4 online softmax, for 128 to 32768 columns
- [ ] **Stage 4, analysis:** Nsight Compute per version, roofline, what didn't work

## Repository layout

```
include/common.cuh   CUDA_CHECK, DeviceBuffer, GpuTimer + benchmark(), test data, compare(), CSV
kernels/             one file per kernel version: copy/, vector_add/, reduce/, gemm/, softmax/
bench/               benchmarks, each writes results/<gpu>/<name>.csv
tests/               every version against a reference, run with ctest
scripts/             plot.py (CSV -> PNG), dev_shell.ps1 (MSVC environment on Windows)
results/<gpu>/       CSVs, plots/*.png, ncu/*.ncu-rep
docs/                per-version write-ups and Nsight Compute screenshots
```

## References

- NVIDIA: [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/),
  [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/),
  [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/)
- Hwu, Kirk, El Hajj: *Programming Massively Parallel Processors*, 4th edition
- Mark Harris: *Optimizing Parallel Reduction in CUDA*
- Simon Boehm: *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*
- Milakov, Gimelshein: *Online normalizer calculation for softmax*
