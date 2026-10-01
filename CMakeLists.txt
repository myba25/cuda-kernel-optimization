cmake_minimum_required(VERSION 3.24)

# Target GPU architecture. Default: sm_89 (RTX 40xx). Override at configure time:
#   -DCMAKE_CUDA_ARCHITECTURES=native   the GPU in this machine
#   -DCMAKE_CUDA_ARCHITECTURES=75       RTX 20xx / T4   (86: RTX 30xx, 120: RTX 50xx)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(cuda_kernel_optimization LANGUAGES CXX CUDA)

# Benchmarks are meaningless in Debug: default single-config generators (Ninja, Make) to Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

# All executables land in build/bin (the $<1:...> stops Visual Studio from adding a Release/ subfolder).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_BINARY_DIR}/bin>)

option(PTXAS_VERBOSE "Print registers, shared memory and spills per kernel (-Xptxas -v)" OFF)

function(add_cuda_executable name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/kernels)
  # Benchmarks write CSVs to results/<gpu>/ in the source tree.
  target_compile_definitions(${name} PRIVATE RESULTS_DIR="${PROJECT_SOURCE_DIR}/results")

  # -lineinfo maps SASS back to source lines in Nsight Compute and does not slow the code down.
  # Release adds -O3 for host code; device code is always optimized by ptxas.
  # Never benchmark a -G (device debug) build.
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  if(PTXAS_VERBOSE)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
  endif()
  if(NOT MSVC)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
  endif()

  # cuBLAS is the SGEMM reference (stage 2). CUB is header-only and ships with the toolkit.
  if(name MATCHES "gemm")
    target_link_libraries(${name} PRIVATE CUDA::cublas)
  endif()
endfunction()

# Every bench/*.cu and tests/*.cu becomes its own executable; tests are registered with CTest.
# New files are picked up on the next build (CONFIGURE_DEPENDS), no need to edit this file.
enable_testing()

file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/bench/*.cu)
foreach(src IN LISTS BENCH_SOURCES)
  get_filename_component(name ${src} NAME_WE)
  add_cuda_executable(${name} ${src})
endforeach()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/*.cu)
foreach(src IN LISTS TEST_SOURCES)
  get_filename_component(name ${src} NAME_WE)
  add_cuda_executable(${name} ${src})
  add_test(NAME ${name} COMMAND ${name})
endforeach()
