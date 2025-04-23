# Set CMake version number
cmake_minimum_required(VERSION 3.25)

project(ECLWavelet VERSION 0.1.0 LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(PROFILE "Enable profiling" OFF)

find_package(OpenMP REQUIRED)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g" CACHE STRING "Flags used by the compiler during Release with Debug Info builds." FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g" CACHE STRING "Flags used by the compiler during Release with Debug Info builds." FORCE)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra --default-stream per-thread -Xcompiler=-fopenmp")

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra)
endif()

#if in Debug mode, enable CUDA debugging
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G --maxrregcount=32")  # enable cuda-gdb
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Build not specified, defaulted to RelWithDebInfo" FORCE)
    message(STATUS "Build type not specified, defaulted to RelWithDebInfo")
endif(NOT CMAKE_BUILD_TYPE)


if(PROFILE AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --ptxas-options=-v")
endif()

add_subdirectory(include/ecl_wavelet)

add_subdirectory(src)

message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
message(STATUS "CUDAToolkit_FLAGS: ${CMAKE_CUDA_FLAGS}")

if(BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF)
    add_subdirectory(third_party/benchmark)
    add_subdirectory(benchmark)
    set(GPU_ARCHITECTURE "sm_62")
endif()


if(BUILD_TESTS)
    add_subdirectory(third_party/googletest)
    add_subdirectory(tests)
    enable_testing()
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()