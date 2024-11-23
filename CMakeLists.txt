# Set CMake version number
cmake_minimum_required(VERSION 3.25)

project(ECLWavelet VERSION 0.1.0 LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build tests" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

#if in Debug mode, enable CUDA debugging
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Build not specified, defaulted to RelWithDebInfo" FORCE)
    message(STATUS "Build type not specified, defaulted to RelWithDebInfo")
endif(NOT CMAKE_BUILD_TYPE)

find_package(OpenMP REQUIRED)

find_package(CUDAToolkit REQUIRED)

# Add locations of other CMakelists.txt (one per call)
add_subdirectory(bitarray)

#add_subdirectory(tree)


message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
message(STATUS "CUDAToolkit_FLAGS: ${CMAKE_CUDA_FLAGS}")

if(BUILD_TESTS)
    add_subdirectory(third_party/googletest)
    add_subdirectory(bitarray/tests)
    #add_subdirectory(tree/tests)
    enable_testing()
endif()