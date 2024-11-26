set(TARGET ecl_RS_benchmarks)

# Collect all the source files in the ./tests directory

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rank_select_construction.cu")

# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_SOURCE_DIR}/third_party/benchmark/include")


# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_bitarray
    OpenMP::OpenMP_CXX
    CUDA::cudart
    benchmark
    benchmark_main
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(TARGET ecl_RS_profile)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/profile_rank_select_construction.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/../include")


# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_bitarray
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)