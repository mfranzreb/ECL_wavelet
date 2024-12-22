set(TARGET ecl_RS_benchmarks)

# Collect all the source files in the ./tests directory

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rank_select_construction.cu")

# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/bitarray/include" "${CMAKE_SOURCE_DIR}/third_party/benchmark/include" "${CMAKE_SOURCE_DIR}/utils/include")


# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_utils
    PRIVATE ecl_test_bm_utils
    PRIVATE ecl_bitarray
    OpenMP::OpenMP_CXX
    CUDA::cudart
    benchmark
    benchmark_main
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

############################################################################################################
set(TARGET ecl_RS_profile)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/profile_rank_select_construction.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/bitarray/include" "${CMAKE_SOURCE_DIR}/utils/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)

# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_bitarray
    PRIVATE ecl_utils
    PRIVATE ecl_test_bm_utils
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

############################################################################################################
set(TARGET ecl_WT_benchmarks)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_tree_construction.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/tree/include" "${CMAKE_SOURCE_DIR}/utils/include" "${CMAKE_SOURCE_DIR}/third_party/benchmark/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)

# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_wavelet_tree
    PRIVATE ecl_utils
    PRIVATE ecl_test_bm_utils
    OpenMP::OpenMP_CXX
    CUDA::cudart
    benchmark
    benchmark_main
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

############################################################################################################
set(TARGET ecl_WT_hist_benchmarks)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_hist_computation.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/utils/include" "${CMAKE_SOURCE_DIR}/third_party/benchmark/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)

# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_utils
    PRIVATE ecl_test_bm_utils
    OpenMP::OpenMP_CXX
    CUDA::cudart
    benchmark
    benchmark_main
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

############################################################################################################
set(TARGET ecl_WT_profile)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/profile_tree_construction.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/tree/include" "${CMAKE_SOURCE_DIR}/utils/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)

# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_wavelet_tree
    PRIVATE ecl_utils
    PRIVATE ecl_test_bm_utils
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)