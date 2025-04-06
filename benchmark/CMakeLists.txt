set(TARGET ecl_RS_benchmarks)

# Collect all the source files in the ./tests directory

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rank_select_structure.cu")

# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/third_party/benchmark/include")


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
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

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
set(TARGET pasta_RS_benchmarks)

# Collect all the source files in the ./tests directory

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rank_select_structure_PASTA.cpp")

# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/third_party/benchmark/include" "${CMAKE_SOURCE_DIR}/third_party/pasta/include" "${CMAKE_SOURCE_DIR}/third_party/pasta_utils/include" "${CMAKE_SOURCE_DIR}/third_party/tlx")


# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    OpenMP::OpenMP_CXX
    benchmark
    benchmark_main
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

############################################################################################################
set(TARGET ecl_WT_benchmarks)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_tree_construction.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/third_party/benchmark/include")

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
set(TARGET ecl_bm_tree_construction_from_data)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_tree_construction_from_data.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

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


############################################################################################################
set(TARGET ecl_WT_hist_benchmarks)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_hist_computation.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/third_party/benchmark/include")

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
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

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


############################################################################################################
set(TARGET ecl_access_profile)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/profile_access.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

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

############################################################################################################
set(TARGET ecl_access_benchmark)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_access.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/third_party/benchmark/include")

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
set(TARGET ecl_access_benchmark_from_data)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_access_from_data.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

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

############################################################################################################
set(TARGET ecl_queries_benchmark)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_queries.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

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

############################################################################################################
set(TARGET sdsl_queries_benchmark)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_queries_sdsl.cpp")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
)

# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libsdsl.a
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libdivsufsort64.a
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libdivsufsort.a
    OpenMP::OpenMP_CXX
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

############################################################################################################
set(TARGET sdsl_access_benchmark_from_data)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_access_SDSL_from_data.cpp")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
)

# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libsdsl.a
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libdivsufsort64.a
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libdivsufsort.a
    OpenMP::OpenMP_CXX
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)


############################################################################################################
set(TARGET ecl_rank_profile)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/profile_rank.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

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

############################################################################################################
set(TARGET ecl_rank_benchmark)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rank.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/third_party/benchmark/include")

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
set(TARGET sdsl_rank_benchmark_from_data)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rank_SDSL_from_data.cpp")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
)

# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libsdsl.a
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libdivsufsort64.a
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libdivsufsort.a
    OpenMP::OpenMP_CXX
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

############################################################################################################
set(TARGET ecl_rank_benchmark_from_data)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rank_from_data.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

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

############################################################################################################
set(TARGET ecl_select_profile)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/profile_select.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

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

############################################################################################################
set(TARGET ecl_select_benchmark)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_select.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/third_party/benchmark/include")

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
set(TARGET ecl_select_benchmark_from_data)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_select_from_data.cu")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

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

############################################################################################################
set(TARGET sdsl_select_benchmark_from_data)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_select_SDSL_from_data.cpp")


# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
)

# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libsdsl.a
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libdivsufsort64.a
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/sdsl-lite/lib/libdivsufsort.a
    OpenMP::OpenMP_CXX
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)