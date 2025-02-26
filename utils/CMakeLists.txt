set(TARGET ecl_utils)

add_library(${TARGET} STATIC "${CMAKE_SOURCE_DIR}/utils/src/utils.cu")  

include_directories(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)


target_link_libraries(${TARGET}
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

################################################################################################

set(TARGET ecl_test_bm_utils)

add_library(${TARGET} STATIC "${CMAKE_SOURCE_DIR}/utils/src/test_benchmark_utils.cu")  

include_directories(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_SOURCE_DIR}/bitarray/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)


target_link_libraries(${TARGET}
    PRIVATE ecl_utils
    PRIVATE ecl_bitarray
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)