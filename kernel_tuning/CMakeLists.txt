set(TARGET ecl_tune_kernels)


add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/tune_kernels.cu")  

include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/bitarray/include" "${CMAKE_SOURCE_DIR}/utils/include" "${CMAKE_SOURCE_DIR}/tree/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)


target_link_libraries(${TARGET}
    PRIVATE ecl_utils
    PRIVATE ecl_bitarray
    PRIVATE ecl_wavelet_tree
    PRIVATE ecl_test_bm_utils
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)