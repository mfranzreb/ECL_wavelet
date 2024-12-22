set(TARGET ecl_wavelet_tree)


add_library(${TARGET} STATIC "${CMAKE_CURRENT_SOURCE_DIR}/include/wavelet_tree.cuh")  

include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/bitarray/include" "${CMAKE_SOURCE_DIR}/utils/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)


target_link_libraries(${TARGET}
    PRIVATE ecl_utils
    PRIVATE ecl_bitarray
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)