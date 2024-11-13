set(TARGET ECL_wavelet_tree)

# Collect all the source files
file(GLOB WT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(${TARGET} SHARED ${WT_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/include/wavelet_tree.cuh")  

include_directories(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)


target_link_libraries(${TARGET}
    PRIVATE ECL_bitarray
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_17)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)