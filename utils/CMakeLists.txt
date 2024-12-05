set(TARGET ecl_utils)

add_library(${TARGET} STATIC "${CMAKE_SOURCE_DIR}/utils/src/utils.cu")  

include_directories(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(${TARGET}
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)


target_link_libraries(${TARGET}
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)