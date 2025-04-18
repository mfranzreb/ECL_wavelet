set(TARGET ecl_bitarray)

# Collect all the source files
file(GLOB BA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/bitarray/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/bitarray/*.cu")

add_library(${TARGET} STATIC ${BA_SOURCES})

include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)


target_link_libraries(${TARGET}
    PRIVATE ecl_utils
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#################################################################################################
set(TARGET ecl_utils)

add_library(${TARGET} STATIC "${CMAKE_CURRENT_SOURCE_DIR}/utils/utils.cu")  

include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/utils")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)


target_link_libraries(${TARGET}
    CUDA::cudart
    OpenMP::OpenMP_CXX
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

################################################################################################

set(TARGET ecl_test_bm_utils)

add_library(${TARGET} STATIC "${CMAKE_CURRENT_SOURCE_DIR}/utils/test_benchmark_utils.cu")  

include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)


target_link_libraries(${TARGET}
    PRIVATE ecl_utils
    PRIVATE ecl_bitarray
    CUDA::cudart
    OpenMP::OpenMP_CXX
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)