set(TARGET ecl_bitarray)

# Collect all the source files
file(GLOB BA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(${TARGET} STATIC ${BA_SOURCES})

include_directories(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_include_directories(${TARGET}
    PRIVATE OpenMP_<CXX>_INCLUDE_DIRS
    PRIVATE CUDAToolkit_INCLUDE_DIRS
)


target_link_libraries(${TARGET}
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)