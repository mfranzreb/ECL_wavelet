set(TARGET ecl_WT_tests)

# Collect all the source files in the ./tests directory
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_executable(${TARGET} ${TEST_SOURCES})

# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_SOURCE_DIR}/bitarray/include" "${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include")


# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_wavelet_tree
    OpenMP::OpenMP_CXX
    CUDA::cudart
    gtest
    gtest_main
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)