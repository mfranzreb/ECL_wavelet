set(TARGET ecl_all_tests)

# Collect all the source files in the ./tests directory
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_executable(${TARGET} ${TEST_SOURCES})

# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/bitarray/include" "${CMAKE_SOURCE_DIR}/tree/include" "${CMAKE_SOURCE_DIR}/utils/include" "${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include")


# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_wavelet_tree
    PRIVATE ecl_bitarray
    PRIVATE ecl_utils
    PRIVATE ecl_test_bm_utils
    OpenMP::OpenMP_CXX
    CUDA::cudart
    gtest
    gtest_main
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

########################################################################################
set(TARGET ecl_WT_tests)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/test_wavelet_tree.cu")

# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/bitarray/include" "${CMAKE_SOURCE_DIR}/tree/include" "${CMAKE_SOURCE_DIR}/utils/include" "${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include")


# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_wavelet_tree
    PRIVATE ecl_utils
    PRIVATE ecl_test_bm_utils
    OpenMP::OpenMP_CXX
    CUDA::cudart
    gtest
    gtest_main
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

########################################################################################
set(TARGET ecl_BA_RS_tests)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/test_bit_array.cu" "${CMAKE_CURRENT_SOURCE_DIR}/test_rank_select.cu")

# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/bitarray/include" "${CMAKE_SOURCE_DIR}/utils/include" "${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include")


# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_bitarray
    PRIVATE ecl_utils
    PRIVATE ecl_test_bm_utils
    OpenMP::OpenMP_CXX
    CUDA::cudart
    gtest
    gtest_main
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

########################################################################################
set(TARGET ecl_utils_tests)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/test_utils.cu")

# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include" "${CMAKE_SOURCE_DIR}/utils/include")


# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_utils
    PRIVATE ecl_test_bm_utils
    CUDA::cudart
    gtest
    gtest_main
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)

set_property(TARGET ${TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)