set(TARGET ecl_example)

add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/example.cu")

# Add the ./include folder to the list of include directories
include_directories(${TARGET} "${CMAKE_SOURCE_DIR}/include")


# target_link_libraries(runTests YourMainProjectLibrary)
target_link_libraries(${TARGET}
    PRIVATE ecl_wavelet_tree
)

target_compile_features(${TARGET} PUBLIC cxx_std_20)