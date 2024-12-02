#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build"
TEST_BINARY_BA="ecl_BA_RS_tests"  # Update this to match your test executable name
TEST_BINARY_WT="ecl_WT_tests"
COMPUTE_SANITIZER="compute-sanitizer"

# Function to print colored status messages
print_status() {
    echo -e "${GREEN}=== $1 ===${NC}"
}

# Function to handle errors
error_handler() {
    echo -e "${RED}Error occurred in script at line: $1${NC}"
    exit 1
}

trap 'error_handler ${LINENO}' ERR

# Build project with tests
print_status "Building project"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -S . -B ${BUILD_DIR}
cmake --build ${BUILD_DIR} --target ${TEST_BINARY_BA}
cmake --build ${BUILD_DIR} --target ${TEST_BINARY_WT}

# Run tests
print_status "Running tests"
./${BUILD_DIR}/bitarray/tests/${TEST_BINARY_BA}
./${BUILD_DIR}/tree/tests/${TEST_BINARY_WT}

cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON -S . -B ${BUILD_DIR}
cmake --build ${BUILD_DIR} --target ${TEST_BINARY_BA}
cmake --build ${BUILD_DIR} --target ${TEST_BINARY_WT}
# Run Compute Sanitizer checks
print_status "Running Compute Sanitizer - Memory Check"
${COMPUTE_SANITIZER} --tool memcheck ./${BUILD_DIR}/bitarray/tests/${TEST_BINARY_BA} --gtest_brief=1
${COMPUTE_SANITIZER} --tool memcheck ./${BUILD_DIR}/tree/tests/${TEST_BINARY_WT} --gtest_brief=1

print_status "Running Compute Sanitizer - Race Check"
${COMPUTE_SANITIZER} --tool racecheck ./${BUILD_DIR}/bitarray/tests/${TEST_BINARY_BA} --gtest_brief=1
${COMPUTE_SANITIZER} --tool racecheck ./${BUILD_DIR}/tree/tests/${TEST_BINARY_WT} --gtest_brief=1

print_status "Running Compute Sanitizer - Initialize Check"
${COMPUTE_SANITIZER} --tool initcheck ./${BUILD_DIR}/bitarray/tests/${TEST_BINARY_BA} --gtest_brief=1
${COMPUTE_SANITIZER} --tool initcheck ./${BUILD_DIR}/tree/tests/${TEST_BINARY_WT} --gtest_brief=1

print_status "Running Compute Sanitizer - Sync Check"
${COMPUTE_SANITIZER} --tool synccheck ./${BUILD_DIR}/bitarray/tests/${TEST_BINARY_BA} --gtest_brief=1
${COMPUTE_SANITIZER} --tool synccheck ./${BUILD_DIR}/tree/tests/${TEST_BINARY_WT} --gtest_brief=1

print_status "All checks completed successfully!"