#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build"
TEST_BINARY="ecl_all_tests"
COMPUTE_SANITIZER="compute-sanitizer"
IWYU_TOOL_PATH = "~/iwyu_tool.py"

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

# Build for IWYU, only if the tool is available
if [ ! -f ${IWYU_TOOL_PATH} ]; then
    print_status "IWYU tool not found, skipping IWYU checks"
else
    print_status "Building for IWYU"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON -DPROFILE=ON  -S . -B ${BUILD_DIR}
python ${IWYU_TOOL_PATH} -p ${BUILD_DIR} > ./IWYU_output.txt

rm -rf ${BUILD_DIR}
# Build project with tests
print_status "Building project"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DPROFILE=ON -S . -B ${BUILD_DIR}
cmake --build ${BUILD_DIR} --target ${TEST_BINARY}

# Run tests
print_status "Running tests"
./${BUILD_DIR}/tests/${TEST_BINARY}

# Run Compute Sanitizer checks
print_status "Running Compute Sanitizer - Memory Check"
${COMPUTE_SANITIZER} --tool memcheck ./${BUILD_DIR}/tests/${TEST_BINARY}

print_status "Running Compute Sanitizer - Race Check"
${COMPUTE_SANITIZER} --tool racecheck ./${BUILD_DIR}/tests/${TEST_BINARY} --gtest_brief=1

print_status "Running Compute Sanitizer - Initialize Check"
${COMPUTE_SANITIZER} --tool initcheck ./${BUILD_DIR}/tests/${TEST_BINARY} --gtest_brief=1

print_status "Running Compute Sanitizer - Sync Check"
${COMPUTE_SANITIZER} --tool synccheck ./${BUILD_DIR}/tests/${TEST_BINARY} --gtest_brief=1

print_status "All checks completed successfully!"