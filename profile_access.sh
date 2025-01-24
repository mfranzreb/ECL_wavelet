#!/bin/bash

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON -DPROFILE=ON -S . -B ./build

cmake --build ./build --target ecl_access_profile
nvprof --profile-from-start off --export-profile "./access_1G_4_500000_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_access_profile 1000000000 4 500000

nvprof --profile-from-start off --export-profile "./access_1G_150_500000_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_access_profile 1000000000 150 500000

nvprof --profile-from-start off --export-profile "./access_500M_20000_500000_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_access_profile 500000000 20000 500000