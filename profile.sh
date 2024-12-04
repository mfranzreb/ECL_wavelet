#!/bin/bash

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON -DPROFILE=ON -S . -B ./build

cmake --build ./build --target ecl_RS_profile

#ncu --export "./RS_construction_1Mbit" --force-overwrite --launch-skip 1 --launch-count 2 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/bitarray/benchmark/ecl_RS_profile 1000000

ncu --export "./RS_construction_1Gbit" --force-overwrite --launch-skip 1 --launch-count 2 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/bitarray/benchmark/ecl_RS_profile 1000000000 1

ncu --export "./RS_construction_10Gbit" --force-overwrite --launch-skip 1 --launch-count 2 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/bitarray/benchmark/ecl_RS_profile 10000000000 1

nvprof --export-profile "./RS_construction_200Mbitx5_vprof.prof" --trace api,gpu ./build/bitarray/benchmark/ecl_RS_profile 200000000 5

nvprof --export-profile "./RS_construction_500Mbitx10_vprof.prof" --trace api,gpu ./build/bitarray/benchmark/ecl_RS_profile 500000000 10