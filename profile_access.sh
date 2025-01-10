#!/bin/bash

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON -DPROFILE=ON -S . -B ./build

cmake --build ./build --target ecl_access_profile

ncu --export "./access_1G_4_5000" --force-overwrite --launch-count 4 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_access_profile 1000000000 4 5000

ncu --export "./access_1G_150_5000" --force-overwrite --launch-count 4 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_access_profile 1000000000 150 5000

ncu --export "./access_500M_20000_5000" --force-overwrite --launch-count 4 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_access_profile 500000000 20000 5000

nvprof --export-profile "./access_1G_4_5000_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_access_profile 1000000000 4 5000

nvprof --export-profile "./access_1G_150_5000_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_access_profile 1000000000 150 5000

nvprof --export-profile "./access_500M_20000_5000_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_access_profile 500000000 20000 5000