#!/bin/bash

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON -DPROFILE=ON -S . -B ./build

cmake --build ./build --target ecl_rank_profile

ncu --export "./rank_1G_4_5000" --force-overwrite --kernel-id ::rankKernel: --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_rank_profile 1000000000 4 5000

ncu --export "./rank_1G_150_5000" --force-overwrite --kernel-id ::rankKernel: --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_rank_profile 1000000000 150 5000

ncu --export "./rank_500M_20000_5000" --force-overwrite --kernel-id ::rankKernel: --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_rank_profile 500000000 20000 5000

nvprof --profile-from-start off --export-profile "./rank_1G_4_5000_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_rank_profile 1000000000 4 5000

nvprof --profile-from-start off --export-profile "./rank_1G_150_5000_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_rank_profile 1000000000 150 5000

nvprof --profile-from-start off --export-profile "./rank_500M_20000_5000_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_rank_profile 500000000 20000 5000