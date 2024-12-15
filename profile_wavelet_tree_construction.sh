#!/bin/bash

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON -DPROFILE=ON -S . -B ./build

cmake --build ./build --target ecl_WT_profile

ncu --export "./WT_construction_1G_4" --force-overwrite --launch-count 4 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_WT_profile 1000000000 4 0

ncu --export "./WT_construction_1G_4_min" --force-overwrite --launch-count 4 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_WT_profile 1000000000 4 1

ncu --export "./WT_construction_1G_150" --force-overwrite --launch-count 4 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_WT_profile 1000000000 150 0

ncu --export "./WT_construction_1G_150_min" --force-overwrite --launch-count 4 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_WT_profile 1000000000 150 1

ncu --export "./WT_construction_500M_20000" --force-overwrite --launch-count 4 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_WT_profile 500000000 20000 0

ncu --export "./WT_construction_500M_20000_min" --force-overwrite --launch-count 4 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_WT_profile 500000000 20000 1

nvprof --export-profile "./WT_construction_1G_4_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_WT_profile 1000000000 4

nvprof --export-profile "./WT_construction_1G_150_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_WT_profile 1000000000 150

nvprof --export-profile "./WT_construction_500M_20000_vprof.prof" -f --trace api,gpu ./build/benchmark/ecl_WT_profile 500000000 20000