#!/bin/bash

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON -DPROFILE=ON -S . -B ./build

cmake --build ./build --target ecl_access_profile

ncu --export "./access_1G_4_500000" --force-overwrite --kernel-id ::accessKernel: --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_access_profile 1000000000 4 500000

ncu --export "./access_1G_150_500000" --force-overwrite --kernel-id ::accessKernel: --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_access_profile 1000000000 150 500000

ncu --export "./access_500M_20000_500000" --force-overwrite --kernel-id ::accessKernel: --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_access_profile 500000000 20000 500000

nsys profile --capture-range=cudaProfilerApi --output="./access_1G_4_500000" --force-overwrite=true --trace=cuda,osrt ./build/benchmark/ecl_access_profile 1000000000 4 500000

nsys profile --capture-range=cudaProfilerApi --output="./access_1G_150_500000" --force-overwrite=true --trace=cuda,osrt ./build/benchmark/ecl_access_profile 1000000000 150 500000

nsys profile --capture-range=cudaProfilerApi --output="./access_500M_20000_500000" --force-overwrite=true --trace=cuda,osrt ./build/benchmark/ecl_access_profile 500000000 20000 500000