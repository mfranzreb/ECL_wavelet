#!/bin/bash

GPU_ID=0

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON -DPROFILE=ON -S . -B ./build

cmake --build ./build --target ecl_RS_profile

ncu --export "./RS_construction_1Gbit_L2_kernel" --force-overwrite --kernel-id ::calculateL2EntriesKernel: --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_RS_profile 1000000000 1 $GPU_ID

ncu --export "./RS_construction_10Gbit_L2_kernel" --force-overwrite --kernel-id ::calculateL2EntriesKernel: --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_RS_profile 10000000000 1 $GPU_ID

ncu --export "./RS_construction_1Gbit_samples_kernel" --force-overwrite --kernel-id ::calculateSelectSamplesKernel: --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_RS_profile 1000000000 1 $GPU_ID

ncu --export "./RS_construction_10Gbit_samples_kernel" --force-overwrite --kernel-id ::calculateSelectSamplesKernel: --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --import-source yes ./build/benchmark/ecl_RS_profile 10000000000 1 $GPU_ID

nsys profile --capture-range=cudaProfilerApi --output="./RS_construction_200Mbitx5" --force-overwrite=true --trace=cuda,osrt ./build/benchmark/ecl_RS_profile 200000000 5 $GPU_ID

nsys profile --capture-range=cudaProfilerApi --output="./RS_construction_500Mbitx10" --force-overwrite=true --trace=cuda,osrt ./build/benchmark/ecl_RS_profile 500000000 10 $GPU_ID