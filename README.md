# GPU implementation of the Wavelet Tree data structure

This repository contains a CUDA C++ implementation of the wavelet tree, which is a compact full text index that can answer access, rank, and select queries (among others) for a text in time logarithmic in the size of the alphabet.

The tree constructor must be called from the host, but the object can be passed by copy to a kernel. The queries have a host and a device API.

## Performance

Given enough parallelism, this implementation is at least twice as fast as the [SDSL](https://github.com/simongog/sdsl-lite) balanced wavelet tree with `rank_support_v5` and `select_support_mcl`.

![Time taken to process different numbers of access queries coming from the CPU
by my implementation and by SDSL, on uniformly randomly distributed texts
of 6GB in size for a variety of alphabet sizes.](results/wt_access_results.png "Time taken to process different numbers of access queries coming from the CPU
by my implementation and by SDSL, on uniformly randomly distributed texts
of 6GB in size for a variety of alphabet sizes.")

![Time taken to process different numbers of rank queries coming from the CPU
by my implementation and by SDSL, on uniformly randomly distributed texts
of 6GB in size for a variety of alphabet sizes.](results/wt_rank_results.png "Time taken to process different numbers of rank queries coming from the CPU
by my implementation and by SDSL, on uniformly randomly distributed texts
of 6GB in size for a variety of alphabet sizes.")

![Time taken to process different numbers of select queries coming from the CPU
by my implementation and by SDSL, on uniformly randomly distributed texts
of 6GB in size for a variety of alphabet sizes.](results/wt_select_results.png "Time taken to process different numbers of select queries coming from the CPU
by my implementation and by SDSL, on uniformly randomly distributed texts
of 6GB in size for a variety of alphabet sizes.")

The underlying binary rank and select support structures have at least 10x more throughput on an NVIDIA RTX 3090 compared to `pasta-wide` and `pasta-flat` from [here](https://github.com/pasta-toolbox/bit_vector) on an AMD Ryzen Threadripper 3970X with 32 cores.

For more information on the implementation and performance results, refer to the Master's thesis (link coming soon) the project is based on.

## Usage

The project uses CMake. The `example` subdirectory contains a simple example of how to use the library.
The `main` branch contains the latest stable version of the code, and only includes source code and tests. The `extended` branch also contains benchmarking and profiling scripts.
