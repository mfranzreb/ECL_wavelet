#include <cstdint>
#include <string>
#include <unordered_map>
#include <utils.cuh>

namespace ecl {

std::unordered_map<std::string, IdealConfigs> configs{
    {"Quadro P600",
     IdealConfigs{.ideal_TPB_calculateL2EntriesKernel = 64,
                  .ideal_TPB_computeGlobalHistogramKernel = 1024,
                  .ideal_tot_threads_computeGlobalHistogramKernel = 622592,
                  .ideal_TPB_fillLevelKernel = 256,
                  .ideal_tot_threads_fillLevelKernel = 98304}},
};
}  // namespace ecl