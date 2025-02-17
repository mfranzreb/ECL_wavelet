#include <cstdint>
#include <string>
#include <unordered_map>
#include <utils.cuh>

namespace ecl {

std::unordered_map<std::string, IdealConfigs> configs{{"Quadro P600", IdealConfigs {.accessKernel_logrel = {0.17371779276130064, 1.9999999999999998}.rankKernel_logrel = {0.08685889638065032, 0.9999999999999999}.selectKernel_logrel = {0.08685889638065032, 0.9999999999999999}}},};
}  // namespace ecl