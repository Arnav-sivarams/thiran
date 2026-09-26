#pragma once

#include <cstddef>
#include <vector>

namespace thiran::bench::th023 {

// Competent single-threaded C++20 baselines for the frozen elementwise
// workload. The implementation is compiled separately with the exact flags
// recorded in PERFORMANCE_QUALIFICATION_V0.md.
[[nodiscard]] std::vector<float> cppFused(const std::vector<float>& x,
                                          const std::vector<float>& y);
[[nodiscard]] std::vector<float> cppMaterialized(const std::vector<float>& x,
                                                 const std::vector<float>& y);

} // namespace thiran::bench::th023
