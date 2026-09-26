#include "CppBaseline.hpp"

#include <stdexcept>

namespace thiran::bench::th023 {

std::vector<float> cppFused(const std::vector<float>& x,
                            const std::vector<float>& y) {
    if (x.size() != y.size()) throw std::invalid_argument("baseline shape mismatch");
    std::vector<float> output(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float a = x[i] + y[i];
        const float b = -a;
        const float c = b * y[i];
        const float d = c - x[i];
        output[i] = d * y[i];
    }
    return output;
}

std::vector<float> cppMaterialized(const std::vector<float>& x,
                                   const std::vector<float>& y) {
    if (x.size() != y.size()) throw std::invalid_argument("baseline shape mismatch");
    std::vector<float> a(x.size()), b(x.size()), c(x.size()), d(x.size()), output(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) a[i] = x[i] + y[i];
    for (std::size_t i = 0; i < x.size(); ++i) b[i] = -a[i];
    for (std::size_t i = 0; i < x.size(); ++i) c[i] = b[i] * y[i];
    for (std::size_t i = 0; i < x.size(); ++i) d[i] = c[i] - x[i];
    for (std::size_t i = 0; i < x.size(); ++i) output[i] = d[i] * y[i];
    return output;
}

} // namespace thiran::bench::th023
