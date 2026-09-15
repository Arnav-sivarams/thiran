#include "frontend/v0/Token.hpp"
#include <sstream>

namespace thiran::v0 {
std::string Diagnostic::format() const {
    std::ostringstream out;
    out << source << ':' << span.begin.line << ':' << span.begin.column << ": " << message;
    return out.str();
}
}
