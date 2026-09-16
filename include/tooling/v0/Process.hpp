#pragma once

#include <string>
#include <vector>

namespace thiran::v0::tooling {

struct ProcessRequest {
    std::string executable;
    std::vector<std::string> arguments;
};
struct ProcessResult {
    bool launched = false;
    int exitStatus = -1;
    std::string standardOutput;
    std::string standardError;
    std::string launcherError;
};

// POSIX argv execution only. No shell, expansion, or command-string parsing.
ProcessResult runProcess(const ProcessRequest& request);

} // namespace thiran::v0::tooling
