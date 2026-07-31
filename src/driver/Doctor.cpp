#include "driver/Doctor.hpp"

#include <array>
#include <cerrno>
#include <filesystem>
#include <fcntl.h>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "thiran/Version.hpp"

extern char** environ;

namespace thiran::driver
{
namespace
{
int run(const std::string& program, const std::string& code)
{
    posix_spawn_file_actions_t actions;
    if(posix_spawn_file_actions_init(&actions) != 0) return -1;
    const int nullFd = ::open("/dev/null", O_WRONLY);
    if(nullFd < 0)
    {
        posix_spawn_file_actions_destroy(&actions);
        return -1;
    }
    posix_spawn_file_actions_adddup2(&actions, nullFd, STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, nullFd, STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, nullFd);

    std::array<char*, 4> arguments{
        const_cast<char*>(program.c_str()),
        const_cast<char*>("-c"),
        const_cast<char*>(code.c_str()),
        nullptr
    };
    pid_t child = 0;
    const int spawned = posix_spawnp(
        &child,
        program.c_str(),
        &actions,
        nullptr,
        arguments.data(),
        environ
    );
    posix_spawn_file_actions_destroy(&actions);
    ::close(nullFd);
    if(spawned != 0) return -1;

    int status = 0;
    while(waitpid(child, &status, 0) < 0)
    {
        if(errno != EINTR) return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

bool temporaryDirectoryWorks()
{
    std::error_code error;
    const auto directory = std::filesystem::temp_directory_path(error);
    if(error || directory.empty()) return false;
    std::string pattern = (directory / "thiran-doctor-XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int descriptor = mkstemp(writable.data());
    if(descriptor < 0) return false;
    const bool closed = ::close(descriptor) == 0;
    const bool removed = ::unlink(writable.data()) == 0;
    return closed && removed;
}
}

DoctorProbeResult Doctor::probe()
{
#if defined(__linux__)
    const bool platform = true;
#else
    const bool platform = false;
#endif
    DoctorProbeResult result{platform, temporaryDirectoryWorks(), false, false, false};
    std::string python;
    for(const char* candidate : {"python3", "python"})
    {
        if(run(candidate, "raise SystemExit(0)") == 0)
        {
            python = candidate;
            result.pythonAvailable = true;
            break;
        }
    }
    if(result.pythonAvailable)
    {
        const int torch = run(
            python,
            "try:\n import torch\nexcept Exception:\n raise SystemExit(2)\n"
            "raise SystemExit(0 if torch.cuda.is_available() else 3)"
        );
        result.torchAvailable = torch == 0 || torch == 3;
        result.cudaVisible = torch == 0;
    }
    return result;
}

std::string Doctor::format(const DoctorProbeResult& result)
{
    const bool usable = result.supportedPlatform && result.temporaryDirectoryAvailable;
    const bool full = usable && result.pythonAvailable && result.torchAvailable;
    return "Thiran doctor\n"
        "  compiler: ok (" + std::string(version::string) + ")\n"
        "  platform: " + (result.supportedPlatform ? "linux" : "unsupported") + "\n"
        "  temporary-directory: " + (result.temporaryDirectoryAvailable ? "ok" : "unavailable") + "\n"
        "  python: " + (result.pythonAvailable ? "available" : "unavailable") + "\n"
        "  torch: " + (!result.pythonAvailable ? "not-checked" : (result.torchAvailable ? "available" : "unavailable")) + "\n"
        "  cuda: " + (!result.torchAvailable ? "not-checked" : (result.cudaVisible ? "visible" : "not-visible")) + "\n"
        "  mode: " + (!usable ? "unusable" : (full ? "full" : "compiler-only")) + "\n";
}

int Doctor::exitCode(const DoctorProbeResult& result) noexcept
{
    return result.supportedPlatform && result.temporaryDirectoryAvailable ? 0 : 1;
}
}
