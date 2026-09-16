#include "tooling/v0/Process.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace thiran::v0::tooling {
namespace {
void appendAvailable(int fd, std::string& output, bool& open) {
    char buffer[4096];
    for (;;) {
        const auto count = ::read(fd, buffer, sizeof(buffer));
        if (count > 0) output.append(buffer, static_cast<std::size_t>(count));
        else if (count == 0) { ::close(fd); open = false; return; }
        else if (errno == EINTR) continue;
        else if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        else { ::close(fd); open = false; return; }
    }
}
}

ProcessResult runProcess(const ProcessRequest& request) {
    ProcessResult result;
    if (request.executable.empty()) { result.launcherError = "empty executable"; return result; }
    int outPipe[2], errPipe[2];
    if (::pipe(outPipe) != 0) {
        result.launcherError = std::string("pipe: ") + std::strerror(errno); return result;
    }
    if (::pipe(errPipe) != 0) {
        result.launcherError = std::string("pipe: ") + std::strerror(errno);
        ::close(outPipe[0]); ::close(outPipe[1]); return result;
    }
    const auto child = ::fork();
    if (child < 0) {
        result.launcherError = std::string("fork: ") + std::strerror(errno);
        ::close(outPipe[0]); ::close(outPipe[1]); ::close(errPipe[0]); ::close(errPipe[1]);
        return result;
    }
    if (child == 0) {
        ::dup2(outPipe[1], STDOUT_FILENO); ::dup2(errPipe[1], STDERR_FILENO);
        ::close(outPipe[0]); ::close(outPipe[1]); ::close(errPipe[0]); ::close(errPipe[1]);
        std::vector<char*> argv;
        argv.reserve(request.arguments.size() + 2);
        argv.push_back(const_cast<char*>(request.executable.c_str()));
        for (const auto& argument : request.arguments) argv.push_back(const_cast<char*>(argument.c_str()));
        argv.push_back(nullptr);
        ::execvp(request.executable.c_str(), argv.data());
        const std::string error = std::string("exec: ") + std::strerror(errno) + "\n";
        const auto written = ::write(STDERR_FILENO, error.data(), error.size());
        (void)written;
        ::_exit(127);
    }
    result.launched = true;
    ::close(outPipe[1]); ::close(errPipe[1]);
    ::fcntl(outPipe[0], F_SETFL, ::fcntl(outPipe[0], F_GETFL) | O_NONBLOCK);
    ::fcntl(errPipe[0], F_SETFL, ::fcntl(errPipe[0], F_GETFL) | O_NONBLOCK);
    bool outOpen = true, errOpen = true;
    while (outOpen || errOpen) {
        pollfd fds[2]{{outPipe[0], static_cast<short>(POLLIN | POLLHUP), 0},
                      {errPipe[0], static_cast<short>(POLLIN | POLLHUP), 0}};
        if (::poll(fds, 2, -1) < 0) { if (errno == EINTR) continue; break; }
        if (outOpen && fds[0].revents) appendAvailable(outPipe[0], result.standardOutput, outOpen);
        if (errOpen && fds[1].revents) appendAvailable(errPipe[0], result.standardError, errOpen);
    }
    int status = 0;
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    if (WIFEXITED(status)) result.exitStatus = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) result.exitStatus = 128 + WTERMSIG(status);
    else result.exitStatus = 125;
    return result;
}

} // namespace thiran::v0::tooling
