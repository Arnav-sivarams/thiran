#pragma once

#include <iosfwd>

namespace thiran::v0::tooling {

bool isWorkflowCommand(int argc, char* argv[]);
void printPrimaryHelp(std::ostream& output);
int runWorkflowCli(int argc, char* argv[]);

} // namespace thiran::v0::tooling
