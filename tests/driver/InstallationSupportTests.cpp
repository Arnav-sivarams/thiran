#include <iostream>
#include <string>

#include "driver/Doctor.hpp"
#include "thiran/Version.hpp"

namespace
{
int failures = 0;
#define CHECK(value) do { if(!(value)) { ++failures; std::cerr << "Failure at line " << __LINE__ << "\n"; } } while(false)

void testVersion()
{
    CHECK(thiran::version::string == "0.2.0-alpha-dev");
    CHECK(thiran::version::major == 0);
    CHECK(thiran::version::minor == 2);
    CHECK(thiran::version::patch == 0);
    CHECK(thiran::version::prerelease == "alpha-dev");
    CHECK(std::string("Thiran ") + std::string(thiran::version::string) ==
          "Thiran 0.2.0-alpha-dev");
}

void testReports()
{
    using thiran::driver::Doctor;
    using thiran::driver::DoctorProbeResult;
    const DoctorProbeResult full{true, true, true, true, true};
    const std::string expected =
        "Thiran doctor\n"
        "  compiler: ok (0.2.0-alpha-dev)\n"
        "  platform: linux\n"
        "  temporary-directory: ok\n"
        "  python: available\n"
        "  torch: available\n"
        "  cuda: visible\n"
        "  mode: full\n";
    CHECK(Doctor::format(full) == expected);
    CHECK(Doctor::exitCode(full) == 0);
    CHECK(Doctor::format({true, true, false, false, false}).find("mode: compiler-only") != std::string::npos);
    CHECK(Doctor::format({true, true, false, false, false}).find("torch: not-checked") != std::string::npos);
    CHECK(Doctor::format({true, true, true, false, false}).find("torch: unavailable") != std::string::npos);
    CHECK(Doctor::format({true, true, true, true, false}).find("cuda: not-visible") != std::string::npos);
    CHECK(Doctor::format({false, true, true, true, false}).find("platform: unsupported") != std::string::npos);
    CHECK(Doctor::format({true, false, true, true, false}).find("temporary-directory: unavailable") != std::string::npos);
    CHECK(Doctor::exitCode({true, true, false, false, false}) == 0);
    CHECK(Doctor::exitCode({false, true, false, false, false}) == 1);
    CHECK(Doctor::exitCode({true, false, false, false, false}) == 1);
    CHECK(expected.find("0x") == std::string::npos);
    for(int iteration = 0; iteration < 50; ++iteration) CHECK(Doctor::format(full) == expected);
}
}

int main()
{
    testVersion();
    testReports();
    if(failures != 0) return 1;
    std::cout << "All Installation Support tests passed\n";
    return 0;
}
