cmake_minimum_required(VERSION 3.20)

file(REMOVE_RECURSE "${TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${TEST_BINARY_DIR}")
set(prefix "${TEST_BINARY_DIR}/prefix")

foreach(artifact generated.py graph.dot)
    if(EXISTS "${SOURCE_DIR}/${artifact}")
        set(${artifact}_existed TRUE)
        file(SHA256 "${SOURCE_DIR}/${artifact}" ${artifact}_sha256)
    else()
        set(${artifact}_existed FALSE)
    endif()
endforeach()

set(real_home_thiran "$ENV{HOME}/.local/bin/thiran")
if(EXISTS "${real_home_thiran}")
    set(real_home_thiran_existed TRUE)
    file(SHA256 "${real_home_thiran}" real_home_thiran_sha256)
else()
    set(real_home_thiran_existed FALSE)
endif()

function(require_equal actual expected label)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR "${label}: expected=[${expected}] actual=[${actual}]")
    endif()
endfunction()

execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BINARY_DIR}" --prefix "${prefix}"
                RESULT_VARIABLE result OUTPUT_VARIABLE install_out ERROR_VARIABLE error)
require_equal("${result}" "0" "install")
set(thiran "${prefix}/bin/thiran")
if(NOT EXISTS "${thiran}" OR EXISTS "${prefix}/bin/Thiran")
    message(FATAL_ERROR "installation must contain only lowercase thiran")
endif()
file(GLOB installed_bin_entries LIST_DIRECTORIES TRUE "${prefix}/bin/*")
list(LENGTH installed_bin_entries installed_bin_count)
require_equal("${installed_bin_count}" "1" "installed executable count")
file(READ "${BINARY_DIR}/install_manifest.txt" manifest)
string(STRIP "${manifest}" manifest)
require_equal("${manifest}" "${thiran}" "install manifest")

execute_process(COMMAND "${thiran}" --version WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "version exit")
require_equal("${output}" "Thiran 0.2.0-alpha-dev\n" "version output")
require_equal("${error}" "" "version stderr")

execute_process(COMMAND "${thiran}" --help WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "help")
if(NOT output MATCHES "Thiran --version")
    message(FATAL_ERROR "installed help omits version")
endif()

file(MAKE_DIRECTORY "${TEST_BINARY_DIR}/fake-full" "${TEST_BINARY_DIR}/fake-no-torch" "${TEST_BINARY_DIR}/fake-none")
file(WRITE "${TEST_BINARY_DIR}/fake-full/python3" "#!/bin/sh\ncase \"$2\" in *torch.cuda*) exit 3;; *) exit 0;; esac\n")
file(WRITE "${TEST_BINARY_DIR}/fake-no-torch/python3" "#!/bin/sh\ncase \"$2\" in *torch.cuda*) exit 2;; *) exit 0;; esac\n")
file(CHMOD "${TEST_BINARY_DIR}/fake-full/python3" "${TEST_BINARY_DIR}/fake-no-torch/python3"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH=${TEST_BINARY_DIR}/fake-full" "${thiran}" doctor
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "doctor full")
if(NOT output MATCHES "torch: available" OR NOT output MATCHES "cuda: not-visible" OR NOT output MATCHES "mode: full")
    message(FATAL_ERROR "full doctor output invalid: ${output}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH=${TEST_BINARY_DIR}/fake-no-torch" "${thiran}" doctor
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT output MATCHES "torch: unavailable" OR NOT output MATCHES "mode: compiler-only")
    message(FATAL_ERROR "missing Torch output invalid")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH=${TEST_BINARY_DIR}/fake-none" "${thiran}" doctor
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT output MATCHES "python: unavailable" OR NOT output MATCHES "mode: compiler-only")
    message(FATAL_ERROR "missing Python output invalid")
endif()

execute_process(COMMAND "${thiran}" doctor --help RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "doctor help exit")
require_equal("${output}" "Usage:\n  thiran doctor\n" "doctor help")
require_equal("${error}" "" "doctor help stderr")
execute_process(COMMAND "${thiran}" doctor bad RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "2" "doctor invalid exit")
require_equal("${output}" "" "doctor invalid stdout")
require_equal("${error}" "Invalid doctor command\nUsage:\n  thiran doctor\n" "doctor invalid")

execute_process(COMMAND "${thiran}" --plan "${SOURCE_DIR}/examples/cnn.th" WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "installed plan")
execute_process(COMMAND "${thiran}" --emit-plan "${SOURCE_DIR}/examples/cnn.th" "${TEST_BINARY_DIR}/plan.txt"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "installed emit plan")
execute_process(COMMAND "${thiran}" --emit-region-executor "${SOURCE_DIR}/examples/cnn.th" "${TEST_BINARY_DIR}/executor.py"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "installed emit executor")
if(NOT EXISTS "${TEST_BINARY_DIR}/executor.py")
    message(FATAL_ERROR "executor absent")
endif()
file(SIZE "${TEST_BINARY_DIR}/executor.py" executor_size)
if(executor_size EQUAL 0)
    message(FATAL_ERROR "executor is empty")
endif()

file(WRITE "${prefix}/sentinel" "keep")
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${BINARY_DIR}" --target uninstall RESULT_VARIABLE result)
require_equal("${result}" "0" "uninstall")
if(EXISTS "${thiran}" OR NOT EXISTS "${prefix}/sentinel")
    message(FATAL_ERROR "uninstall scope failure")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BINARY_DIR}" --prefix "${prefix}" RESULT_VARIABLE result)
file(WRITE "${BINARY_DIR}/install_manifest.txt" "/etc/passwd\n")
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${BINARY_DIR}" --target uninstall RESULT_VARIABLE unsafe_result OUTPUT_QUIET ERROR_QUIET)
if(unsafe_result EQUAL 0)
    message(FATAL_ERROR "unsafe manifest accepted")
endif()

execute_process(COMMAND "${SOURCE_DIR}/scripts/install.sh" --help WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "install help exit")
require_equal("${output}" "Usage: ./scripts/install.sh [--prefix <path>]\n" "install help")
require_equal("${error}" "" "install help stderr")

set(help_home "${TEST_BINARY_DIR}/help-home")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "HOME=${help_home}" "${SOURCE_DIR}/scripts/install.sh" --help
                WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "controlled-home install help")
if(EXISTS "${help_home}")
    message(FATAL_ERROR "install help performed filesystem work")
endif()

set(test_home "${TEST_BINARY_DIR}/test-home")
file(MAKE_DIRECTORY "${test_home}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "HOME=${test_home}" "${SOURCE_DIR}/scripts/install.sh"
                WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "install script default prefix")
if(NOT EXISTS "${test_home}/.local/bin/thiran")
    message(FATAL_ERROR "default install did not use controlled HOME")
endif()
if(NOT output MATCHES "Installed thiran to ${test_home}/.local/bin/thiran")
    message(FATAL_ERROR "default install success output invalid: ${output}")
endif()

execute_process(COMMAND "${SOURCE_DIR}/scripts/install.sh" --prefix "${TEST_BINARY_DIR}/script-prefix"
                WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "install script custom prefix")
if(NOT EXISTS "${TEST_BINARY_DIR}/script-prefix/bin/thiran")
    message(FATAL_ERROR "install script did not install lowercase executable")
endif()

set(compiler_only_build "${TEST_BINARY_DIR}/compiler-only-build")
set(compiler_only_prefix "${TEST_BINARY_DIR}/compiler-only-prefix")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${compiler_only_build}" -G Ninja
            -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
)
require_equal("${result}" "0" "compiler-only configure")
set(configure_log "${output}${error}")
if(configure_log MATCHES "Python3|THIRAN_PYTHON_EXECUTABLE|Torch import")
    message(FATAL_ERROR "compiler-only configure selected Python or checked Torch: ${configure_log}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${compiler_only_build}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "compiler-only build")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${compiler_only_build}" --prefix "${compiler_only_prefix}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "compiler-only install")
if(NOT EXISTS "${compiler_only_prefix}/bin/thiran")
    message(FATAL_ERROR "compiler-only installation missing thiran")
endif()

set(second_prefix "${TEST_BINARY_DIR}/second-prefix")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BINARY_DIR}" --prefix "${second_prefix}"
                WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "second-directory install")
execute_process(COMMAND "${second_prefix}/bin/thiran" --version
                WORKING_DIRECTORY "${TEST_BINARY_DIR}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
require_equal("${result}" "0" "second-directory invocation")
require_equal("${output}" "Thiran 0.2.0-alpha-dev\n" "second-directory version")

foreach(token sudo curl wget .bashrc .profile)
    file(READ "${SOURCE_DIR}/scripts/install.sh" script)
    if(script MATCHES "${token}")
        message(FATAL_ERROR "install script contains prohibited token ${token}")
    endif()
endforeach()

foreach(artifact generated.py graph.dot)
    if(${artifact}_existed)
        if(NOT EXISTS "${SOURCE_DIR}/${artifact}")
            message(FATAL_ERROR "repository ${artifact} was removed")
        endif()
        file(SHA256 "${SOURCE_DIR}/${artifact}" final_sha256)
        if(NOT "${final_sha256}" STREQUAL "${${artifact}_sha256}")
            message(FATAL_ERROR "repository ${artifact} was modified")
        endif()
    elseif(EXISTS "${SOURCE_DIR}/${artifact}")
        message(FATAL_ERROR "repository ${artifact} was created")
    endif()
endforeach()
if(real_home_thiran_existed)
    if(NOT EXISTS "${real_home_thiran}")
        message(FATAL_ERROR "default-prefix test removed the real HOME installation")
    endif()
    file(SHA256 "${real_home_thiran}" final_real_home_thiran_sha256)
    if(NOT "${final_real_home_thiran_sha256}" STREQUAL "${real_home_thiran_sha256}")
        message(FATAL_ERROR "default-prefix test modified the real HOME installation")
    endif()
elseif(EXISTS "${real_home_thiran}")
    message(FATAL_ERROR "default-prefix test created an installation in the real HOME")
endif()
