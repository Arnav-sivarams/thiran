cmake_minimum_required(VERSION 3.20)

file(READ "${CMAKE_SOURCE_DIR}/VERSION" THIRAN_VERSION_FILE)
if(NOT THIRAN_VERSION_FILE STREQUAL "0.2.0-alpha-dev\n")
    message(FATAL_ERROR "VERSION must contain exactly 0.2.0-alpha-dev and one newline")
endif()
string(STRIP "${THIRAN_VERSION_FILE}" THIRAN_VERSION)
if(NOT THIRAN_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-([0-9A-Za-z.-]+)$")
    message(FATAL_ERROR "VERSION must use numeric SemVer with a prerelease suffix")
endif()
set(THIRAN_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(THIRAN_VERSION_MINOR "${CMAKE_MATCH_2}")
set(THIRAN_VERSION_PATCH "${CMAKE_MATCH_3}")
set(THIRAN_VERSION_PRERELEASE "${CMAKE_MATCH_4}")

project(Thiran VERSION "${THIRAN_VERSION_MAJOR}.${THIRAN_VERSION_MINOR}.${THIRAN_VERSION_PATCH}")

configure_file(
    include/thiran/Version.hpp.in
    generated/thiran/Version.hpp
    @ONLY
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include "${CMAKE_BINARY_DIR}/generated")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

set(CORE_SOURCES ${SOURCES})
list(FILTER CORE_SOURCES EXCLUDE REGEX "/main\\.cpp$")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/frontend/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/semantic/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/analysis/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/storage/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/backend/v0/")

add_library(thiran_v0_frontend STATIC
    src/frontend/v0/Diagnostic.cpp
    src/frontend/v0/Lexer.cpp
    src/frontend/v0/Ast.cpp
    src/frontend/v0/Parser.cpp)

add_library(thiran_v0_semantic STATIC
    src/semantic/v0/Ir.cpp
    src/semantic/v0/Analyzer.cpp
    src/semantic/v0/Verifier.cpp
    src/semantic/v0/Evaluator.cpp)
target_link_libraries(thiran_v0_semantic PUBLIC thiran_v0_frontend)

add_library(thiran_v0_analysis STATIC src/analysis/v0/Ownership.cpp)
target_link_libraries(thiran_v0_analysis PUBLIC thiran_v0_semantic)

add_library(thiran_v0_storage STATIC src/storage/v0/Storage.cpp)
target_link_libraries(thiran_v0_storage PUBLIC thiran_v0_analysis)

add_library(thiran_v0_native_cpu STATIC src/backend/v0/NativeCpu.cpp)
target_link_libraries(thiran_v0_native_cpu PUBLIC thiran_v0_storage)

add_library(thiran_core STATIC ${CORE_SOURCES})

add_executable(Thiran src/main.cpp)
target_link_libraries(Thiran PRIVATE thiran_core)

install(PROGRAMS "$<TARGET_FILE:Thiran>" DESTINATION bin RENAME thiran)
configure_file(cmake/Uninstall.cmake.in cmake_uninstall.cmake @ONLY)
add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_BINARY_DIR}/cmake_uninstall.cmake"
)

include(CTest)
option(THIRAN_V0_ONLY_TESTING "Configure only the independent V0 frontend tests" OFF)

if(BUILD_TESTING)
    add_executable(v0_frontend_tests tests/frontend/v0/FrontendTests.cpp)
    target_link_libraries(v0_frontend_tests PRIVATE thiran_v0_frontend)
    add_test(NAME V0FrontendTests COMMAND v0_frontend_tests)

    add_executable(v0_semantic_tests tests/semantic/v0/SemanticTests.cpp)
    target_link_libraries(v0_semantic_tests PRIVATE thiran_v0_semantic)
    add_test(NAME V0SemanticTests COMMAND v0_semantic_tests)

    add_executable(v0_ownership_tests tests/analysis/v0/OwnershipTests.cpp)
    target_link_libraries(v0_ownership_tests PRIVATE thiran_v0_analysis)
    add_test(NAME V0OwnershipTests COMMAND v0_ownership_tests)

    add_executable(v0_storage_tests tests/storage/v0/StorageTests.cpp)
    target_link_libraries(v0_storage_tests PRIVATE thiran_v0_storage)
    add_test(NAME V0StorageTests COMMAND v0_storage_tests)

    add_executable(v0_native_cpu_tests tests/backend/v0/NativeCpuTests.cpp)
    target_link_libraries(v0_native_cpu_tests PRIVATE thiran_v0_native_cpu)
    target_compile_definitions(v0_native_cpu_tests PRIVATE
        TH008_CXX="${CMAKE_CXX_COMPILER}"
        TH008_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}"
        TH008_INCLUDE="${CMAKE_SOURCE_DIR}/include"
        TH008_BUILD="${CMAKE_BINARY_DIR}")
    add_test(NAME V0NativeCpuTests COMMAND v0_native_cpu_tests)

if(NOT THIRAN_V0_ONLY_TESTING)
    add_executable(rewrite_regression_tests tests/optimizer/RewriteRegressionTests.cpp)
    target_link_libraries(rewrite_regression_tests PRIVATE thiran_core)
    add_test(NAME RewriteRegressionTests COMMAND rewrite_regression_tests)

    if(THIRAN_PYTHON_EXECUTABLE)
        set(THIRAN_TEST_PYTHON "${THIRAN_PYTHON_EXECUTABLE}")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
        set(THIRAN_TEST_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    else()
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        set(THIRAN_TEST_PYTHON "${Python3_EXECUTABLE}")
    endif()

    execute_process(
        COMMAND "${THIRAN_TEST_PYTHON}" -c "import torch"
        RESULT_VARIABLE THIRAN_TORCH_IMPORT_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT THIRAN_TORCH_IMPORT_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Selected Python interpreter cannot import torch: "
            "${THIRAN_TEST_PYTHON}")
    endif()

    add_executable(
        region_ir_tests
        tests/region/RegionIRTests.cpp
    )

    target_link_libraries(
        region_ir_tests
        PRIVATE
        thiran_core
    )

    add_test(
        NAME RegionIRTests
        COMMAND region_ir_tests
    )

    add_executable(
        region_formation_tests
        tests/region/RegionFormationTests.cpp
    )

    target_link_libraries(
        region_formation_tests
        PRIVATE
        thiran_core
    )

    add_test(
        NAME RegionFormationTests
        COMMAND region_formation_tests
    )

    add_executable(
        strategy_classification_tests
        tests/region/StrategyClassificationTests.cpp
    )

    target_link_libraries(
        strategy_classification_tests
        PRIVATE
        thiran_core
    )

    add_test(
        NAME StrategyClassificationTests
        COMMAND strategy_classification_tests
    )

    add_executable(
        region_plan_tests
        tests/region/RegionPlanTests.cpp
    )

    target_link_libraries(
        region_plan_tests
        PRIVATE
        thiran_core
    )

    add_test(
        NAME RegionPlanTests
        COMMAND region_plan_tests
    )

    add_executable(
        region_plan_cli_support_tests
        tests/region/RegionPlanCliSupportTests.cpp
    )

    target_link_libraries(
        region_plan_cli_support_tests
        PRIVATE
        thiran_core
    )

    add_test(
        NAME RegionPlanCliSupportTests
        COMMAND region_plan_cli_support_tests
    )

    add_test(
        NAME RegionPlanCliTests
        COMMAND
            ${CMAKE_COMMAND}
            -DTHIRAN_EXECUTABLE=$<TARGET_FILE:Thiran>
            -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DTEST_BINARY_DIR=${CMAKE_BINARY_DIR}/region-plan-cli-tests
            -P
            ${CMAKE_SOURCE_DIR}/tests/integration/RegionPlanCliTests.cmake
    )

    add_executable(
        region_python_emitter_tests
        tests/region/RegionPythonEmitterTests.cpp
    )

    target_link_libraries(
        region_python_emitter_tests
        PRIVATE
        thiran_core
    )

    add_test(
        NAME RegionPythonEmitterTests
        COMMAND region_python_emitter_tests
    )

    add_test(
        NAME RegionExecutionTests
        COMMAND
            ${CMAKE_COMMAND}
            -DTHIRAN_EXECUTABLE=$<TARGET_FILE:Thiran>
            -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DTEST_BINARY_DIR=${CMAKE_BINARY_DIR}/region-execution-tests
            -DPYTHON_EXECUTABLE=${THIRAN_TEST_PYTHON}
            -P
            ${CMAKE_SOURCE_DIR}/tests/integration/RegionExecutionTests.cmake
    )

    add_executable(
        region_python_runtime_tests
        tests/region/RegionPythonRuntimeTests.cpp
    )
    target_link_libraries(
        region_python_runtime_tests
        PRIVATE
        thiran_core
    )
    add_test(
        NAME RegionPythonRuntimeTests
        COMMAND region_python_runtime_tests
    )

    add_test(
        NAME RegionRuntimeCliTests
        COMMAND
            ${CMAKE_COMMAND}
            -DTHIRAN_EXECUTABLE=$<TARGET_FILE:Thiran>
            -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
            -DTEST_BINARY_DIR=${CMAKE_BINARY_DIR}/region-runtime-cli-tests
            -DPYTHON_EXECUTABLE=${THIRAN_TEST_PYTHON}
            -DRUNTIME_TEST_EXECUTABLE=$<TARGET_FILE:region_python_runtime_tests>
            -P
            ${CMAKE_SOURCE_DIR}/tests/integration/RegionRuntimeCliTests.cmake
    )

    add_executable(installation_support_tests tests/driver/InstallationSupportTests.cpp)
    target_link_libraries(installation_support_tests PRIVATE thiran_core)
    add_test(NAME InstallationSupportTests COMMAND installation_support_tests)

    add_test(NAME InstallationCliTests COMMAND ${CMAKE_COMMAND}
        -DTHIRAN_EXECUTABLE=$<TARGET_FILE:Thiran>
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTEST_BINARY_DIR=${CMAKE_BINARY_DIR}/installation-cli-tests
        -P ${CMAKE_SOURCE_DIR}/tests/integration/InstallationCliTests.cmake)

    add_executable(source_manager_tests tests/frontend/SourceManagerTests.cpp)
    target_link_libraries(source_manager_tests PRIVATE thiran_core)
    add_test(NAME SourceManagerTests COMMAND source_manager_tests)

    add_executable(module_linker_tests tests/frontend/ModuleLinkerTests.cpp)
    target_compile_definitions(module_linker_tests PRIVATE THIRAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(module_linker_tests PRIVATE thiran_core)
    add_test(NAME ModuleLinkerTests COMMAND module_linker_tests)

    add_test(NAME ModuleCliTests COMMAND ${CMAKE_COMMAND}
        -DTHIRAN_EXECUTABLE=$<TARGET_FILE:Thiran>
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTEST_BINARY_DIR=${CMAKE_BINARY_DIR}/module-cli-tests
        -DPYTHON_EXECUTABLE=${THIRAN_TEST_PYTHON}
        -P ${CMAKE_SOURCE_DIR}/tests/integration/ModuleCliTests.cmake)

    add_executable(function_semantic_tests tests/frontend/FunctionSemanticTests.cpp)
    target_link_libraries(function_semantic_tests PRIVATE thiran_core)
    add_test(NAME FunctionSemanticTests COMMAND function_semantic_tests)

    add_executable(function_lowering_tests tests/frontend/FunctionLoweringTests.cpp)
    target_compile_definitions(function_lowering_tests PRIVATE THIRAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_link_libraries(function_lowering_tests PRIVATE thiran_core)
    add_test(NAME FunctionLoweringTests COMMAND function_lowering_tests)

    add_test(NAME FunctionIntegrationTests COMMAND ${CMAKE_COMMAND}
        -DTHIRAN_EXECUTABLE=$<TARGET_FILE:Thiran>
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTEST_BINARY_DIR=${CMAKE_BINARY_DIR}/function-integration-tests
        -DPYTHON_EXECUTABLE=${THIRAN_TEST_PYTHON}
        -P ${CMAKE_SOURCE_DIR}/tests/integration/FunctionIntegrationTests.cmake)
endif()
endif()
