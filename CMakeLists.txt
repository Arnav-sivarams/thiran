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

option(THIRAN_ENABLE_NATIVE_GPU "Build the native CUDA Driver/PTX backend" OFF)
if(THIRAN_ENABLE_NATIVE_GPU)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "Native GPU backend: enabled (CUDA compiler detected: ${CMAKE_CUDA_COMPILER})")
    else()
        message(STATUS "Native GPU backend: enabled (CUDA compiler not found; low-level Driver/PTX path does not require nvcc)")
    endif()
else()
    message(STATUS "Native GPU backend: disabled")
endif()

configure_file(
    include/thiran/Version.hpp.in
    generated/thiran/Version.hpp
    @ONLY
)

# Convert trusted CMake toolchain flags into C++ argv elements at configure
# time. The developer driver never parses a shell command string at runtime.
separate_arguments(THIRAN_V0_HOST_ARGUMENT_LIST NATIVE_COMMAND
    "${CMAKE_CXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
set(THIRAN_V0_HOST_ARGUMENTS_CPP "")
foreach(THIRAN_V0_HOST_ARGUMENT IN LISTS THIRAN_V0_HOST_ARGUMENT_LIST)
    string(REPLACE "\\" "\\\\" THIRAN_V0_HOST_ARGUMENT_ESCAPED "${THIRAN_V0_HOST_ARGUMENT}")
    string(REPLACE "\"" "\\\"" THIRAN_V0_HOST_ARGUMENT_ESCAPED "${THIRAN_V0_HOST_ARGUMENT_ESCAPED}")
    string(APPEND THIRAN_V0_HOST_ARGUMENTS_CPP "\"${THIRAN_V0_HOST_ARGUMENT_ESCAPED}\",")
endforeach()
configure_file(
    include/tooling/v0/BuildConfig.hpp.in
    generated/tooling/v0/BuildConfig.hpp
    @ONLY
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
list(FILTER CORE_SOURCES EXCLUDE REGEX "/interop/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/tooling/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/autodiff/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/training/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/runtime/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/artifact/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/persistence/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/model/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/graphics/v0/")
list(FILTER CORE_SOURCES EXCLUDE REGEX "/extension/v0/")

add_library(thiran_v0_frontend STATIC
    src/frontend/v0/Diagnostic.cpp
    src/frontend/v0/Lexer.cpp
    src/frontend/v0/Ast.cpp
    src/frontend/v0/Parser.cpp)

add_library(thiran_v0_extension STATIC src/extension/v0/Extension.cpp)
target_link_libraries(thiran_v0_extension PRIVATE ${CMAKE_DL_LIBS})

add_library(thiran_v0_semantic STATIC
    src/semantic/v0/Ir.cpp
    src/semantic/v0/Analyzer.cpp
    src/semantic/v0/Verifier.cpp
    src/semantic/v0/Evaluator.cpp)
target_link_libraries(thiran_v0_semantic PUBLIC thiran_v0_frontend)
target_link_libraries(thiran_v0_semantic PUBLIC thiran_v0_extension)

add_library(thiran_v0_analysis STATIC src/analysis/v0/Ownership.cpp)
target_link_libraries(thiran_v0_analysis PUBLIC thiran_v0_semantic)

add_library(thiran_v0_autodiff STATIC src/autodiff/v0/Autodiff.cpp)
target_link_libraries(thiran_v0_autodiff PUBLIC thiran_v0_analysis)

add_library(thiran_v0_training STATIC src/training/v0/Training.cpp)
target_link_libraries(thiran_v0_training PUBLIC thiran_v0_autodiff)

add_library(thiran_v0_async STATIC src/runtime/v0/Async.cpp)

add_library(thiran_v0_storage STATIC src/storage/v0/Storage.cpp)
target_link_libraries(thiran_v0_storage PUBLIC thiran_v0_analysis thiran_v0_async)

add_library(thiran_v0_graphics STATIC
    src/graphics/v0/Graphics.cpp
    src/graphics/v0/GraphicsGpu.cpp)
target_link_libraries(thiran_v0_graphics PUBLIC thiran_v0_storage)
target_compile_definitions(thiran_v0_graphics PRIVATE
    THIRAN_ENABLE_NATIVE_GPU=$<BOOL:${THIRAN_ENABLE_NATIVE_GPU}>)

add_library(thiran_v0_persistence STATIC src/persistence/v0/Persistence.cpp)
target_link_libraries(thiran_v0_persistence PUBLIC thiran_v0_storage)

add_library(thiran_v0_checkpoint STATIC src/training/v0/Checkpoint.cpp)
target_link_libraries(thiran_v0_checkpoint PUBLIC thiran_v0_training thiran_v0_persistence)

add_library(thiran_v0_tensor_region STATIC src/backend/v0/TensorRegion.cpp)
target_link_libraries(thiran_v0_tensor_region PUBLIC thiran_v0_storage)

add_library(thiran_v0_physical_plan STATIC src/backend/v0/PhysicalPlan.cpp)
target_link_libraries(thiran_v0_physical_plan PUBLIC thiran_v0_tensor_region)

add_library(thiran_v0_native_cpu STATIC src/backend/v0/NativeCpu.cpp)
target_link_libraries(thiran_v0_native_cpu PUBLIC thiran_v0_physical_plan)

add_library(thiran_v0_native_gpu STATIC src/backend/v0/NativeGpu.cpp)
target_link_libraries(thiran_v0_native_gpu PUBLIC thiran_v0_physical_plan PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(thiran_v0_native_gpu PRIVATE
    THIRAN_ENABLE_NATIVE_GPU=$<BOOL:${THIRAN_ENABLE_NATIVE_GPU}>)
target_link_libraries(thiran_v0_graphics PUBLIC thiran_v0_native_gpu)

add_library(thiran_v0_process STATIC src/tooling/v0/Process.cpp)

add_library(thiran_v0_artifact STATIC src/artifact/v0/NativeArtifacts.cpp)
target_link_libraries(thiran_v0_artifact PUBLIC
    thiran_v0_native_cpu thiran_v0_native_gpu thiran_v0_process PRIVATE ${CMAKE_DL_LIBS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Keep build/JIT and execution code independently discardable so the
    # deployment-only model launcher does not retain host-compiler process code.
    target_compile_options(thiran_v0_artifact PRIVATE -ffunction-sections -fdata-sections)
endif()

add_executable(thiran-artifact src/artifact/v0/ArtifactMain.cpp)
target_link_libraries(thiran-artifact PRIVATE thiran_v0_artifact)
install(TARGETS thiran-artifact RUNTIME DESTINATION bin)

add_library(thiran_v0_model_runtime STATIC src/model/v0/ModelBundle.cpp)
target_link_libraries(thiran_v0_model_runtime PUBLIC thiran_v0_artifact thiran_v0_persistence)

add_library(thiran_v0_model_build STATIC src/model/v0/DeploymentBuild.cpp)
target_link_libraries(thiran_v0_model_build PUBLIC thiran_v0_model_runtime thiran_v0_checkpoint)

add_library(thiran_v0_reference_model STATIC src/model/v0/ReferenceModel.cpp)
target_link_libraries(thiran_v0_reference_model PUBLIC thiran_v0_model_build thiran_v0_frontend)

add_executable(thiran-model src/model/v0/ModelMain.cpp)
target_link_libraries(thiran-model PRIVATE thiran_v0_model_runtime)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(thiran-model PRIVATE -Wl,--gc-sections)
endif()
install(TARGETS thiran-model RUNTIME DESTINATION bin)

add_library(thiran_v0_interop STATIC src/interop/v0/NativeLibrary.cpp)
target_link_libraries(thiran_v0_interop PUBLIC thiran_v0_analysis)

add_library(thiran_v0_tooling STATIC
    src/tooling/v0/Driver.cpp
    src/tooling/v0/Cli.cpp)
target_link_libraries(thiran_v0_tooling PUBLIC
    thiran_v0_native_cpu thiran_v0_interop thiran_v0_process
    thiran_v0_artifact thiran_v0_model_runtime)
target_compile_definitions(thiran_v0_tooling PRIVATE
    TH009_CXX="${CMAKE_CXX_COMPILER}"
    TH009_INCLUDE="${CMAKE_SOURCE_DIR}/include"
    TH009_BUILD="${CMAKE_BINARY_DIR}")

add_executable(thiran-v0 src/tooling/v0/Main.cpp)
target_link_libraries(thiran-v0 PRIVATE thiran_v0_tooling)
target_compile_definitions(thiran-v0 PRIVATE
    TH009_CXX="${CMAKE_CXX_COMPILER}"
    TH009_INCLUDE="${CMAKE_SOURCE_DIR}/include"
    TH009_BUILD="${CMAKE_BINARY_DIR}")

add_library(thiran_core STATIC ${CORE_SOURCES})

add_executable(Thiran src/main.cpp)
target_link_libraries(Thiran PRIVATE thiran_core thiran_v0_tooling)
set_target_properties(Thiran PROPERTIES OUTPUT_NAME thiran)

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

    add_executable(v0_autodiff_tests tests/autodiff/v0/AutodiffTests.cpp)
    target_link_libraries(v0_autodiff_tests PRIVATE thiran_v0_autodiff)
    target_compile_definitions(v0_autodiff_tests PRIVATE THIRAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    add_test(NAME V0AutodiffTests COMMAND v0_autodiff_tests)

    add_executable(v0_training_tests tests/training/v0/TrainingTests.cpp)
    target_link_libraries(v0_training_tests PRIVATE thiran_v0_training)
    target_compile_definitions(v0_training_tests PRIVATE THIRAN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    add_test(NAME V0TrainingTests COMMAND v0_training_tests)

    add_executable(v0_scan_tests tests/semantic/v0/ScanTests.cpp)
    target_link_libraries(v0_scan_tests PRIVATE thiran_v0_autodiff thiran_v0_native_cpu)
    add_test(NAME V0ScanTests COMMAND v0_scan_tests)

    add_executable(v0_storage_tests tests/storage/v0/StorageTests.cpp)
    target_link_libraries(v0_storage_tests PRIVATE thiran_v0_storage)
    add_test(NAME V0StorageTests COMMAND v0_storage_tests)

    add_executable(v0_graphics_tests tests/graphics/v0/GraphicsTests.cpp)
    target_link_libraries(v0_graphics_tests PRIVATE thiran_v0_graphics)
    add_test(NAME V0GraphicsTests COMMAND v0_graphics_tests)

    add_executable(v0_graphics_gpu_tests tests/graphics/v0/GraphicsGpuTests.cpp)
    target_link_libraries(v0_graphics_gpu_tests PRIVATE thiran_v0_graphics)
    add_test(NAME V0GraphicsGpuTests COMMAND v0_graphics_gpu_tests)

    add_executable(v0_async_lifetime_tests tests/runtime/v0/AsyncLifetimeTests.cpp)
    target_link_libraries(v0_async_lifetime_tests PRIVATE thiran_v0_storage)
    add_test(NAME V0AsyncLifetimeTests COMMAND v0_async_lifetime_tests)

    add_executable(v0_native_cpu_tests tests/backend/v0/NativeCpuTests.cpp)
    target_link_libraries(v0_native_cpu_tests PRIVATE thiran_v0_native_cpu)
    target_compile_definitions(v0_native_cpu_tests PRIVATE
        TH008_CXX="${CMAKE_CXX_COMPILER}"
        TH008_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}"
        TH008_INCLUDE="${CMAKE_SOURCE_DIR}/include"
        TH008_BUILD="${CMAKE_BINARY_DIR}")
    add_test(NAME V0NativeCpuTests COMMAND v0_native_cpu_tests)

    add_executable(v0_physical_plan_tests tests/backend/v0/PhysicalPlanTests.cpp)
    target_link_libraries(v0_physical_plan_tests PRIVATE thiran_v0_physical_plan thiran_v0_autodiff)
    add_test(NAME V0PhysicalPlanTests COMMAND v0_physical_plan_tests)

    add_executable(v0_native_gpu_tests tests/backend/v0/NativeGpuTests.cpp)
    target_link_libraries(v0_native_gpu_tests PRIVATE thiran_v0_native_gpu)
    add_test(NAME V0NativeGpuTests COMMAND v0_native_gpu_tests)

    add_executable(v0_native_artifact_tests tests/artifact/v0/NativeArtifactTests.cpp)
    target_link_libraries(v0_native_artifact_tests PRIVATE thiran_v0_artifact)
    target_compile_definitions(v0_native_artifact_tests PRIVATE
        TH016_CXX="${CMAKE_CXX_COMPILER}"
        TH016_INCLUDE="${CMAKE_SOURCE_DIR}/include"
        TH016_SOURCE="${CMAKE_SOURCE_DIR}"
        TH016_BUILD="${CMAKE_BINARY_DIR}"
        TH016_RUNTIME="$<TARGET_FILE:thiran-artifact>")
    add_dependencies(v0_native_artifact_tests thiran-artifact)
    add_test(NAME V0NativeArtifactTests COMMAND v0_native_artifact_tests)

    add_executable(v0_model_deployment_tests tests/model/v0/ModelDeploymentTests.cpp)
    target_link_libraries(v0_model_deployment_tests PRIVATE thiran_v0_reference_model thiran_v0_process)
    target_compile_definitions(v0_model_deployment_tests PRIVATE
        TH017_CXX="${CMAKE_CXX_COMPILER}"
        TH017_INCLUDE="${CMAKE_SOURCE_DIR}/include"
        TH017_SOURCE="${CMAKE_SOURCE_DIR}"
        TH017_BUILD="${CMAKE_BINARY_DIR}"
        TH017_RUNTIME="$<TARGET_FILE:thiran-model>"
        TH022_CLI="$<TARGET_FILE:Thiran>")
    add_dependencies(v0_model_deployment_tests thiran-model Thiran)
    add_test(NAME V0ModelDeploymentTests COMMAND v0_model_deployment_tests)

    if(THIRAN_ENABLE_NATIVE_GPU)
        add_executable(v0_extension_gpu_integration_tests
            tests/extension/v0/ExtensionGpuIntegrationTests.cpp)
        target_link_libraries(v0_extension_gpu_integration_tests PRIVATE
            thiran_v0_artifact thiran_v0_process)
        target_compile_definitions(v0_extension_gpu_integration_tests PRIVATE
            TH021_CXX="${CMAKE_CXX_COMPILER}"
            TH021_INCLUDE="${CMAKE_SOURCE_DIR}/include"
            TH021_BUILD="${CMAKE_BINARY_DIR}"
            TH021_EXTENSION="$<TARGET_FILE:th021_research_extension>"
            TH021_RUNTIME="$<TARGET_FILE:th021_extension_artifact_runner>")
        add_dependencies(v0_extension_gpu_integration_tests th021_research_extension th021_extension_artifact_runner)
        add_test(NAME V0ExtensionGpuIntegrationTests COMMAND v0_extension_gpu_integration_tests)
        set_tests_properties(V0ExtensionGpuIntegrationTests PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(v0_graphics_gpu_integration_tests
            tests/graphics/v0/GraphicsGpuIntegrationTests.cpp)
        target_link_libraries(v0_graphics_gpu_integration_tests PRIVATE thiran_v0_graphics)
        add_test(NAME V0GraphicsGpuIntegrationTests COMMAND v0_graphics_gpu_integration_tests)
        set_tests_properties(V0GraphicsGpuIntegrationTests PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(v0_native_gpu_integration_tests tests/backend/v0/NativeGpuIntegrationTests.cpp)
        target_link_libraries(v0_native_gpu_integration_tests PRIVATE thiran_v0_native_gpu)
        add_test(NAME V0NativeGpuIntegrationTests COMMAND v0_native_gpu_integration_tests)
        set_tests_properties(V0NativeGpuIntegrationTests PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(v0_native_artifact_gpu_integration_tests
            tests/artifact/v0/NativeArtifactGpuIntegrationTests.cpp)
        target_link_libraries(v0_native_artifact_gpu_integration_tests PRIVATE thiran_v0_artifact)
        target_compile_definitions(v0_native_artifact_gpu_integration_tests PRIVATE
            TH016_RUNTIME="$<TARGET_FILE:thiran-artifact>")
        add_dependencies(v0_native_artifact_gpu_integration_tests thiran-artifact)
        add_test(NAME V0NativeArtifactGpuIntegrationTests
                 COMMAND v0_native_artifact_gpu_integration_tests)
        set_tests_properties(V0NativeArtifactGpuIntegrationTests PROPERTIES SKIP_RETURN_CODE 77)

        add_executable(v0_model_gpu_integration_tests
            tests/model/v0/ModelGpuIntegrationTests.cpp)
        target_link_libraries(v0_model_gpu_integration_tests PRIVATE
            thiran_v0_reference_model thiran_v0_process)
        target_compile_definitions(v0_model_gpu_integration_tests PRIVATE
            TH017_CXX="${CMAKE_CXX_COMPILER}"
            TH017_INCLUDE="${CMAKE_SOURCE_DIR}/include"
            TH017_BUILD="${CMAKE_BINARY_DIR}"
            TH017_RUNTIME="$<TARGET_FILE:thiran-model>"
            TH022_CLI="$<TARGET_FILE:Thiran>")
        add_dependencies(v0_model_gpu_integration_tests thiran-model Thiran)
        add_test(NAME V0ModelGpuIntegrationTests COMMAND v0_model_gpu_integration_tests)
        set_tests_properties(V0ModelGpuIntegrationTests PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(v0_buffer_tests tests/storage/v0/BufferTests.cpp)
    target_link_libraries(v0_buffer_tests PRIVATE thiran_v0_storage)
    add_test(NAME V0BufferTests COMMAND v0_buffer_tests)

    add_library(thiran_v0_test_native STATIC tests/interop/v0/TestNativeLibrary.cpp)
    add_executable(v0_interop_tests tests/interop/v0/InteropTests.cpp)
    target_link_libraries(v0_interop_tests PRIVATE thiran_v0_tooling)
    target_compile_definitions(v0_interop_tests PRIVATE
        TH009_CXX="${CMAKE_CXX_COMPILER}"
        TH009_INCLUDE="${CMAKE_SOURCE_DIR}/include"
        TH009_BUILD="${CMAKE_BINARY_DIR}"
        TH009_TEST_ARCHIVE="$<TARGET_FILE:thiran_v0_test_native>")
    add_dependencies(v0_interop_tests thiran_v0_test_native)
    add_test(NAME V0InteropTests COMMAND v0_interop_tests)

    add_executable(v0_tooling_tests tests/tooling/v0/ToolingTests.cpp)
    target_link_libraries(v0_tooling_tests PRIVATE thiran_v0_tooling)
    target_compile_definitions(v0_tooling_tests PRIVATE
        TH009_CXX="${CMAKE_CXX_COMPILER}"
        TH009_INCLUDE="${CMAKE_SOURCE_DIR}/include"
        TH009_BUILD="${CMAKE_BINARY_DIR}"
        TH009_CLI="$<TARGET_FILE:thiran-v0>")
    add_dependencies(v0_tooling_tests thiran-v0)
    add_test(NAME V0ToolingTests COMMAND v0_tooling_tests)

    add_executable(v0_workflow_cli_tests tests/tooling/v0/WorkflowCliTests.cpp)
    target_link_libraries(v0_workflow_cli_tests PRIVATE thiran_v0_process)
    target_compile_definitions(v0_workflow_cli_tests PRIVATE
        TH022_CLI="$<TARGET_FILE:Thiran>"
        TH022_EXTENSION="$<TARGET_FILE:th021_research_extension>"
        TH022_BAD_ABI="$<TARGET_FILE:th021_bad_abi_extension>"
        TH022_MALFORMED="$<TARGET_FILE:th021_malformed_extension>"
        TH022_DUPLICATE_OP="$<TARGET_FILE:th021_duplicate_op_extension>"
        TH022_NATIVE_GPU=$<BOOL:${THIRAN_ENABLE_NATIVE_GPU}>)
    add_test(NAME V0WorkflowCliTests COMMAND v0_workflow_cli_tests)

    add_library(th021_research_extension SHARED
        tests/fixtures/extensions/research_ops/ResearchOps.cpp)
    target_include_directories(th021_research_extension PRIVATE "${CMAKE_SOURCE_DIR}/include")
    set_target_properties(th021_research_extension PROPERTIES PREFIX "")
    foreach(kind IN ITEMS no_entry bad_abi malformed init_fail zero_ops duplicate_op)
        add_library(th021_${kind}_extension SHARED
            tests/fixtures/extensions/research_ops/FailureExtension.cpp)
        target_include_directories(th021_${kind}_extension PRIVATE "${CMAKE_SOURCE_DIR}/include")
        set_target_properties(th021_${kind}_extension PROPERTIES PREFIX "")
    endforeach()
    target_compile_definitions(th021_no_entry_extension PRIVATE TH021_NO_ENTRY=1)
    target_compile_definitions(th021_malformed_extension PRIVATE TH021_MALFORMED=1)
    target_compile_definitions(th021_init_fail_extension PRIVATE TH021_INIT_FAIL=1)
    target_compile_definitions(th021_zero_ops_extension PRIVATE TH021_ZERO_OPS=1)
    target_compile_definitions(th021_duplicate_op_extension PRIVATE TH021_DUPLICATE_OP=1)

    add_executable(th021_extension_artifact_runner
        tests/extension/v0/ExtensionArtifactRunner.cpp)
    target_link_libraries(th021_extension_artifact_runner PRIVATE thiran_v0_artifact)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(th021_extension_artifact_runner PRIVATE -Wl,--gc-sections)
    endif()

    add_executable(v0_extension_tests tests/extension/v0/ExtensionTests.cpp)
    target_link_libraries(v0_extension_tests PRIVATE thiran_v0_artifact thiran_v0_autodiff thiran_v0_process)
    target_compile_definitions(v0_extension_tests PRIVATE
        TH021_CXX="${CMAKE_CXX_COMPILER}"
        TH021_INCLUDE="${CMAKE_SOURCE_DIR}/include"
        TH021_BUILD="${CMAKE_BINARY_DIR}"
        TH021_RUNTIME="$<TARGET_FILE:th021_extension_artifact_runner>"
        TH021_EXTENSION="$<TARGET_FILE:th021_research_extension>"
        TH021_NO_ENTRY="$<TARGET_FILE:th021_no_entry_extension>"
        TH021_BAD_ABI="$<TARGET_FILE:th021_bad_abi_extension>"
        TH021_MALFORMED="$<TARGET_FILE:th021_malformed_extension>"
        TH021_INIT_FAIL="$<TARGET_FILE:th021_init_fail_extension>"
        TH021_ZERO_OPS="$<TARGET_FILE:th021_zero_ops_extension>")
    add_dependencies(v0_extension_tests th021_extension_artifact_runner th021_research_extension th021_no_entry_extension
        th021_bad_abi_extension th021_malformed_extension th021_init_fail_extension th021_zero_ops_extension)
    add_test(NAME V0ExtensionTests COMMAND v0_extension_tests)
    add_dependencies(v0_workflow_cli_tests Thiran th021_research_extension
        th021_bad_abi_extension th021_malformed_extension th021_duplicate_op_extension)

    if(THIRAN_ENABLE_NATIVE_GPU)
        add_executable(v0_workflow_gpu_cli_tests tests/tooling/v0/WorkflowGpuCliTests.cpp)
        target_link_libraries(v0_workflow_gpu_cli_tests PRIVATE thiran_v0_process)
        target_compile_definitions(v0_workflow_gpu_cli_tests PRIVATE
            TH022_CLI="$<TARGET_FILE:Thiran>"
            TH022_EXTENSION="$<TARGET_FILE:th021_research_extension>")
        add_dependencies(v0_workflow_gpu_cli_tests Thiran th021_research_extension)
        add_test(NAME V0WorkflowGpuCliTests COMMAND v0_workflow_gpu_cli_tests)
        set_tests_properties(V0WorkflowGpuCliTests PROPERTIES SKIP_RETURN_CODE 77)
    endif()

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
