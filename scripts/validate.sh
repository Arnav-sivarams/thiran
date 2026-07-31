#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
temporary_dir="$(mktemp -d)"

restore_artifacts() {
    for artifact in generated.py graph.dot; do
        if [[ -f "${temporary_dir}/${artifact}.present" ]]; then
            cp "${temporary_dir}/${artifact}" "${repo_root}/${artifact}"
        else
            rm -f "${repo_root}/${artifact}"
        fi
    done
    rm -rf "${temporary_dir}"
}
trap restore_artifacts EXIT

for artifact in generated.py graph.dot; do
    if [[ -e "${repo_root}/${artifact}" ]]; then
        cp "${repo_root}/${artifact}" "${temporary_dir}/${artifact}"
        : > "${temporary_dir}/${artifact}.present"
    fi
done

cd "${repo_root}"

cmake_arguments=(--preset debug)
if [[ -n "${THIRAN_PYTHON_EXECUTABLE:-}" ]]; then
    cmake_arguments+=("-DTHIRAN_PYTHON_EXECUTABLE=${THIRAN_PYTHON_EXECUTABLE}")
    python_executable="${THIRAN_PYTHON_EXECUTABLE}"
else
    python_executable="${repo_root}/.venv/bin/python"
fi

if [[ ! -x "${python_executable}" ]]; then
    echo "Validation requires .venv/bin/python or THIRAN_PYTHON_EXECUTABLE" >&2
    exit 1
fi

cmake "${cmake_arguments[@]}"
cmake --build --preset debug
ctest --preset debug --output-on-failure

for harness in \
    region_ir_tests \
    region_formation_tests \
    strategy_classification_tests \
    region_plan_tests \
    region_plan_cli_support_tests \
    region_python_emitter_tests \
    region_python_runtime_tests \
    installation_support_tests \
    source_manager_tests \
    module_linker_tests \
    function_semantic_tests \
    function_lowering_tests
do
    "${repo_root}/build/debug/${harness}"
done

"${repo_root}/build/debug/Thiran" --help
"${repo_root}/build/debug/Thiran" --version
"${repo_root}/build/debug/Thiran" doctor
"${repo_root}/build/debug/Thiran" --plan \
    "${repo_root}/examples/cnn.th" > "${temporary_dir}/plan.txt"
"${repo_root}/build/debug/Thiran" --emit-region-executor \
    "${repo_root}/examples/mlp.th" "${temporary_dir}/executor.py"
"${repo_root}/build/debug/Thiran" --plan \
    "${repo_root}/tests/fixtures/modules/basic/main.th" > "${temporary_dir}/module-plan.txt"
"${repo_root}/build/debug/Thiran" --emit-region-executor \
    "${repo_root}/tests/fixtures/modules/basic/main.th" "${temporary_dir}/module-executor.py"
"${repo_root}/build/debug/Thiran" --plan \
    "${repo_root}/tests/fixtures/functions/cross_module.th" > "${temporary_dir}/function-plan.txt"
"${repo_root}/build/debug/Thiran" --emit-region-executor \
    "${repo_root}/tests/fixtures/functions/cross_module.th" "${temporary_dir}/function-executor.py"
"${python_executable}" "${temporary_dir}/executor.py" --describe

git diff --check
echo "Thiran validation passed"
