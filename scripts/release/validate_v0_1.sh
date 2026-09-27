#!/usr/bin/env bash
set -euo pipefail

usage='Usage: scripts/release/validate_v0_1.sh <thiran-cli> [build-directory]'
if [[ $# -lt 1 || $# -gt 2 || "$1" == --help || "$1" == -h ]]; then
    echo "${usage}"
    [[ $# -eq 1 ]] && exit 0
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cli="$1"

python3 "${script_dir}/validate_v0_1.py" \
    --source-root "${repo_root}" --cli "${cli}"

if [[ $# -eq 2 ]]; then
    build_dir="$2"
    ctest --test-dir "${build_dir}" --output-on-failure \
        -R 'TH023StructureTests|TH024StructureTests|TH025ReleaseStructure'
fi
