#!/usr/bin/env bash
set -euo pipefail

usage='Usage: ./scripts/install.sh [--prefix <path>]'
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
prefix="${HOME}/.local"

if [[ $# -eq 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    echo "${usage}"
    exit 0
elif [[ $# -eq 2 && "$1" == --prefix ]]; then
    prefix="$2"
elif [[ $# -ne 0 ]]; then
    echo "${usage}" >&2
    exit 2
fi

if [[ -z "${prefix}" || "${prefix}" == *$'\n'* ]]; then
    echo "Invalid installation prefix" >&2
    exit 2
fi
normalized="$(realpath -m -- "${prefix}")"
if [[ "${normalized}" == / ]]; then
    echo "Invalid installation prefix" >&2
    exit 2
fi

cd "${repo_root}"
cmake --preset install
cmake --build --preset install
cmake --install build/install --prefix "${normalized}"
echo "Installed thiran to ${normalized}/bin/thiran"
echo "Add ${normalized}/bin to PATH if needed."
