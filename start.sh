#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime="$script_dir/.cache/runtime/linux-x86_64"
python="$runtime/venv/bin/python"

if [[ ! -f "$runtime/ready" ]] || ! "$python" -c 'import sys' >/dev/null 2>&1; then
    bash "$script_dir/install.sh"
fi

exec "$python" "$script_dir/demo.py" "$@"
