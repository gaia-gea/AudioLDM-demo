#!/usr/bin/env bash
set -euo pipefail

device="${1:-auto}"
device="${device,,}"
if (( $# > 1 )) || [[ ! "$device" =~ ^(auto|gpu|cpu)$ ]]; then
    echo "Usage: bash install.sh [auto|gpu|cpu]" >&2
    exit 1
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "This installer requires x86_64 Linux. Use install.ps1 on Windows." >&2
    exit 1
fi
for tool in curl tar sha256sum flock awk getconf; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required command is missing: $tool" >&2
        exit 1
    fi
done
if ! getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
    echo "A glibc-based Linux distribution is required, such as Ubuntu or Debian." >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime="$script_dir/.cache/runtime/linux-x86_64"
uv_version="0.9.26"
uv="$runtime/uv-$uv_version"
venv="$runtime/venv"
python="$venv/bin/python"
ready="$runtime/ready"
mkdir -p -- "$runtime"

# Serialize setup if two launchers start on the same computer.
exec 9>"$runtime/install.lock"
flock 9

if [[ ! -x "$uv" ]]; then
    archive="$(mktemp "$runtime/uv.XXXXXX.tar.gz")"
    downloaded_uv="$(mktemp "$runtime/uv.XXXXXX")"
    trap 'rm -f -- "$archive" "$downloaded_uv"' EXIT
    echo "Downloading the local Python installer (uv)..."
    curl --fail --location --retry 3 --connect-timeout 15 \
        --output "$archive" \
        "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-x86_64-unknown-linux-gnu.tar.gz"
    expected_hash="30ccbf0a66dc8727a02b0e245c583ee970bdafecf3a443c1686e1b30ec4939e8"
    if ! printf '%s  %s\n' "$expected_hash" "$archive" | sha256sum --check --status; then
        echo "The uv download did not match its expected checksum." >&2
        exit 1
    fi
    tar -xOf "$archive" uv-x86_64-unknown-linux-gnu/uv > "$downloaded_uv"
    chmod +x "$downloaded_uv"
    mv -- "$downloaded_uv" "$uv"
fi

export UV_PYTHON_INSTALL_DIR="$runtime/python"
export UV_CACHE_DIR="$runtime/uv-cache"

backend="cpu"
if [[ "$device" == gpu ]]; then
    backend="cu124"
elif [[ "$device" == auto ]] && command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --id=0 --query-gpu=driver_version,compute_cap --format=csv,noheader,nounits 2>/dev/null |
        awk -F, '
            {
                split($1, driver, ".");
                architecture = $2 + 0;
                driver_ok = driver[1] > 550 ||
                    (driver[1] == 550 && driver[2] > 54) ||
                    (driver[1] == 550 && driver[2] == 54 && driver[3] >= 14);
                if (driver_ok && architecture >= 5 && architecture < 10) compatible = 1;
            }
            END { exit !compatible }
        '; then
        backend="cu124"
    fi
fi

install_torch() {
    local selected_backend="$1"
    echo "Installing PyTorch ($selected_backend)..."
    "$uv" --no-config --quiet pip install --python "$python" \
        --index-url "https://download.pytorch.org/whl/$selected_backend" \
        "torch==2.6.0+$selected_backend"
}

rm -f -- "$ready"
echo "Preparing local Python 3.12..."
"$uv" --no-config venv --python 3.12 --managed-python --allow-existing "$venv"
install_torch "$backend"
echo "Installing the demo dependencies..."
"$uv" --no-config --quiet pip install --python "$python" \
    --index-url https://pypi.org/simple -r "$script_dir/requirements.txt"

if [[ "$backend" == cu124 ]] && ! "$python" -c 'import torch; torch.ones(1).cuda().item()'; then
    if [[ "$device" == gpu ]]; then
        echo "CUDA could not run. Check the NVIDIA driver or retry with: bash install.sh cpu" >&2
        exit 1
    fi
    echo "CUDA could not run. Installing the CPU version." >&2
    install_torch cpu
fi

"$python" "$script_dir/engine.py" --diagnose
printf 'ready\n' > "$ready"
echo "Setup complete. Run: bash start.sh"
