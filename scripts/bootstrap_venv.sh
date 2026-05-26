#!/usr/bin/env bash
# Populate the shared uv venv at the repo root's .venv/.
#
# Idempotent and safe to re-run. Exits non-zero with a clear message on any
# precondition failure.
#
# Exit codes:
#   0  success
#   1  uv missing or configuration error
#   2  /jfs out of space
#   3  another bootstrap is in progress (flock contention)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

venv_path="${repo_root}/.venv"
lock_path="${repo_root}/.venv.lock"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found on PATH. Install:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# Refuse to proceed if /jfs is below a 5 GB safety margin (JuiceFS is shared).
avail_mb="$(df -BM --output=avail /jfs 2>/dev/null | tail -n1 | tr -dc '0-9' || echo 0)"
if [[ -n "${avail_mb}" && "${avail_mb}" -lt 5120 ]]; then
  echo "error: /jfs has only ${avail_mb} MB free; need at least 5120 MB to sync the venv." >&2
  echo "       Run: df -h /jfs" >&2
  exit 2
fi

# Single-writer: another bootstrap might be running on a peer node.
exec 9>"${lock_path}"
if ! flock -n 9; then
  echo "error: another bootstrap is running (held lock on ${lock_path})." >&2
  exit 3
fi

# Tell uv to install its managed Python interpreters under the repo (which
# lives on /jfs and is bind-mounted into worker containers at the same path).
# Otherwise the venv's bin/python symlinks at ~/.local/share/uv/python/... —
# a per-host local path that doesn't exist inside the container OR on other
# nodes. Worker containers would crash with exit 127 "required file not found".
export UV_PYTHON_INSTALL_DIR="${repo_root}/.uv-python"
mkdir -p "${UV_PYTHON_INSTALL_DIR}"

# If a stale venv exists with bin/python symlinks pointing outside
# UV_PYTHON_INSTALL_DIR (e.g. left over from a pre-fix bootstrap), it WILL
# break in docker. Rebuild from scratch to get the correct symlink target.
if [[ -L "${venv_path}/bin/python" ]]; then
  current_python="$(readlink -f "${venv_path}/bin/python" 2>/dev/null || true)"
  if [[ -n "${current_python}" && "${current_python}" != ${UV_PYTHON_INSTALL_DIR}/* ]]; then
    echo "[bootstrap] existing .venv references ${current_python}"
    echo "[bootstrap]   that path is not under ${UV_PYTHON_INSTALL_DIR}, so the venv"
    echo "[bootstrap]   won't work inside docker containers. Removing and rebuilding."
    rm -rf "${venv_path}"
  fi
fi

echo "[bootstrap] Repo root: ${repo_root}"
echo "[bootstrap] Venv path: ${venv_path}"
echo "[bootstrap] UV_PYTHON_INSTALL_DIR: ${UV_PYTHON_INSTALL_DIR}"
echo "[bootstrap] uv version: $(uv --version)"

uv sync --extra orchestrator --group dev

# Verify the venv's interpreter lives under UV_PYTHON_INSTALL_DIR (i.e. on /jfs).
resolved_python="$(readlink -f "${venv_path}/bin/python")"
if [[ "${resolved_python}" != ${UV_PYTHON_INSTALL_DIR}/* ]]; then
  echo "error: venv's python resolves to ${resolved_python}, not under" >&2
  echo "       ${UV_PYTHON_INSTALL_DIR}. The venv will not work in docker containers." >&2
  exit 1
fi
echo "[bootstrap] venv python -> ${resolved_python}"

# Sanity check.
"${venv_path}/bin/python" -c "import torch, accelerate, asyncssh, click, yaml, rich; print('venv OK')"

echo "[bootstrap] Done. Activate with:  source ${venv_path}/bin/activate"
