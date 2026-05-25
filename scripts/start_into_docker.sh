#!/usr/bin/env bash
# Interactive docker session for development.
# Shares DOCKER_RUN_FLAGS with the orchestrator via scripts/lib/docker_env.sh
# so flags don't drift between interactive and batch launches.
#
# Behaviour preserved from the original (untracked) script:
#  - Runs as the host UID via --user "$(id -u):$(id -g)" so writes to
#    /workspace (bind-mounted) work natively, no chown gymnastics.
#  - --group-add babiking keeps /home/babiking readable (where colmap
#    lives inside the image, owned by babiking with 750 perms).
#  - HOME=/tmp because the host uid has no entry in /etc/passwd inside
#    the container -> no real home dir.
#  - --privileged for /dev access (NVIDIA + perf counters during dev).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Override the helper's 16G default. The interactive session typically does
# not need as much shared memory as a full DDP training launch.
: "${DOCKER_SHM_SIZE:=8G}"
export DOCKER_SHM_SIZE

# shellcheck source=lib/docker_env.sh
source "${repo_root}/scripts/lib/docker_env.sh"

exec docker run -it --rm \
  --name ironman \
  "${DOCKER_RUN_FLAGS[@]}" \
  babiking/ubuntu:12.8.1-cudnn-devel-ubuntu24.04 /bin/bash

# ---------------------------------------------------------------------------
# Resuming the uv env inside the new container:
#   source /workspace/.venv/bin/activate
#   # — or invoke directly: /workspace/.venv/bin/python train.py ...
# The .venv lives on the host bind-mount, so it survives `docker rm` and is
# immediately ready in any container that mounts the project at /workspace.
# JIT-built CUDA extensions (diff_gaussian_rasterization) are cached in
# /workspace/.torch_extensions/ via the env var above — also bind-mounted, so
# the first-run ~60 s compile happens only once per host, not per container.
#
# Optional X11 (forwards DISPLAY at container start; goes stale on re-SSH).
# Add these to DOCKER_RUN_FLAGS in lib/docker_env.sh if you need GUI tools:
#   --env DISPLAY=$DISPLAY
#   -v $HOME/.Xauthority:/home/babiking/.Xauthority
#   -v /tmp/.X11-unix:/tmp/.X11-unix
#
# Launch COLMAP GUI from the host (replace :10 with your current $DISPLAY):
#   # 1. Inject host's X cookie into babiking's .Xauthority inside the container.
#   xauth -f ~/.Xauthority nlist :10 | docker exec -i -u 0 ironman \
#     bash -c 'xauth -f /home/babiking/.Xauthority nmerge - && chown babiking:babiking /home/babiking/.Xauthority'
#   # 2. Spawn the GUI detached; window opens on the host's X server.
#   docker exec -d ironman bash -c \
#     "DISPLAY=:10 XAUTHORITY=/home/babiking/.Xauthority /home/babiking/install/colmap/bin/colmap gui"
#   # Kill later: docker exec ironman pkill -f 'colmap gui'
#
# Commit a snapshot of the container (e.g., after installing 3rd-party deps):
#   docker commit -a "babiking" -m "install 3rd-party dependencies" ironman \
#     babiking/ubuntu:12.8.1-cudnn-devel-ubuntu24.04-gsplat1.5.2
