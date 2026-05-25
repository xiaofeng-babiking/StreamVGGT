#!/usr/bin/env bash
# Sourced helper — defines DOCKER_RUN_FLAGS as a bash array for both interactive
# (start_into_docker.sh) and batch (svggt_orch.launch) docker invocations.
#
# Caller must have $(pwd) at the repo root before sourcing.
# Caller may override DOCKER_SHM_SIZE before sourcing (default: 16G).

set -u

: "${DOCKER_SHM_SIZE:=16G}"

DOCKER_RUN_FLAGS=(
  --network host
  --shm-size "${DOCKER_SHM_SIZE}"
  --user "$(id -u):$(id -g)"
  --group-add babiking
  --env HOST_USER_ID="$(id -u)"
  --env HOST_GROUP_ID="$(id -g)"
  --env HOME=/tmp
  --env USER="$(whoami)"
  --env VIRTUAL_ENV=/workspace/.venv
  --env PATH=/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  --env TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions
  --privileged
  -v "$(pwd)":/workspace
  -v /jfs:/jfs
)
