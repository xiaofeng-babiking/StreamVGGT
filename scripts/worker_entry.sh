#!/usr/bin/env bash
# Container CMD: activates the shared venv, starts background nvidia-smi dmon,
# sets accelerate launch args from env vars, and execs into accelerate.
#
# All inputs come from env vars set by the orchestrator's `docker run -e ...`:
#   JOB_ID, EXP_NAME, NUM_MACHINES, MACHINE_RANK, NUM_PROCESSES,
#   NPROC_PER_NODE, MAIN_PROCESS_IP, MAIN_PROCESS_PORT, CONFIG_NAME,
#   CUDA_VISIBLE_DEVICES, plus NCCL_*.
#
# Exit codes:
#   non-zero from accelerate launch propagates through
#   11 if /jfs is not mounted or not writable
#   12 if /workspace/.venv is missing
set -euo pipefail

required_vars=(
  JOB_ID EXP_NAME NUM_MACHINES MACHINE_RANK NUM_PROCESSES
  NPROC_PER_NODE MAIN_PROCESS_IP MAIN_PROCESS_PORT CONFIG_NAME
  CUDA_VISIBLE_DEVICES
)
for v in "${required_vars[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "worker_entry: missing required env var ${v}" >&2
    exit 1
  fi
done

# Verify /jfs is mounted and writable as the host UID.
if ! mountpoint -q /jfs; then
  echo "worker_entry: /jfs is not mounted in this container" >&2
  exit 11
fi
if ! touch "/jfs/.svggt_worker_${HOSTNAME}_$$" 2>/dev/null; then
  echo "worker_entry: cannot write to /jfs as $(id)" >&2
  exit 11
fi
rm -f "/jfs/.svggt_worker_${HOSTNAME}_$$"

# Activate shared venv.
if [[ ! -d /workspace/.venv ]]; then
  echo "worker_entry: /workspace/.venv missing (run scripts/bootstrap_venv.sh)" >&2
  exit 12
fi
export VIRTUAL_ENV=/workspace/.venv
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
unset PYTHONHOME  # if anything baked it in upstream, drop it

# Background dmon: write GPU utilization samples to the job dir.
job_dir="/workspace/workflows/jobs/${JOB_ID}"
mkdir -p "${job_dir}"
host_short="$(hostname -s)"
dmon_log="${job_dir}/dmon-${host_short}.csv"
nvidia-smi dmon -s pucvmet -d 2 -o T > "${dmon_log}" 2>&1 &
dmon_pid=$!
trap 'kill ${dmon_pid} 2>/dev/null || true' EXIT

echo "[worker ${MACHINE_RANK}/${NUM_MACHINES}] host=${host_short} cuda=${CUDA_VISIBLE_DEVICES}"
echo "[worker ${MACHINE_RANK}/${NUM_MACHINES}] master=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}"
echo "[worker ${MACHINE_RANK}/${NUM_MACHINES}] dmon -> ${dmon_log}"

cd /workspace/src

exec accelerate launch \
  --multi_gpu \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --num_processes "${NUM_PROCESSES}" \
  --num_processes_per_node "${NPROC_PER_NODE}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  ./train.py --config-name "${CONFIG_NAME}"
