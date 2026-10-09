#!/usr/bin/env bash
# Run one command on a leased fleet GPU as a detached systemd user unit, with
# workboard start/heartbeat/finish bookkeeping (fleet-compute runbook).
#
# Usage: scripts/mv_leased_run.sh JOB_ID UNIT_NAME LOG_FILE -- command args...
# The job must already be acquired (gpu-workboard acquire JOB_ID). The GPU UUID
# is read from the lease file and exported as CUDA_VISIBLE_DEVICES.
# KEEP_LEASE=1 leaves the lease open after the unit exits so several short
# units can share one lease (finish it yourself with `gpu-workboard finish`);
# the first unit registers `start`, later ones only heartbeat.
set -Eeuo pipefail
JOB_ID=$1; UNIT=$2; LOG=$3; shift 3
[ "$1" = "--" ] && shift
WB=/home/orwel/.local/bin/gpu-workboard
LEASE_DIR=${HOME}/.local/state/atlas-gpu-workboard-client/leases
LEASE_FILE=$(ls "${LEASE_DIR}"/*"${JOB_ID}"* 2>/dev/null | head -1 || true)
[ -n "${LEASE_FILE}" ] || { echo "no lease file for ${JOB_ID} in ${LEASE_DIR}" >&2; exit 1; }
GPU_UUID=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('gpu_uuid') or d.get('gpu') or d.get('assigned_gpu_uuid') or '')" "${LEASE_FILE}")
[[ "${GPU_UUID}" == GPU-* ]] || { echo "could not read a GPU UUID from ${LEASE_FILE}" >&2; exit 1; }
mkdir -p "$(dirname "${LOG}")"
echo "job=${JOB_ID} unit=${UNIT} gpu=${GPU_UUID} log=${LOG}"
systemd-run --user --unit "${UNIT}" --collect --same-dir \
    --setenv=CUDA_VISIBLE_DEVICES="${GPU_UUID}" --setenv=PATH="${PATH}" --setenv=HOME="${HOME}" \
    --setenv=COMFY_ROOT="${COMFY_ROOT:-}" --setenv=PYTHONPATH="${PYTHONPATH:-}" \
    bash -c "exec \"\$@\" >> '${LOG}' 2>&1" _ "$@"
sleep 3
MAIN_PID=$(systemctl --user show -p MainPID --value "${UNIT}")
vram() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -v p="$1" '$1==p {s+=$2} END {print s+0}'; }
STARTED_MARKER="${LEASE_DIR}/.started-${JOB_ID}"
if [ -f "${STARTED_MARKER}" ]; then
    "${WB}" heartbeat "${JOB_ID}" --pid "${MAIN_PID}" --observed-mib "$(vram "${MAIN_PID}")" || true
else
    "${WB}" start "${JOB_ID}" --unit "${UNIT}.service" --pid "${MAIN_PID}" --observed-mib "$(vram "${MAIN_PID}")"
    touch "${STARTED_MARKER}"
fi
while systemctl --user is-active --quiet "${UNIT}"; do
    "${WB}" heartbeat "${JOB_ID}" --pid "${MAIN_PID}" --observed-mib "$(vram "${MAIN_PID}")" >/dev/null || true
    sleep 30
done
RESULT=$(systemctl --user show -p Result --value "${UNIT}" 2>/dev/null || echo unknown)
if grep -q "MV_RUN_OK" "${LOG}"; then RESULT=success; fi
if [ -n "${KEEP_LEASE:-}" ]; then
    echo "unit ${UNIT} finished: ${RESULT} (lease kept open)"
    [ "${RESULT}" = "success" ]
    exit $?
fi
if [ "${RESULT}" = "success" ]; then
    "${WB}" finish "${JOB_ID}" --outcome completed --detail "unit ${UNIT} exited (${RESULT})"
else
    "${WB}" finish "${JOB_ID}" --outcome failed --detail "unit ${UNIT} result=${RESULT}; see ${LOG}"
fi
echo "unit ${UNIT} finished: ${RESULT}"
