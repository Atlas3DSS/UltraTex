#!/usr/bin/env bash
# Keep one workboard lease alive while any `ultratex-ovn-*` systemd user unit is active.
# Usage: scripts/mv_lease_heartbeat.sh JOB_ID   (stop by creating $STOP_FILE; the lease is then finished)
set -u
JOB_ID=$1
WB=/home/orwel/.local/bin/gpu-workboard
STOP_FILE=${STOP_FILE:-/tmp/ultratex-lease-stop-${JOB_ID}}
started=""
vram() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -v p="$1" '$1==p {s+=$2} END {print s+0}'; }
while [ ! -f "${STOP_FILE}" ]; do
    unit=$(systemctl --user list-units --type=service --state=active --plain --no-legend 'ultratex-ovn-*' 2>/dev/null | awk '{print $1}' | head -1)
    if [ -n "${unit}" ]; then
        pid=$(systemctl --user show -p MainPID --value "${unit}")
        if [ -z "${started}" ]; then
            "${WB}" start "${JOB_ID}" --unit "${unit}" --pid "${pid}" --observed-mib "$(vram "${pid}")" && started=1
        else
            "${WB}" heartbeat "${JOB_ID}" --pid "${pid}" --observed-mib "$(vram "${pid}")" >/dev/null 2>&1 || true
        fi
    elif [ -n "${started}" ]; then
        "${WB}" heartbeat "${JOB_ID}" --pid "$$" --observed-mib 0 >/dev/null 2>&1 || true
    fi
    sleep 30
done
"${WB}" finish "${JOB_ID}" --outcome completed --detail "overnight lease closed by operator"
