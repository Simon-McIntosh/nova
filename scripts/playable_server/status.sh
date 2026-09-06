#!/usr/bin/env bash
# Report the live playable serving job: node, port and age from the scheduler,
# plus the allocation record the payload wrote (job id, node, port, start time,
# nova revision).

set -euo pipefail

readonly PORT=18506
readonly JOB_NAME=nova-playable
readonly allocation_log="${PLAYABLE_ALLOCATION_LOG:-${HOME}/.local/share/nova/playable/allocation.log}"

# -- print the live scheduler row, if any -------------------------------
live=$(squeue -n "${JOB_NAME}" --format='%i %N %S' --noheader 2>/dev/null || true)
if [[ -n "${live}" ]]; then
  read -r live_job live_node live_start <<<"${live}"
  if [[ -n "${live_job}" ]]; then
    live_epoch=$(date -d "${live_start}" +%s 2>/dev/null || echo -n '')
    age='unknown'
    if [[ -n "${live_epoch}" ]]; then
      now_epoch=$(date +%s)
      age=$((now_epoch - live_epoch))
    fi
    printf 'LIVE job=%s node=%s port=%s started=%s age_seconds=%s\n' \
      "${live_job}" "${live_node}" "${PORT}" "${live_start}" "${age}"
  fi
else
  printf 'LIVE no %s job running\n' "${JOB_NAME}"
fi

# -- the payload's allocation record, whether or not the job is live -----
if [[ -r "${allocation_log}" ]]; then
  read -r rec_job rec_node rec_port rec_start rec_revision <"${allocation_log}"
  printf 'RECORD job=%s node=%s port=%s started=%s revision=%s\n' \
    "${rec_job}" "${rec_node}" "${rec_port}" "${rec_start}" "${rec_revision}"
else
  printf 'RECORD none (no allocation log at %s)\n' "${allocation_log}"
fi
