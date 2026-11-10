#!/usr/bin/env bash
# Run field audits and rendering in a separate CPU allocation.
set -uo pipefail
export TMPDIR=/tmp JAX_PLATFORMS=cpu MPLBACKEND=Agg
D=$(mktemp -d "$TMPDIR/fidelity-artifacts-${SLURM_JOB_ID}-XXXX")
trap 'rm -rvf "$D"' EXIT
export TMPDIR="$D" PYTHONPATH="$FIDELITY_ROOT"
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
REV=$(git -C "$FIDELITY_ROOT" rev-parse HEAD)
printf '# revision: %s tree: %s command: bash scripts/diiid_map_fidelity/artifacts.sh\n' "$REV" "$FIDELITY_ROOT"
if [[ "${AUDIT_COMPLETE:-0}" == 1 ]]; then
    "$PY" "$FIDELITY_ROOT/scripts/diiid_map_fidelity/audit.py" --input "$FIDELITY_RUN/measurement"
    RESULT=$?
else
    RESULT=0
fi
if [[ "$RESULT" == 0 ]]; then
    "$PY" "$FIDELITY_ROOT/scripts/diiid_map_fidelity/render.py" \
        --input "$FIDELITY_RUN/measurement" \
        --output "$FIDELITY_ROOT/docs/figures/diiid-reference-map-fidelity" \
        --previews "$FIDELITY_RUN/previews"
    RESULT=$?
fi
rm -rvf "$D"
trap - EXIT
printf 'EXIT=%s\n' "$RESULT"
exit "$RESULT"
