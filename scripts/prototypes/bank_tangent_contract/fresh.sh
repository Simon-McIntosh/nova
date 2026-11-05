#!/usr/bin/env bash
set -u
export TMPDIR=/tmp PYTHONPATH="$PWD" JAX_PLATFORMS=cpu
RUN=/home/ITER/mcintos/.config/reckon/crew/runs/r-20261103T095720496569-bank-tangent-contract
PY=/home/ITER/mcintos/Code/nova/.venv/bin/python
LOG="$RUN/fresh-sirius.log"
(
  printf '# revision: %s tree: %s cwd: %s command: %s\n' "$(git rev-parse HEAD)" "$PWD" "$PWD" "$PY -m scripts.prototypes.bank_step_booking.measure --arm fresh --max-steps 15"
  "$PY" -P -c 'import nova, nova.equilibrium.current as m; print("NOVA="+nova.__file__+" MODULE="+m.__file__)'
  "$PY" -m scripts.prototypes.bank_step_booking.measure --arm fresh --max-steps 15
) > "$LOG" 2>&1
code=$?
printf 'EXIT=%s\n' "$code" >> "$LOG"
exit "$code"
