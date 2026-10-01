"""Show the fixed-design read refuses a state sliced to the physical node count.

The census's widened probe passes the whole state vector; this driver reproduces
the previous defect by slicing the state down to the physical node count again.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from benchmarks import unit_amplitude_current_census as census
from nova.jax.config import configure_dtypes


configure_dtypes()
print("declared mutation: pass the sliced state to _fixed_design_read again", flush=True)
control = census._read_control_row()
context = census._build_context(control)
operator = context["operator"]
state = np.asarray(control["terminal_flux_wb"], dtype=np.float64)
print(
    "full_state_rows=%d physical_node_number=%d"
    % (state.size, operator.physical_node_number),
    flush=True,
)

masks, topology, _connected, _admitted = operator._fixed_design_read(jnp.asarray(state))
print("full_state_read_ok=True diverted=%s" % bool(topology.diverted), flush=True)

sliced = jnp.asarray(state)[: operator.physical_node_number]
try:
    operator._fixed_design_read(sliced)
except ValueError as error:
    print("REFUSED_AS_EXPECTED=%s" % error, flush=True)
    raise SystemExit(1)
raise SystemExit("NO_REFUSAL: the sliced read returned without raising")