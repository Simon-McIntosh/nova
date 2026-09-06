# Transport forward-model guidance

This guidance applies to the coupled forward transport under `nova/transport/`:
advancing a current-diffusion state over a flux-surface geometry through the
native or TORAX rung, and coupling the evolved state back into an equilibrium
forward source. Repository-wide development and git rules remain in the root
`AGENTS.md`.

## Run a coupled forward through the declared-defaults seam

The public deterministic forward is `ForwardTransport` in
`nova/transport/forward.py`; the only supported way to advance an interval is
`ForwardTransport.solve` on a typed `ForwardTransportInput`, returning a
`ForwardTransportReceipt`. Engine selection is explicit in the input's
`TransportModel.rung` — `TransportRung.NATIVE_PSI_DIFFUSION` or
`TransportRung.TORAX_MULTI_CHANNEL` — and is repeated in the receipt's
`TransportProvenance`; a failing engine raises `TransportEngineError`, never
manufactures a replacement state. The batched member form is
`EnsembleForwardTransport` over `EnsembleTransportInput` and
`EnsembleTransportReceipt` in `nova/transport/ensemble.py`.

- The coupled forward spans both directions of the equilibrium–transport
  boundary. Current diffusion (`nova/transport/current_diffusion.py`) evolves
  the state on flux-surface geometry built from `torax_geometry_from_fsa`
  (`nova/transport/torax_geometry.py`), and
  `forward_source_from_receipt` (`nova/transport/evolved_state.py`) maps the
  evolved receipt back into an equilibrium `ForwardSource`. The equilibrium
  side of a coupled solve goes through the equilibrium request seam —
  `ForwardSolveRequest` / `ForwardProfile.solve` — never through a private
  fixed-point kernel.
- The declared equilibrium defaults table lives once at
  `nova/equilibrium/solve_request.py` (`FORWARD_SOLVE_DEFAULTS`), keyed by the
  installed Nova package version. A coupled driver does not hand-configure an
  equilibrium default and does not call the bare fixed-point routine; it
  builds its request with `ForwardSolveRequest.from_defaults(...)` and records
  any deliberate deviation on the receipt's `resolved_defaults.deviations`.
- Every coupled forward records which equilibrium defaults and which transport
  route ran: the equilibrium resolved-defaults block rides the equilibrium
  solve, and the transport rung and engine ride the `TransportProvenance` on
  the transport receipt. TORAX is the optional `transport` extra
  (`pyproject.toml` declares `transport = ["torax>=1.4.3"]`), so the native
  rung is the only one that must always work.
- After touching a coupled route or its equilibrium coupling, run the
  default-wiring tests named in `nova/equilibrium/AGENTS.md`:
  `tests/test_route_default_wiring.py` and `tests/test_default_wiring.py`.
