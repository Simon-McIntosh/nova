"""Own the JAX global configuration the package depends on.

JAX reads ``jax_enable_x64`` when an array is first traced, not when it is
computed, so the flag has to be set before any tracing happens anywhere in the
process. That makes it a process-wide precondition rather than a module-local
one, and setting it independently from every module that happens to need it
made a single global requirement read as a scattered pile of local ones -- with
no way to tell, at any one site, whether it had already been satisfied.

fp64 is mandatory rather than a preference. The quantities the package traces
through JAX are small differences of much larger grid values -- the saddle flux
against the grid fluxes around it, sub-grid boundary crossings against the cell
they fall in, flux-surface metrics against the volume differences they ratio --
and fp32 loses the difference before the result is formed.
"""

_enabled = False


def enable_x64() -> bool:
    """Enable JAX double precision once per process.

    Idempotent: repeat calls after the first are free, so a module may call this
    at import and a caller may call it again before tracing without cost.

    Returns
    -------
    bool
        True once fp64 is enabled, False when JAX is not installed. Returning
        rather than raising lets a module whose JAX support is optional call
        this unconditionally from inside its import guard.
    """
    global _enabled
    if _enabled:
        return True
    try:
        import jax
    except ModuleNotFoundError:
        return False
    jax.config.update("jax_enable_x64", True)
    _enabled = True
    return True
