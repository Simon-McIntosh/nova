"""Reject separately evaluated requests in the executable-body counter."""

import re
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp

from machine import operator_evaluation, shared_operator_call
from nova.jax.config import configure_dtypes

revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("revert the single body to per-request evaluation", flush=True)
print(
    f"revision={revision} tree={Path.cwd()} command=python negative-control.py",
    flush=True,
)
configure_dtypes()
assert jax.config.jax_enable_x64
operator = operator_evaluation(jnp.sin)


def per_request(state):
    return operator(operator(operator(state)))


def shared(state):
    return shared_operator_call(per_request, state)


state = jnp.arange(4, dtype=jnp.float64)
for function in (shared, per_request):
    hlo = jax.jit(function).lower(state).compile().as_text()
    count = len(re.findall(r" = f64\[4\].* sine\(", hlo))
    print(f"mode={function.__name__} optimized_operator_bodies={count}", flush=True)
    assert count == 1, f"single-body guard refused {count} per-request bodies"
