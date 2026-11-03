"""Census the fused operations the local booking tangent compiles to."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


def census(text, pattern):
    return sorted(re.findall(pattern, text))


def main():
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    support = jax.jit(
        lambda b, c, m: support_read.read(b, c, m, seeds, roots, convention, policy)
    )(bank, cache, moments)
    residual = jax.jit(
        lambda b, m, s: support_read.step_residual(b, m, m, s, profile, quadrature)
    )(bank, moments, support)
    booked = residual.booked

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    tangent = support_read._local_booking_sensitivities(
        bank, moments, support, profile, gradient, quadrature, booked
    )
    lowered = jax.jit(
        lambda b, m, s, p: support_read._local_booking_sensitivities(
            b, m, s, profile, gradient, quadrature, p
        )
    ).lower(bank, moments, support, booked)
    text = lowered.compiler_ir("hlo").as_hlo_text(print_large_constants=False)
    Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tangent.hlo").write_text(text)
    lines = text.splitlines()
    counts = {}
    for name in re.findall(r"\bHloInstruction\b.*?name=\"([^\"]+)\", opcode=\"([^\"]+)\"", text):
        pass
    opcodes = re.findall(r'opcode="([^"]+)"', text)
    from collections import Counter

    print(json.dumps({"lines": len(lines), "opcodes": Counter(opcodes).most_common(20)}))
    for op in ("reduce", "concatenate", "dot", "sort", "select", "scatter", "dynamic-slice", "convert"):
        print(op, opcodes.count(op))
    # Region boundaries mark cond/while bodies
    for token in ('"fusion"', "region", "while", "conditional"):
        print(token, text.count(token))


if __name__ == "__main__":
    main()
