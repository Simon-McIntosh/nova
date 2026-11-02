"""Count the optimized-HLO instructions each tangent call site contributes."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import re
import sys
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp

from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


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

    lowered = jax.jit(
        lambda b, m, s, p: support_read._local_booking_sensitivities(
            b, m, s, profile, gradient, quadrature, p
        )
    ).lower(bank, moments, support, booked)
    executable = lowered.compile()
    text = executable.as_text()
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/dev/null")
    out.write_text(text)
    print(
        json.dumps(
            {
                "hlo_lines": len(text.splitlines()),
                "executable_bytes": len(text.encode()),
            }
        )
    )

    tagged = Counter()
    untagged = Counter()
    for line in text.splitlines():
        match = re.search(r'frontend_attributes=\{tangent_site="([^"]+)"', line)
        opcode = re.search(r"= ([a-z0-9]+(?:-[a-z]+)?)\(", line)
        name = opcode.group(1) if opcode else (line.split("=")[-1].strip().split("(")[0].strip())
        if match:
            tagged[(match.group(1), name)] += 1
        elif "=" in line and ("(" in line):
            untagged[name] += 1
    for (site, name), count in sorted(tagged.items()):
        print("TAG-TAGGED", site, name, count)
    print("UNTAGGED", dict(untagged.most_common(15)))
    # Reductions and concatenations regardless of tag.
    for name in ("reduce", "concatenate", "dot", "select", "sort"):
        print("COUNT", name, text.count(f" {name}("))


if __name__ == "__main__":
    main()