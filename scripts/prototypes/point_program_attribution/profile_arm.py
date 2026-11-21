"""Profile one production program arm's cold compile with a source block stubbed.

The cold-process measurement is the pass profile harness's own ``main``: this
module calls it with the harness's ``build`` replaced by a hook that applies one
source block's stub before the program is traced.  Only the hook and the device
refusal live here, so an unstubbed arm measures exactly the program the H200
profile compiles, and a stub that changes nothing is visible by its unchanged
instruction count.
"""

import argparse
import sys
from pathlib import Path

import jax

from scripts.prototypes.compile_pass_profile import profile
from scripts.prototypes.point_program_attribution import blocks

assert jax.config.jax_enable_x64 is True
assert not jax.config.jax_enable_compilation_cache

DEVICE = jax.devices()[0]
DEVICE_KIND = getattr(DEVICE, "device_kind", str(DEVICE))


def stub_hook(block):
    """Return a build stand-in that stubs one block before the program is traced."""
    owner_build = profile.build

    def build(name, size):
        function, operands = owner_build(name, size)
        if block == "elliptic":
            replaced = blocks.apply_elliptic()
        else:
            replaced = blocks.apply(block)
        print("block=%s replaced=%s" % (block, replaced))
        return function, operands

    return build


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--block", default="control")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-device", default="H200")
    args = parser.parse_args()
    if args.expect_device not in DEVICE_KIND:
        raise SystemExit(
            "refusing to measure %r: device %r does not name %r, so this run "
            "would record another backend's wall against the card's.  Set the "
            "platform before the interpreter starts (JAX_PLATFORMS=cuda,cpu)."
            % (" ".join(sys.argv), DEVICE_KIND, args.expect_device)
        )
    profile.build = stub_hook(args.block)
    sys.argv = [
        "profile.py",
        "--program",
        args.program,
        "--size",
        str(args.size),
        "--output",
        str(args.output),
    ]
    profile.main()


if __name__ == "__main__":
    main()
