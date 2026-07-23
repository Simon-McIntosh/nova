"""Characterization harness for the assembly fitting and metrology code.

This package pins the observable behaviour of the coil-fitting, winding-pack,
gap-asymmetry and vault spectral-proxy code against recorded inputs. It
observes; it does not refactor. Canonical outputs live under ``goldens/`` and
are described by ``goldens/manifest.json``.

Acceptance is tolerance-based (see :mod:`_tolerance`): positional and gap
outputs in millimetres agree to 1 micron (three decimal places); other
quantities carry their own tolerance class at least two orders below their
physical noise floor. A sha256 fingerprint of each canonical output is kept in
the manifest as a cheap change detector -- a fingerprint mismatch triggers the
tolerance comparison rather than failing the gate directly, so the harness
survives BLAS and dependency bumps without re-baselining ceremony.
"""
