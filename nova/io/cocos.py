"""Compatibility import path for the COCOS algebra.

The convention table and its factor algebra now live in the standalone
``nova_cocos`` distribution, so an engine that needs only the factors can
install them without nova's own dependency floor.  This module re-exports
every public name for importers that still reach for ``nova.io.cocos``; it is
retired once those importers name ``nova_cocos`` directly.
"""

from nova_cocos import (
    B0_LIKE,
    CONVENTION_DIGITS,
    Convention,
    ConventionTransform,
    ConventionError,
    DODPSI_LIKE,
    IP_LIKE,
    ONE_LIKE,
    PSI_LIKE,
    Q_LIKE,
    TRANSFORMATIONS,
    convention,
    convention_transform,
    conventions,
    identify_convention,
    transform_factor,
)

__all__ = [
    "B0_LIKE",
    "CONVENTION_DIGITS",
    "Convention",
    "ConventionError",
    "ConventionTransform",
    "DODPSI_LIKE",
    "IP_LIKE",
    "ONE_LIKE",
    "PSI_LIKE",
    "Q_LIKE",
    "TRANSFORMATIONS",
    "convention",
    "convention_transform",
    "conventions",
    "identify_convention",
    "transform_factor",
]
