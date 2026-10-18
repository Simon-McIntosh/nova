"""Flux orientation for external fields read by the topology adapter."""

from __future__ import annotations

from nova.equilibrium.topology import TopologyConvention


def read_convention(*, polarity=None, dd_version=None, plasma_current=None):
    """Use a trace's signed axis direction or the IDS's written convention."""
    if polarity is not None:
        if dd_version is not None or plasma_current is not None:
            raise ValueError("trace polarity and IDS orientation are exclusive")
        if polarity not in (-1, 1):
            raise ValueError("trace polarity must be -1 or 1")
        return TopologyConvention.from_cocos(17, polarity)
    if dd_version is None or plasma_current is None:
        raise ValueError("IDS orientation needs DD version and plasma current")
    major = int(str(dd_version).split(".", 1)[0])
    if major not in (3, 4):
        raise ValueError(f"unsupported equilibrium DD version: {dd_version}")
    return TopologyConvention.from_cocos({3: 11, 4: 17}[major], plasma_current)
