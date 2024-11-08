"""Fiducial fit plots for sector #7."""

import itertools

from nova.assembly.fiducialfit import FiducialFit

sectors = {7: [8, 9]}

phase = "FAT"
color = {"FAT": "C0", "SSAT": "C1"}

fat = FiducialFit(phase="FAT", sectors=sectors, fill=False)
ssat = FiducialFit(phase="SSAT", sectors=sectors, fill=False)


def color_kwargs(color: str) -> dict[str, str]:
    """Return dict of color attributes for plot gpr array."""
    return dict(
        zip(
            [f"{attr}_color" for attr in ["marker", "line", "fill"]],
            itertools.cycle([color]),
        )
    )


stage = 2
coil_index = 1
fat.plot_single(coil_index, stage, color=color["FAT"])
ssat.plot_single(coil_index, stage, color=color["SSAT"], axes=fat.axes)

fat.plot_gpr_array(coil_index, stage, **color_kwargs(color["FAT"]), label="FAT")
ssat.plot_gpr_array(
    coil_index, stage, **color_kwargs(color["SSAT"]), label="SSAT", axes=fat.axes
)

fat.plot_ensemble(True, 500)
ssat.plot_ensemble(True, 500, axes=fat.axes, color=[1, 1])


"""
stage = 3
# fat.plot_single(0, stage, color=color["FAT"])

ssat.plot_single(0, stage + 1, color=color["SSAT"])  # , axes=fat.axes

# fat.plot_gpr_array(1, stage, **color_kwargs(color["FAT"]), label="FAT")
ssat.plot_gpr_array(
    1, stage, **color_kwargs(color["SSAT"]), label="SSAT"
)  # , axes=fat.axes
"""
