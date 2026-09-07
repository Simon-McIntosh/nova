"""Animate one pulse's EFIT reconstruction beside its flux-function profiles.

Three panels: the poloidal reconstruction on the left as unfilled contours
over the coils and first wall with its O-point and X-points marked, and the
two flux-function profiles on the right, ``p'`` above and ``FF'`` below.

Two choices keep the animation readable, and both are about what stays still.
The contour levels are computed once from the whole pulse rather than per
frame, so a contour line means the same flux throughout and the map is not
silently renormalised as the plasma grows. The profile panels are likewise
pinned once, so a profile that doubles looks like a profile that doubles.

Those profile limits cover a central quantile of the FRAMES rather than the
pulse's extreme values. An early slice whose plasma has barely formed carries
flux-function gradients orders of magnitude above the flat top -- on MAST
21978 one 20 ms frame reaches -7.8e5 Pa/Wb where every other frame stays
above -4.3e3 -- and bounding by extremes spends the animation's whole vertical
range on that one frame. Excluded frames are drawn outside the axes rather
than rescaling it, and the quantile is recorded in the receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova.media import gif, poloidal as pol, traces as tr
from nova.media.ink import DEFAULT_INK
from nova.media.layout import three_view
from nova.media.sources.frame import Pulse

DEFAULT_OUTPUT = Path("docs/figures/presentation-media")


def pulse_levels(pulse: Pulse, count: int) -> np.ndarray:
    """Return one absolute level array spanning every frame of the pulse.

    Taken over the whole pulse so the levels are a property of the discharge
    rather than of a frame: a per-frame array would redraw the same physical
    surface at a different line in every frame.
    """
    maps = np.concatenate([frame.flux.reshape(-1) for frame in pulse.frames])
    boundary = float(np.median([frame.flux_boundary for frame in pulse.frames]))
    return pol.contour_levels(maps, count, boundary=boundary)


def render_pulse(
    pulse: Pulse,
    path: Path,
    duration: float = 10.0,
    levels: int = 24,
    height: float = 6.5,
    quantile: float | None = 0.98,
) -> dict[str, object]:
    """Write the three-panel animation and return its receipt."""
    style = DEFAULT_INK
    level_array = pulse_levels(pulse, levels)
    view = three_view(extent=pulse.extent(), height=height, style=style)
    scale_p = tr.TraceScale.over(
        [frame.psi_norm for frame in pulse.frames],
        [frame.p_prime for frame in pulse.frames],
        quantile=quantile,
    )
    scale_f = tr.TraceScale.over(
        [frame.psi_norm for frame in pulse.frames],
        [frame.ff_prime for frame in pulse.frames],
        quantile=quantile,
    )

    def render(index: int) -> None:
        frame = pulse.frames[index]
        view.clear()
        pol.draw_flux_contours(
            view.poloidal, frame.radius, frame.height, frame.flux, level_array
        )
        pol.draw_coils(view.poloidal, pulse.geometry.coils)
        pol.draw_wall(view.poloidal, *pulse.geometry.limiter.T)
        pol.draw_boundary(view.poloidal, *frame.boundary.T)
        pol.draw_nulls(
            view.poloidal,
            magnetic_axis=frame.magnetic_axis,
            x_points=frame.x_points,
        )
        tr.draw_trace(view.upper, frame.psi_norm, frame.p_prime)
        scale_p.apply(view.upper)
        tr.label_axes(view.upper, None, "$p'$  [Pa Wb$^{-1}$]")
        tr.draw_trace(view.lower, frame.psi_norm, frame.ff_prime)
        scale_f.apply(view.lower)
        tr.label_axes(
            view.lower, "normalised poloidal flux", "$FF'$  [T$^2$m$^2$Wb$^{-1}$]"
        )
        tr.annotate_time(
            view.upper,
            f"{pulse.machine} {pulse.identifier}\n"
            f"t = {1e3 * frame.time:.0f} ms\n"
            f"$I_p$ = {1e-3 * frame.plasma_current:.0f} kA",
        )

    receipt = gif.animate(
        view.figure, range(len(pulse)), render, path, duration=duration
    )
    receipt |= {
        "machine": pulse.machine,
        "pulse": pulse.identifier,
        "contour_levels": len(level_array),
        "level_span_wb": [float(level_array[0]), float(level_array[-1])],
        "levels_shared_across_frames": True,
        "profile_limits_fixed": True,
        "profile_limit_quantile": quantile,
        "p_prime_limit": list(scale_p.y_limit),
        "ff_prime_limit": list(scale_f.y_limit),
        "time_span_s": [float(pulse.times()[0]), float(pulse.times()[-1])],
        "provenance": pulse.provenance,
    }
    return receipt


def _read_pulse(arguments: argparse.Namespace) -> Pulse:
    """Return the requested machine's pulse through its own adapter."""
    if arguments.machine == "mast":
        from nova.media.sources.mast_efit import read_pulse

        return read_pulse(int(arguments.pulse), require_plasma=arguments.current_floor)
    from nova.media.sources.diiid_efit import read_pulse

    return read_pulse(arguments.pulse, stride=arguments.stride)


def main(argv: list[str] | None = None) -> int:
    """Render one pulse animation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", choices=("mast", "diiid"), default="mast")
    parser.add_argument("--pulse", default="21978", help="shot or pulse identifier")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--levels", type=int, default=24)
    parser.add_argument("--height", type=float, default=6.5)
    parser.add_argument("--stride", type=int, default=4, help="DIII-D slice stride")
    parser.add_argument("--current-floor", type=float, default=1.0e4)
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.98,
        help="central fraction of FRAMES the fixed profile limits must cover",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)

    pulse = _read_pulse(arguments)
    name = f"{pulse.machine.lower()}_{pulse.identifier}_efit_pulse"
    receipt = render_pulse(
        pulse,
        arguments.output / f"{name}.gif",
        duration=arguments.duration,
        levels=arguments.levels,
        height=arguments.height,
        quantile=arguments.quantile,
    )
    receipt_path = arguments.output / f"{name}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
