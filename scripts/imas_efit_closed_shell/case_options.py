"""Solver options and weight files for one imas-efit closed-shell run.

Only solver settings live here: the grid, the convergence controls, the
current setup, the constraint weights and the profile constraints.  Every
machine and pulse fact (coil, shell, wall, probes) is in the IMAS pulse that
make_pulse.py writes.
"""

from __future__ import annotations

from pathlib import Path

PLASMA_IP = 100.0
PLASMA_R0, PLASMA_A = 0.09, 0.04


def efit_options(
    relative_error: float, iterations: int, nr: int = 257, nz: int = 257
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Top>
<debugOptions writeNonConvergedSlices="yes" computeEfitSlices="yes"
              timeSliceInputDataDump="no" writeMdsPlusDebugInformation="no"
              writeIdamDebugInformation="no" efitDataDump="no"
              writeDebugInformation="no" exitOnIdamError="no"
              tokamakDataDump="no" computeForwardFunctions="yes"
              suppressWarnings="true"/>

<responseFunctions compute="auto" fileName="responseFunctions.nc"/>

<timeVector tMin="0.0" nSteps="1" deltaT="1.0"/>

<outputOptions imasWrite="yes" hdf5Write="no" hdf5FileName="efitOut.hdf5"
               netCDF4Write="no" netCDF4FileName="efitOut.nc"
               profileCount="33" fourierHarmonicCount="10" fourierSurface="0.95"
               surfaceCoordCount="361" writeIterationDetails="true"
               errorMessCountMax="20"/>

<regularGrid rMin="0.001" rMax="4.501" zMin="-3.5" zMax="3.5" nr="{nr}" nz="{nz}"/>

<ironModel responseTolerance="1e-6" rootFindingAlgorithm="5"
           absoluteError="1e3" relativeError="{relative_error}"
           maximumIterationCount="{iterations}" relaxFactor="0.5"/>

<currentSetup icinit="2" cutip="1.0e-2" relip="2.96" zelip="0.0"
              aelip="{PLASMA_A}" eelip="1.2" enp="1.0" emp="1.0"
              gammap="1.e-10" gammaf="1.e-10" betap0="0.5"
              rzero="{PLASMA_R0}"/>

<numericalControls mxiter="80" firstWallGridSize="33" scalea="0" error="1.0e-4"
                   saimin="500." conditionFactor="0.0" relaxfactor="0.0"
                   suppressRevCurr="-1" xResol="1e-2" fieldLineResol="1e-4"
                   regimeDetector="off"
                   plasmaRegionRCoord="{PLASMA_R0}" plasmaRegionZCoord="0.0">
<pp ndeg="2" edge="1" func="0"/>
<ff ndeg="2" edge="1" func="0"/>
<ww ndeg="0" edge="1" func="0"/>
</numericalControls>

<codeControls fitDzAlgorithm="1" computeConstraintsWithDz="1"
              computeChi2WithWeights="1" fcurrtFit="0" itell="1"
              alpgamSwitch="1" alpagamaCoef="0.005" lcfsTol="1.e-12"
              profileCount="33"/>

<fitdz option="1" ndelzon="2" scaledz="1"/>

<!-- The data-selection objects for the time-traces read from the pulse.  The
     circuits themselves are built from pf_active after this file is read, so
     this element carries only the selection parameters.  Bvac*R is the zero
     vacuum toroidal field: present so its selection is specified, weighted
     zero so it constrains nothing. -->
<efitCircuits interpolationMethod="1" timeWindow="0" timeMargin="0"
              deselectWhenNoData="false"/>
<magneticProbes interpolationMethod="1" timeWindow="0" timeMargin="0"/>
<bVacRadiusProduct interpolationMethod="1" timeWindow="1" timeMargin="0" weights="0"/>

<plasmaCurrent interpolationMethod="1" timeWindow="0" timeMargin="0" weights="1"/>

<relationalPPrimes interpolationMethod="1" timeWindow="0" timeMargin="0">
  <relationalPPrime weights="1" x="0" c="[[1e-2 1e-2]]"/>
</relationalPPrimes>

<relationalFFPrimes interpolationMethod="1" timeWindow="0" timeMargin="0">
  <relationalFFPrime weights="1" x="0" c="[[5e-4 5e-4]]"/>
</relationalFFPrimes>
</Top>
"""


def magnetic_probes_xml(names: list[str], weight: float = 1.0) -> str:
    """Per-probe weights for the four shell probes.

    The probes are constrained, not deselected.  Their sigma is 10 T against a
    signal of order 1e-4 T, so the pull is negligible, but a zero weight would
    remove their rows from the fit's Jacobian and leave the reconstruction
    rank-deficient: imas-efit then never drives cerror below its tolerance.
    """
    entries = "\n".join(
        f'  <magneticProbe idsIdentifier="{name}" weights="{weight}"/>'
        for name in names
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<magneticProbes>\n" + entries + "\n</magneticProbes>\n"
    )


def write_options(
    outdir: Path,
    relative_error: float,
    iterations: int,
    probe_names: list[str],
    nr: int = 257,
    nz: int = 257,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "efitOptions.xml").write_text(
        efit_options(relative_error, iterations, nr=nr, nz=nz)
    )
    (outdir / "magneticProbes.xml").write_text(magnetic_probes_xml(probe_names))
