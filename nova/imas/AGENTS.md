# IMAS machine-description guidance

This directory treats a machine description as a version-pinned set of static
IMAS IDSs. It is separate from a pulse or reconstruction dataset: a pulse may
omit static geometry that is present in the machine-description artifact, and
an artifact may declare a quantity absent even though some experimental dataset
contains a measurement related to it.

## Find the description before reading it

Runtime descriptions live in content-addressed machine-artifact caches. Start
from the configured artifact digest and cache root, then call
`nova.imas.machine_artifact.resolve_machine_artifact`; its returned
`VerifiedMachineArtifact.directory` and `manifest.files` are the verified
locations. `manifest.dd_version` is the exact Data Dictionary version with
which the IDSs must be opened. A receipt or a checked-in evidence copy may point
to a netCDF payload elsewhere in the tree, but that path is evidence, not a
portable default. Follow the receipt to its manifest instead of guessing a
facility name, pulse, run, or directory layout.

Before dereferencing an IDS, list its occurrences. Open occurrence zero only
after establishing that it exists, and use `autoconvert=False`. A missing
occurrence is evidence about that entry; it is not, by itself, evidence that the
machine description or the physical machine lacks the quantity.

## Load and inspect the static IDSs

The following recipe accepts any IMAS entry URI or netCDF path plus its manifest
DD pin. It opens the three common description IDSs, routes the wall and active
coils through Nova's pure machine classes, and examines magnetic diagnostics at
their IMAS nodes. `StaticMachineDescription` deliberately models contours and
conductor sections; it does not recast point magnetic probes as sightlines.

Set `NOVA_MACHINE_DESCRIPTION_ENTRY` to an IMAS URI or a netCDF path and
`NOVA_MACHINE_DESCRIPTION_DD_VERSION` to the manifest's exact `dd_version`, then
run:

```bash
UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" \
  uv run --no-sync python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import imas

from nova.imas.machine import (
    CrossSection,
    ImasGeometryReader,
    MachineCoil,
    MachineContour,
    MachineSection,
    StaticMachineDescription,
)


source_text = os.environ["NOVA_MACHINE_DESCRIPTION_ENTRY"]
source = source_text if source_text.startswith("imas:") else Path(source_text)
dd_version = os.environ["NOVA_MACHINE_DESCRIPTION_DD_VERSION"]


def text(value: object) -> str:
    return str(value).strip()


def optional_text(parent: object, name: str) -> str:
    return text(getattr(parent, name)) if hasattr(parent, name) else ""


def machine_section(element: object, fallback: str) -> MachineSection:
    geometry = element.geometry
    geometry_type = int(geometry.geometry_type)
    try:
        section_type = CrossSection.transform[geometry_type]
    except KeyError as error:
        raise ValueError(f"unsupported IMAS geometry type {geometry_type}") from error
    name = optional_text(element, "identifier") or optional_text(element, "name")
    name = name or fallback
    return MachineSection(
        name=name,
        section=ImasGeometryReader(geometry).section(section_type),
    )


with imas.DBEntry(source, "r", dd_version=dd_version) as entry:
    occurrences = {
        name: entry.list_all_occurrences(name)
        for name in ("wall", "pf_active", "magnetics", "pf_passive", "tf")
    }
    for required in ("wall", "pf_active", "magnetics"):
        if 0 not in occurrences[required]:
            raise ValueError(f"{required} occurrence 0 is absent: {occurrences[required]}")

    ids = {
        name: entry.get(name, 0, lazy=False, autoconvert=False)
        for name in ("wall", "pf_active", "magnetics")
    }
    optional = {
        name: (
            entry.get(name, 0, lazy=False, autoconvert=False)
            if 0 in occurrences[name]
            else None
        )
        for name in ("pf_passive", "tf")
    }

wall = ids["wall"]
active = ids["pf_active"]
magnetics = ids["magnetics"]
outline = wall.description_2d[0].limiter.unit[0].outline
contour = MachineContour(
    kind="limiter",
    r=tuple(float(value) for value in outline.r),
    z=tuple(float(value) for value in outline.z),
)

coils = []
for coil_index, coil in enumerate(active.coil):
    coil_name = (
        optional_text(coil, "name")
        or optional_text(coil, "identifier")
        or f"coil_{coil_index}"
    )
    elements = tuple(
        machine_section(element, f"{coil_name}_{element_index}")
        for element_index, element in enumerate(coil.element)
    )
    coils.append(
        MachineCoil(
            name=coil_name,
            identifier=optional_text(coil, "identifier"),
            elements=elements,
        )
    )

machine = StaticMachineDescription(
    contour=contour,
    active_sections=tuple(
        element for coil in coils for element in coil.elements
    ),
    active_coils=tuple(coils),
    passive_loop_count=(
        0 if optional["pf_passive"] is None else len(optional["pf_passive"].loop)
    ),
    toroidal_coil_count=(
        0
        if optional["tf"] is None or not hasattr(optional["tf"], "coil")
        else len(optional["tf"].coil)
    ),
    sightlines=(),
)

written_versions = {
    name: text(value.ids_properties.version_put.data_dictionary)
    for name, value in ids.items()
}
if written_versions != {name: dd_version for name in ids}:
    raise ValueError(f"IDS DD versions disagree with the manifest: {written_versions}")

print(f"DD version: {dd_version}")
print(f"wall limiter vertices: {len(machine.contour.r)}")
print(f"active coils: {len(machine.active_coils)}")
print(f"active elements: {len(machine.active_sections)}")
print(f"poloidal probes: {len(magnetics.b_field_pol_probe)}")
print(f"flux loops: {len(magnetics.flux_loop)}")
PY
```

The accessor paths above are the same paths used by the native IDS exporter and
its exact round-trip snapshot:

- wall limiter ring:
  `wall.description_2d[0].limiter.unit[0].outline.r` and `.z`;
- active conductors:
  `pf_active.coil[i].element[j].turns_with_sign`,
  `.geometry.geometry_type`, `.geometry.outline.r`, and `.geometry.outline.z`;
- magnetic probes:
  `magnetics.b_field_pol_probe[i].position.{r,z,phi}` plus
  `.poloidal_angle` and `.toroidal_angle`;
- flux loops:
  `magnetics.flux_loop[i].position[j].{r,z,phi}`.

Keep static and dynamic content distinct while inspecting. A machine-description
`magnetics` IDS describes sensor identity and geometry; filled `field.data`,
`flux.data`, or plasma-current arrays are pulse data and must not silently become
part of the static description.

## Absence is scoped to its source

Write absence claims with both the source and the quantity. These statements are
different:

- “This dataset has no `wall` occurrence or wall column” means only that the
  selected dataset cannot supply the wall.
- “The verified machine-description manifest declares wall absent” means the
  governed description cannot supply it for the artifact's stated shot range.
- “The machine has no physical wall” is a physical claim and needs independent
  evidence; neither of the preceding observations establishes it.

DIII-D is the worked example. The competition dataset does not ship a wall
contour, but Nova's separate machine-description IDS carries a deterministic
physical limiter ring at
`wall.description_2d[0].limiter.unit[0].outline`. Therefore the correct statement
is “wall absent from the competition dataset,” not “DIII-D has no wall” or “Nova
has no DIII-D wall.” The same distinction transfers to any tokamak: always name
the dataset or artifact whose occurrence list was examined.

## Check whether a receipt predates a mechanism

A receipt proves the tree it measured, not later trees. Compare the author time
of the commit that last banked the receipt with the author time of the commit that
introduced the named mechanism. Then confirm ancestry; dates alone do not prove
that two commits belong to the same line of development.

Set the receipt path, the source path that owns the mechanism, and a stable code
needle naming that mechanism:

```bash
NOVA_RECEIPT_PATH='docs/figures/efit-forward-parity/reference-seeded-forward-slice.json'
NOVA_MECHANISM_PATH='nova/equilibrium/forward_operator.py'
NOVA_MECHANISM_NEEDLE='def current_normalisation_amplitude'

NOVA_RECEIPT_COMMIT=$(git log -1 --format='%H' -- "$NOVA_RECEIPT_PATH")
NOVA_MECHANISM_COMMIT=$(git log --reverse --format='%H' \
  -S "$NOVA_MECHANISM_NEEDLE" -- "$NOVA_MECHANISM_PATH" | head -1)

test -n "$NOVA_RECEIPT_COMMIT" && test -n "$NOVA_MECHANISM_COMMIT"
git show -s --format='receipt   %H %aI %s' "$NOVA_RECEIPT_COMMIT"
git show -s --format='mechanism %H %aI %s' "$NOVA_MECHANISM_COMMIT"

NOVA_RECEIPT_AUTHORED=$(git show -s --format='%at' "$NOVA_RECEIPT_COMMIT")
NOVA_MECHANISM_AUTHORED=$(git show -s --format='%at' "$NOVA_MECHANISM_COMMIT")
if (( NOVA_RECEIPT_AUTHORED < NOVA_MECHANISM_AUTHORED )); then
  echo 'receipt author date predates the named mechanism'
else
  echo 'receipt author date does not predate the named mechanism'
fi

if git merge-base --is-ancestor "$NOVA_RECEIPT_COMMIT" "$NOVA_MECHANISM_COMMIT"; then
  echo 'the receipt tree is ancestral to the mechanism commit: re-run before citing'
else
  echo 'ancestry is not established: inspect the compared production diff'
fi

git show --stat --oneline "$NOVA_RECEIPT_COMMIT" -- "$NOVA_RECEIPT_PATH"
git show --stat --oneline "$NOVA_MECHANISM_COMMIT" -- "$NOVA_MECHANISM_PATH"
```

Here the named mechanism is declared-current normalisation. The commands resolve
the banked receipt to `57a1bfb9` (authored 2026-08-21) and the mechanism to
`cf812416` (authored 2026-08-22), with the former ancestral to the latter. The
receipt therefore predates a mechanism that changes which roots are reachable
and must be re-run on a tree containing that mechanism before its vacuum-branch
conclusion is cited as current evidence. For another receipt, replace all three
inputs; do not reuse these commits merely because the subject matter looks
similar.
