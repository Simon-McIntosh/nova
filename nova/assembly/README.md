# Nova Assembly Module

This module provides tools for ITER TF coil assembly analysis, including fiducial metrology processing, gap calculations, and Monte Carlo assembly simulations.

## Coordinate System

### Sector Module Coordinates

The TF coil assembly uses a right-handed coordinate system centered on the sector module:

- **x**: Radial direction, positive outward from machine axis
- **y**: Tangential (toroidal) direction, positive clockwise when viewed from above
- **z**: Vertical direction, positive upward

### Sector Geometry

Each of the 9 sector modules spans 40° toroidally and contains two TF coils positioned at ±10° from the sector midplane:

```
Sector Midplane (y = 0 in clocked coordinates)
       │
       │
    ┌──┼──┐
    │  │  │
+10°│  │  │-10°
    │  │  │
Coil 1 │ Coil 2
(first)│ (second)
    │  │  │
    └──┼──┘
       │
```

- **First coil**: Located at +10° (anticlock direction from sector midplane)
- **Second coil**: Located at -10° (clock direction from sector midplane)

### Clocking Transformations

To analyze gaps, both coils' data are transformed ("clocked") to align with the sector midplane:

- **`anticlock()`**: Rotates by +10° (π/18 rad) about z-axis, used for first coil
- **`clock()`**: Rotates by -10° (-π/18 rad) about z-axis, used for second coil

After clocking, both coils' ILIS surfaces appear at y ≈ 0 (sector midplane).

## ILIS Surfaces

ILIS (Intercoil Structure Interface Surface) are the mating surfaces between adjacent TF coils. Each coil has two ILIS surfaces:

- **ILIS +1**: Surface on the anticlock (positive y) side of the coil
- **ILIS -1**: Surface on the clock (negative y) side of the coil

### Intra-Sector Gap Formation

The intra-sector gap (gap between two coils within a sector) is formed by:

- **First coil's ILIS +1** (faces toward sector midplane after clocking)
- **Second coil's ILIS -1** (faces toward sector midplane after clocking)

```
sector_index = [(first_coil, 'ILIS +1'), (second_coil, 'ILIS -1')]
```

### Inter-Sector Gap Formation

The inter-sector gap (gap between two adjacent sectors) is formed by:

- **First sector's second coil ILIS +1** (faces clockwise)
- **Next sector's first coil ILIS -1** (faces anticlockwise)

## Gap Calculation Methodology

### Step 1: Clock Data to Sector Midplane

Apply clocking transformations to both plane parameters and point cloud data:

```python
sector_planes = ilis.planes.groupby(['coil'], group_keys=False).apply(
    lambda x: clock(x, x.name, cords=[('x', 'y', 'z'), ('nx', 'ny', 'nz')])
)
sector_data = ilis.data.groupby(['coil'], group_keys=False).apply(
    lambda x: clock(x, x.name)
)
```

### Step 2: Find Midplane

Calculate the bisecting plane between the two gap-forming ILIS surfaces:

```python
midplane = ilis.intersect(sector_planes.loc[sector_index])
```

The midplane has:
- **Point**: Average of the two ILIS plane centers
- **Normal**: Average of the two ILIS plane normals (pointing in ±y direction)

### Step 3: Calculate Offsets

Calculate signed perpendicular distance from each ILIS point to the midplane:

```python
offset = ilis.offset(sector_data.loc[plane_index, ('x', 'y', 'z')], midplane)
```

The `offset()` method computes:
```
offset = (point - midplane_point) · midplane_normal
```

This gives:
- **ILIS +1**: Negative offset (below midplane in y)
- **ILIS -1**: Positive offset (above midplane in y)

### Step 4: Compute Gap

The gap is the difference between the two offsets:

```python
gap = offset[ILIS -1] - offset[ILIS +1]
```

Since ILIS +1 has negative offset and ILIS -1 has positive offset:
```
gap = (+0.92) - (-0.92) ≈ 1.84 mm
```

## Installed Sector Layout (In-Pit Phase)

As of January 2026, four sectors (5, 6, 7, 8) are installed in the ITER pit.

### Sector Coil Mapping

| Sector | First Coil | Second Coil | Angular Position |
|--------|------------|-------------|------------------|
| 8      | 4          | 11          | 80°-120°         |
| 7      | 8          | 9           | 120°-160°        |
| 6      | 12         | 13          | 160°-200°        |
| 5      | 16         | 5           | 200°-240°        |

### Coil Adjacency (Verified from ILIS Gap Data)

The coil numbering is **not based on angular position** - it follows a manufacturing/delivery convention. However, the adjacency has been verified from ILIS gap calculations:

```
Anticlockwise direction →

    Sector 8      Sector 7      Sector 6      Sector 5
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  4  ──  11  │  8  ──  9   │ 12  ──  13  │ 16  ──  5   │
└─────────────┴─────────────┴─────────────┴─────────────┘
     intra         inter         intra         inter         intra         inter         intra
              ↑            ↑             ↑             ↑
          gap 8-7      gap 7-6       gap 6-5       gap 5-?
         (11↔8)       (9↔12)       (13↔16)      (5↔next)
```

**Coil sequence around the machine (anticlockwise):**
```
... ── 4 ── 11 ── 8 ── 9 ── 12 ── 13 ── 16 ── 5 ── ...
       └──S8──┘   └──S7──┘   └───S6───┘   └──S5──┘
```

### Inter-Sector Gap Interfaces

| Gap | From Sector | To Sector | Coil Pair | ILIS Surfaces |
|-----|-------------|-----------|-----------|---------------|
| 8→7 | 8 (second)  | 7 (first) | 11 → 8    | 11 ILIS +1 ↔ 8 ILIS -1 |
| 7→6 | 7 (second)  | 6 (first) | 9 → 12    | 9 ILIS +1 ↔ 12 ILIS -1 |
| 6→5 | 6 (second)  | 5 (first) | 13 → 16   | 13 ILIS +1 ↔ 16 ILIS -1 |

### Notes on Sector 8 Coil Numbering

Sector 8 contains coils [4, 11], which is anomalous compared to other sectors:
- Sectors 6, 7 have sequential coil numbers (12-13, 8-9)
- Sector 5 wraps around (16-5)
- Sector 8 has non-sequential numbers (4-11)

The adjacency of coil 11 to coil 8 (sector 7) has been verified via inter-sector gap calculation (~2.5 mm gap). The coil numbering appears to follow manufacturing order rather than angular position.

## Measured Intra-Sector Gap Statistics (In-pit Phase)

| Sector | Coils  | Mean Gap (mm) | Std (mm) | Range (mm)   |
|--------|--------|---------------|----------|--------------|
| 5      | 16-5   | 0.933         | 0.158    | 0.59 - 1.33  |
| 6      | 12-13  | 1.911         | 0.229    | 1.34 - 2.35  |
| 7      | 8-9    | 0.830         | 0.147    | 0.33 - 1.06  |
| 8      | 4-11   | 1.883         | 0.116    | 1.67 - 2.16  |
| **Overall** |    | **1.389**     | **0.588**| 0.33 - 2.35  |

The pooled within-sector standard deviation is 0.168 mm.

**Variance interpretation:**
- **Within-sector spatial variation**: ~0.17 mm (pooled std) - how gap varies across the ILIS surface
- **Between-sector systematic variation**: ~0.59 mm - differences in mean gap between sector modules

## Measured Inter-Sector Gap Statistics (In-pit Phase)

| Gap | Coil Pair | Mean Gap (mm) | Std (mm) |
|-----|-----------|---------------|----------|
| 8→7 | 11 → 8    | 2.460         | 0.228    |
| 7→6 | 9 → 12    | 3.595         | 0.120    |
| 6→5 | 13 → 16   | 2.251         | 0.136    |
| **Overall** |   | **2.769**     | **0.723**|

Inter-sector gaps are generally larger than intra-sector gaps (mean 2.77 mm vs 1.39 mm).

## Key Classes

### FiducialSector

Manages sector fiducial data loading and coordinate transformations.

```python
from nova.assembly.fiducialsector import FiducialSector

fiducial = FiducialSector(
    phase='In-pit target',  # 'In-pit target', 'In-pit actual', 'Factory'
    sectors={6: [12, 13]},   # {sector_number: [first_coil, second_coil]}
    private=False            # Use published (not private) data
)
```

### FiducialIlis

Processes ILIS point cloud data and provides gap calculation utilities.

```python
from nova.assembly.fiducialilis import FiducialIlis

ilis = FiducialIlis(fiducial.ilis, pcr=True)

# Get midplane between two surfaces
midplane = ilis.intersect(planes.loc[sector_index])

# Calculate offset from points to a reference plane
offset = ilis.offset(points, midplane)
```

### Rotate

Provides clocking transformations.

```python
from nova.assembly.transform import Rotate

rotate = Rotate()
clocked_coords = rotate.clock(coords)      # -10° rotation
anticlocked_coords = rotate.anticlock(coords)  # +10° rotation
```

## References

- Sector positions defined in `nova/assembly/sectorposition.py`
- ILIS nominal definitions in `nova/assembly/ilisnominal.py`
- Monte Carlo trial simulation in `nova/assembly/trial.py`
