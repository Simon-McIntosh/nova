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

As of January 2026, four sector modules (5, 6, 7, 8) have delivered coils to the ITER pit.

### Full Torus Coil Positions

Coil numbering follows **manufacturing order**, NOT angular position. The `location` list in
`FiducialPit` defines the mapping from position index (0-17) to coil number.

```
                           0° (Port 1)
                              │
                            (14)
                      15            [4]
                     (no)           S8
               18                         17
              (no)                       (no)

         1                                      6
       (no)                                   (no)

                      ITER PIT
                   (top view)

       [11]                                     7
        S8                                    (no)
       300°                                  100°

           10                               2
          (no)                            (no)

             [9]                       3
              S7                     (no)
             260°                   140°

                [8]               [16]
                 S7                S5
                240°              160°

                   [13]       [5]
                    S6         S5
                   220°       180°
                       [12]
                        S6
                       200°
                          │
                        180°
```

Legend: `[coil]` = installed, `(coil)` = not installed, `S#` = sector module number

### Sector Module to Pit Position Mapping

| Sector Module | Coils | Pit Positions | Angular Range | Adjacent? |
|---------------|-------|---------------|---------------|-----------|
| 5             | 16, 5 | 8, 9          | 160°-180°     | Yes (to 6) |
| 6             | 12, 13| 10, 11        | 200°-220°     | Yes (to 5,7) |
| 7             | 8, 9  | 12, 13        | 240°-260°     | Yes (to 6) |
| 8             | 4, 11 | 2, 15         | 40°, 300°     | **NO - SPLIT!** |

### Critical Finding: Sector 8 Coils are NOT Adjacent

**Sector Module 8** contains coils 4 and 11, which were measured together as a sector module unit.
However, in the pit they are installed at **opposite positions**:

- **Coil 4**: Position 2 (40°) - isolated, no adjacent installed coils
- **Coil 11**: Position 15 (300°) - isolated, no adjacent installed coils

This means:
1. The intra-sector gap for Sector 8 (4↔11) was measured during sector assembly, not in-pit
2. Neither coil from Sector 8 is adjacent to the main 5-6-7 cluster
3. **No inter-sector gaps exist involving Sector 8 coils** in the current installation

### Contiguous Installed Cluster (160°-260°)

The only contiguous cluster of installed coils spans 100° from position 8 to 13:

```
Position:   8     9     10    11    12    13
Angle:    160°  180°  200°  220°  240°  260°
Coil:      16 ── 5 ── 12 ── 13 ── 8 ── 9
            └─S5─┘     └─S6──┘    └─S7─┘
                 ↑           ↑
              Gap 5-6     Gap 6-7
            (inter)      (inter)
```

### Verified Inter-Sector Gap Interfaces

Only two inter-sector gaps exist between adjacent installed coils:

| Gap | From Sector | To Sector | Coil Pair | ILIS Surfaces | Verified |
|-----|-------------|-----------|-----------|---------------|----------|
| 5→6 | 5 (second)  | 6 (first) | 5 → 12    | 5 ILIS +1 ↔ 12 ILIS -1 | ✓ |
| 6→7 | 6 (second)  | 7 (first) | 13 → 8    | 13 ILIS +1 ↔ 8 ILIS -1 | ✓ |

**Previous inter-sector gap calculations for 7→8 and 8→5 were INVALID** because those coil
pairs are not physically adjacent in the pit.

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

Only gaps between physically adjacent coils in the pit are valid:

| Gap | Coil Pair | Mean Gap (mm) | Std (mm) |
|-----|-----------|---------------|----------|
| 5→6 | 5 → 12    | 2.336         | 0.126    |
| 6→7 | 13 → 8    | 2.330         | 0.208    |
| **Mean** |       | **2.333**     | **0.167**|

Inter-sector gaps (~2.33 mm) are larger than intra-sector gaps (~1.39 mm).

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
