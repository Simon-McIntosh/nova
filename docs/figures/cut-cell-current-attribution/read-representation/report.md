# Boundary-representation discriminator: where the cut-cell shortfall comes from

Measured on the committed Solov'ev -110 rows and the single-null -300 row.  Each row is evaluated on the analytic flux sampled exactly at the nodes (exact input) and on the committed terminal state; each state's boundary level is located under the linear edge root, the ring quadratic and the order-6 global split spline, clipped with ``traced_clip`` and the production geometric candidacy.

- source revision `676d6bebcf8284d7459fb6b7f4feda2e98b92330`; lane `98dci4-clu-3141` (cpu).
- A representation whose exact-input ratio is within 0.1 percent of unity carries no representation error; the terminal shortfall under it is then state error.

## Per-row verdicts

- moderate: the spline boundary carries no representation error (exact-input shortfall -0.0000) while linear falls short by 0.0053 even on the exact input; the terminal shortfall of 0.0095 is 100.0 percent state error.
- single-null: no representation is within the 0.1 percent band on the exact input (linear -0.0099, quadratic -0.0154, spline -0.0140); the terminal shortfall of 0.1990 is the read boundary level, which sits 0.0010 Wb (21.3 percent of the flux span, about 57.1 mm inward) on both the exact and the terminal state, so it is a read error under every representation, not a representation or terminal-field error.
- strong: the spline boundary carries no representation error (exact-input shortfall -0.0000) while linear falls short by 0.0059 even on the exact input; the terminal shortfall of -0.0107 is 99.7 percent state error.
- weak: the spline boundary carries no representation error (exact-input shortfall -0.0000) while linear falls short by 0.0055 even on the exact input; the terminal shortfall of 0.0169 is 100.0 percent state error.

| row | analytic total [A] | cut cells | exact: linear | exact: quadratic | exact: spline | terminal: linear | terminal: quadratic | terminal: spline | terminal best | state error |
|---|---|---|---|---|---|---|---|---|---|---|
| moderate | 1624924.7 | 69 | 0.99467 | 0.99997 | 1.00003 | 0.98137 | 0.98656 | 0.99047 | spline | 0.00956 |
| single-null | 37174.3 | 201 | 1.00985 | 1.01542 | 1.01396 | 0.80098 | 0.80536 | 0.80536 | linear | 0.20888 |
| strong | 251236.0 | 66 | 0.99412 | 0.99999 | 1.00003 | 0.99744 | 1.00279 | 1.01075 | spline | -0.01072 |
| weak | 16314109.3 | 62 | 0.99453 | 0.99997 | 1.00003 | 0.96710 | 0.97311 | 0.98310 | spline | 0.01693 |

## Per-row detail (exact input then terminal)
### moderate (cells 136, pitch 0.1314 m, qualification unqualified)
- **exact** boundary level 0.000000 Wb (analytic separatrix at 0; the terminal read error is its value)
  - linear: ratio 0.99467, participating cut cells 61, boundary root error median 0.403 mm (0.00307 pitch), max 2.037 mm (0.01550 pitch), sign disagreements 0
  - quadratic: ratio 0.99997, participating cut cells 61, boundary root error median 0.061 mm (0.00046 pitch), max 0.151 mm (0.00115 pitch), sign disagreements 7
  - spline: ratio 1.00003, participating cut cells 61, boundary root error median 0.000 mm (0.00000 pitch), max 0.000 mm (0.00000 pitch), sign disagreements 7
- **terminal** boundary level 0.077009 Wb (analytic separatrix at 0; the terminal read error is its value)
  - linear: ratio 0.98137, participating cut cells 60, boundary root error median 7.681 mm (0.05844 pitch), max 13.099 mm (0.09966 pitch), sign disagreements 0
  - quadratic: ratio 0.98656, participating cut cells 60, boundary root error median 6.528 mm (0.04967 pitch), max 11.135 mm (0.08472 pitch), sign disagreements 5
  - spline: ratio 0.99047, participating cut cells 60, boundary root error median 9.339 mm (0.07106 pitch), max 32.713 mm (0.24890 pitch), sign disagreements 16
### single-null (cells 340, pitch 0.0917 m, qualification qualified)
- **exact** boundary level 0.000000 Wb (analytic separatrix at 0; the terminal read error is its value)
  - linear: ratio 1.00985, participating cut cells 142, boundary root error median 0.340 mm (0.00371 pitch), max 5.008 mm (0.05460 pitch), sign disagreements 0
  - quadratic: ratio 1.01542, participating cut cells 142, boundary root error median 0.030 mm (0.00033 pitch), max 3.329 mm (0.03630 pitch), sign disagreements 0
  - spline: ratio 1.01396, participating cut cells 142, boundary root error median 0.002 mm (0.00002 pitch), max 0.011 mm (0.00012 pitch), sign disagreements 0
- **terminal** boundary level 0.001003 Wb (analytic separatrix at 0; the terminal read error is its value)
  - linear: ratio 0.80098, participating cut cells 113, boundary root error median 74.846 mm (0.81611 pitch), max 139.704 mm (1.52332 pitch), sign disagreements 0
  - quadratic: ratio 0.80536, participating cut cells 113, boundary root error median 74.563 mm (0.81303 pitch), max 140.488 mm (1.53187 pitch), sign disagreements 0
  - spline: ratio 0.80536, participating cut cells 113, boundary root error median 74.590 mm (0.81333 pitch), max 140.299 mm (1.52981 pitch), sign disagreements 0
### strong (cells 135, pitch 0.0526 m, qualification unqualified)
- **exact** boundary level 0.000000 Wb (analytic separatrix at 0; the terminal read error is its value)
  - linear: ratio 0.99412, participating cut cells 60, boundary root error median 0.216 mm (0.00411 pitch), max 1.273 mm (0.02419 pitch), sign disagreements 0
  - quadratic: ratio 0.99999, participating cut cells 59, boundary root error median 0.015 mm (0.00029 pitch), max 0.043 mm (0.00082 pitch), sign disagreements 7
  - spline: ratio 1.00003, participating cut cells 59, boundary root error median 0.000 mm (0.00000 pitch), max 0.000 mm (0.00000 pitch), sign disagreements 7
- **terminal** boundary level -0.001906 Wb (analytic separatrix at 0; the terminal read error is its value)
  - linear: ratio 0.99744, participating cut cells 59, boundary root error median 0.572 mm (0.01087 pitch), max 2.563 mm (0.04871 pitch), sign disagreements 0
  - quadratic: ratio 1.00279, participating cut cells 59, boundary root error median 0.549 mm (0.01044 pitch), max 2.996 mm (0.05693 pitch), sign disagreements 5
  - spline: ratio 1.01075, participating cut cells 59, boundary root error median 3.202 mm (0.06083 pitch), max 10.786 mm (0.20495 pitch), sign disagreements 17
### weak (cells 135, pitch 0.4661 m, qualification unqualified)
- **exact** boundary level 0.000000 Wb (analytic separatrix at 0; the terminal read error is its value)
  - linear: ratio 0.99453, participating cut cells 56, boundary root error median 1.822 mm (0.00391 pitch), max 5.734 mm (0.01230 pitch), sign disagreements 0
  - quadratic: ratio 0.99997, participating cut cells 56, boundary root error median 0.154 mm (0.00033 pitch), max 0.488 mm (0.00105 pitch), sign disagreements 10
  - spline: ratio 1.00003, participating cut cells 56, boundary root error median 0.000 mm (0.00000 pitch), max 0.000 mm (0.00000 pitch), sign disagreements 12
- **terminal** boundary level 10.395132 Wb (analytic separatrix at 0; the terminal read error is its value)
  - linear: ratio 0.96710, participating cut cells 54, boundary root error median 37.489 mm (0.08044 pitch), max 80.639 mm (0.17302 pitch), sign disagreements 0
  - quadratic: ratio 0.97311, participating cut cells 54, boundary root error median 32.951 mm (0.07070 pitch), max 78.254 mm (0.16790 pitch), sign disagreements 12
  - spline: ratio 0.98310, participating cut cells 53, boundary root error median 86.279 mm (0.18512 pitch), max 194.791 mm (0.41795 pitch), sign disagreements 24

### Worse cells (booked minus analytic, top 6 per state and representation)
- moderate exact linear: 69:-1087; 6:-769; 70:-730; 130:-628; 7:-607; 135:-492
- moderate exact quadratic: 123:-11; 77:-11; 126:-8; 110:-8; 20:+7; 74:-7
- moderate exact spline: 69:+2; 70:+2; 6:+2; 126:+2; 130:+2; 74:+2
- moderate terminal linear: 69:-2435; 130:-1724; 6:-1689; 70:-1597; 62:-1357; 18:-1283
- moderate terminal quadratic: 6:-1227; 69:-1219; 18:-1090; 27:-1088; 62:-1059; 3:-998
- moderate terminal spline: 69:+3194; 126:-3183; 70:+2780; 51:-2743; 53:-2245; 135:+1976
- single-null exact linear: 2:+180; 13:+111; 5:+87; 11:+68; 15:+31; 1:+12
- single-null exact quadratic: 2:+183; 5:+121; 13:+110; 11:+68; 15:+64; 1:+13
- single-null exact spline: 2:+183; 13:+110; 5:+99; 11:+68; 15:+33; 1:+13
- single-null terminal linear: 25:-202; 245:-191; 23:-188; 27:-186; 19:-173; 308:-169
- single-null terminal quadratic: 25:-202; 245:-189; 23:-188; 27:-185; 19:-173; 308:-169
- single-null terminal spline: 25:-202; 245:-189; 23:-188; 27:-185; 19:-173; 308:-169
- strong exact linear: 69:-144; 7:-117; 8:-114; 129:-107; 70:-78; 125:-77
- strong exact quadratic: 76:-1; 122:-1; 109:-1; 59:+1; 97:+1; 63:+1
- strong exact spline: 69:+0; 125:+0; 7:+0; 70:+0; 129:+0; 8:+0
- strong terminal linear: 69:-170; 7:-135; 8:-131; 70:-105; 54:+95; 2:-79
- strong terminal quadratic: 54:+154; 56:+108; 52:+101; 30:+61; 59:+51; 67:+49
- strong terminal spline: 70:+414; 69:+413; 56:+356; 125:+320; 8:-300; 130:+288
- weak exact linear: 134:-10345; 75:-10345; 3:-6776; 125:-6776; 120:-4732; 76:-4732
- weak exact quadratic: 83:-109; 116:-109; 85:-81; 113:-81; 78:-80; 119:-80
- weak exact spline: 134:+23; 75:+23; 76:+18; 120:+18; 125:+17; 3:+17
- weak terminal linear: 68:-27487; 28:-27153; 134:-26186; 75:-26186; 72:-25717; 13:-25356
- weak terminal quadratic: 68:-25409; 28:-25409; 13:-23120; 72:-23120; 14:-21753; 71:-21753
- weak terminal spline: 119:-38596; 78:-38596; 75:+35661; 134:+35661; 50:+34887; 68:-30284