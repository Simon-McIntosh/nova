# Own-vertex-ring stationary-point census

The production defect reproduced before this independent evaluation: the neighbour-centroid read admitted no saddle at 132 and 340 realised cells, while its 500-requested positive control admitted one.

The raw vertex-minus-centroid rule is retained as a negative control. The centroid-value contour generally cuts only one hyperbola branch pair when the centroid is displaced from the saddle, so it produces two cyclic changes. The operative criterion instead solves the own-node quadratic gradient, classifies its Hessian, and requires the stationary point to lie in or within one quarter pitch of its cell.

## Analytic single-null ladder

| requested | realised | centroid_sign_census_admitted | vertex_census_admitted | vertex_census_position_error_m | vertex_census_position_error_in_pitch | vertex_census_level_error_wb | vertex_census_level_error_in_span | production_admitted | production_position_error_m | production_position_error_in_pitch | production_level_error_wb | production_level_error_in_span | centroid_sign_false_saddle_before→vertex_census_false_saddle_after | centroid_sign_false_extremum_before→vertex_census_false_extremum_after | noise X |
|---:|---:|:---:|:---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 110 | 132 | no | yes | 0.013140019912662523 | 0.08675922024027956 | 7.866768734324812e-07 | 0.00013778201083524652 | no | — | — | — | — | 0→0 | 0→0 | yes |
| 200 | 233 | no | yes | 0.005519659311168668 | 0.0491417416787003 | 3.17735137215188e-06 | 0.0005564951455545611 | yes | 0.010067034155547064 | 0.08962719690715878 | 1.078488869692518e-05 | 0.0018889123367933985 | 0→0 | 0→0 | yes |
| 300 | 340 | no | yes | 0.008463733902874445 | 0.09228814194144236 | 5.055136649323912e-06 | 0.000885378630176013 | no | — | — | — | — | 0→0 | 0→0 | yes |
| 342 | 382 | no | yes | 0.003681312394885456 | 0.04285870063862407 | 5.768029181054914e-06 | 0.0010102377303332997 | yes | 0.0051293797294255095 | 0.05971743951714121 | 2.7424282327220576e-07 | 4.8032081434796e-05 | 0→0 | 0→0 | yes |
| 400 | 449 | no | yes | 0.0024767275056051465 | 0.03118397668656098 | 4.846551727027462e-08 | 8.48846159228289e-06 | yes | 0.0015863576208060243 | 0.01997350897577911 | 1.9860398420727945e-06 | 0.00034784366018761705 | 0→0 | 0→0 | yes |
| 500 | 550 | no | yes | 0.00651659553889861 | 0.0917337279500199 | 6.031451745701822e-06 | 0.001056374704588135 | yes | 0.0011992892842294948 | 0.01688232394909023 | 2.2659545388106416e-07 | 3.968691382223275e-05 | 0→0 | 0→0 | yes |
| 750 | 814 | no | yes | 0.002686748620054884 | 0.0463213214415035 | 2.576155884164071e-06 | 0.00045119915168784035 | yes | 0.0006808918419851411 | 0.011739025245631966 | 8.359372073656233e-07 | 0.00014640968007650657 | 0→0 | 0→0 | yes |
| 1000 | 1074 | no | yes | 0.0013580924762046876 | 0.027036633715538137 | 1.3422789219857373e-07 | 2.3509257131967158e-05 | yes | 0.0015162433043237832 | 0.03018506880856973 | 9.324762606697268e-07 | 0.00016331794996161592 | 0→0 | 0→0 | yes |
| 2500 | 2616 | no | yes | 0.0005012584747705756 | 0.01577811099102676 | 3.094393369863295e-08 | 5.419655200422163e-06 | yes | 0.00040351512399440986 | 0.01270144393240582 | 1.5273856343616027e-07 | 2.6751296641653768e-05 | 0→0 | 0→0 | yes |

## O-point positive control

| case | requested | realised | vertex_census_axis_admitted | vertex_census_axis_position_error_m | vertex_census_axis_position_error_in_pitch | production_axis_admitted | production_axis_position_error_m | production_axis_position_error_in_pitch | extremal_centroid_and_mode_agree |
|:---|---:|---:|:---:|---:|---:|:---:|---:|---:|:---:|
| diverted-single-null | 110 | 132 | yes | 0.00064097128897971 | 0.00423212214272957 | yes | 0.00509702989866791 | 0.03365400832640096 | yes |
| diverted-single-null | 200 | 233 | yes | 0.0011316745531291161 | 0.01007534259255847 | yes | 0.0038092232655017448 | 0.0339136629920277 | yes |
| diverted-single-null | 300 | 340 | yes | 0.0006875993381935895 | 0.007497549669006078 | yes | 0.0025967423081181503 | 0.028314751558477912 | yes |
| diverted-single-null | 342 | 382 | yes | 0.0008160619565193623 | 0.009500784325075314 | yes | 0.002536257715747575 | 0.029527706024798813 | yes |
| diverted-single-null | 400 | 449 | yes | 0.0005879972464016032 | 0.007403354782491315 | yes | 0.0020556571007676184 | 0.025882364111847793 | yes |
| diverted-single-null | 500 | 550 | yes | 0.0006990326229944777 | 0.009840240672171926 | yes | 0.0014329635061060713 | 0.02017174207709966 | yes |
| diverted-single-null | 750 | 814 | yes | 5.735576251335774e-05 | 0.000988851242752107 | yes | 0.0007754456824421752 | 0.013369195930245341 | yes |
| diverted-single-null | 1000 | 1074 | yes | 0.0002477484250074109 | 0.004932126153327788 | yes | 0.0008004055693931741 | 0.015934314181634886 | yes |
| diverted-single-null | 2500 | 2616 | yes | 8.610751706595881e-05 | 0.0027104059677998706 | yes | 0.00031742967180824536 | 0.009991732500738346 | yes |
| weak-rotation-reactor-static | 300 | 342 | yes | 0.0003546427733860561 | 0.0012566385851719587 | yes | 0.003219247087854704 | 0.011407056365975684 | yes |
| weak-rotation-reactor-static | 1000 | 1072 | yes | 0.0005548949118039901 | 0.003589792874510207 | yes | 0.0016663758774022597 | 0.010780319162608173 | yes |
| moderate-rotation-conventional-static | 300 | 341 | yes | 0.00015665635802795356 | 0.0019683960913502185 | yes | 0.0012010143221983238 | 0.015090813594996066 | yes |
| moderate-rotation-conventional-static | 1000 | 1070 | yes | 0.0001326370825533209 | 0.003042767728846065 | yes | 0.00045793718441711206 | 0.010505331237385021 | yes |
| strong-rotation-compact-static | 300 | 341 | yes | 5.5756829363729216e-05 | 0.0017495949170170174 | yes | 0.00028022534602745175 | 0.008793198010428846 | yes |
| strong-rotation-compact-static | 1000 | 1069 | yes | 4.7764795270557557e-05 | 0.002736445053294464 | yes | 4.798908471803642e-05 | 0.002749294595422367 | yes |

## Static axis controls

| case | requested | realised | centroid_sign_census_axis_admitted | vertex_census_axis_admitted | vertex_census_position_error_m | vertex_census_position_error_in_pitch | vertex_census_level_error_in_span | centroid_sign_false_saddle_before→vertex_census_false_saddle_after | centroid_sign_false_extremum_before→vertex_census_false_extremum_after |
|:---|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|
| weak-rotation-reactor-static | 300 | 342 | no | yes | 0.0003546427733860561 | 0.0012566385851719587 | 5.56293948661489e-05 | 0→0 | 0→0 |
| weak-rotation-reactor-static | 1000 | 1072 | yes | yes | 0.0005548949118039901 | 0.003589792874510207 | 6.282776303429697e-07 | 0→0 | 0→0 |
| moderate-rotation-conventional-static | 300 | 341 | no | yes | 0.00015665635802795356 | 0.0019683960913502185 | 5.461803079512695e-05 | 0→0 | 0→0 |
| moderate-rotation-conventional-static | 1000 | 1070 | yes | yes | 0.0001326370825533209 | 0.003042767728846065 | 5.489442511986969e-06 | 0→0 | 0→0 |
| strong-rotation-compact-static | 300 | 341 | no | yes | 5.5756829363729216e-05 | 0.0017495949170170174 | 3.855108474473893e-05 | 0→0 | 0→0 |
| strong-rotation-compact-static | 1000 | 1069 | no | yes | 4.7764795270557557e-05 | 0.002736445053294464 | 4.765312224361612e-06 | 0→0 | 0→0 |

## Periodic six-vertex mode decomposition

The six samples are represented by one periodic trigonometric polynomial: the mean; the cosine and sine m=1 gradient pair; the cosine and sine m=2 traceless-Hessian pair; the single Nyquist m=3 cosine; and the isotropic Hessian from ring mean minus centroid.

| case | requested | null | m1 amplitude (Wb) | m2 amplitude (Wb) | m2 phase (rad) | isotropic (Wb) | m3 amplitude (Wb) | m1:m2 distance / pitch | closed-form distance / pitch | stationary-level changes | source cell | analytic cell | four-sample cells within 2 pitch |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diverted-single-null | 110 | axis | 0.00023700729 | 5.7439921e-05 | -1.5538107 | -0.00014874708 | 4.9807938e-07 | 1.2799468 | 0.36379118 | 0 | 89 | — | 0 |
| diverted-single-null | 110 | saddle | 0.00072766944 | 0.00038412393 | 1.4407181 | -0.0001088346 | 3.4931345e-06 | 0.58763389 | 0.59576775 | 2 | 2 | 9 | 0 |
| diverted-single-null | 200 | axis | 6.9984018e-05 | 3.0223138e-05 | -1.5142211 | -7.7185704e-05 | 1.5632638e-07 | 0.71829589 | 0.2882105 | 0 | 118 | — | 0 |
| diverted-single-null | 200 | saddle | 0.00019053865 | 0.00017630789 | 1.5210028 | -4.8610127e-05 | 3.1467659e-06 | 0.33523966 | 0.45562492 | 2 | 64 | 64 | 0 |
| diverted-single-null | 300 | axis | 3.2177067e-05 | 1.9449111e-05 | -1.5129571 | -4.9852257e-05 | 8.8653908e-08 | 0.51320487 | 0.32803333 | 0 | 69 | — | 0 |
| diverted-single-null | 300 | saddle | 0.00026068884 | 8.99267e-05 | 1.5684409 | -2.7640667e-05 | 2.0858555e-06 | 0.89924459 | 0.68823415 | 4 | 15 | 5 | 0 |
| diverted-single-null | 342 | axis | 1.4499451e-05 | 1.6602304e-05 | -1.5506414 | -4.3324013e-05 | 9.30062e-08 | 0.27091139 | 0.075601384 | 0 | 204 | — | 0 |
| diverted-single-null | 342 | saddle | 0.00021237712 | 9.5537307e-05 | 1.561836 | -2.5648429e-05 | 1.7255464e-06 | 0.68957069 | 0.69700383 | 2 | 5 | 15 | 0 |
| diverted-single-null | 400 | axis | 2.7284561e-05 | 1.3992858e-05 | -1.5602238 | -3.6667298e-05 | 7.7517606e-08 | 0.60485965 | 0.18964421 | 0 | 203 | — | 0 |
| diverted-single-null | 400 | saddle | 0.00011774837 | 6.7124312e-05 | 1.5087989 | -2.3114555e-05 | 1.0312474e-06 | 0.54415063 | 0.5373376 | 2 | 99 | 99 | 0 |
| diverted-single-null | 500 | axis | 4.0742764e-05 | 1.118259e-05 | -1.5248157 | -2.8953083e-05 | 4.5820923e-08 | 1.1301918 | 0.42721271 | 0 | 347 | — | 0 |
| diverted-single-null | 500 | saddle | 0.0001280376 | 7.2691784e-05 | 1.5076879 | -2.0121847e-05 | 7.3260511e-07 | 0.54638184 | 0.75144126 | 2 | 5 | 8 | 0 |
| diverted-single-null | 750 | axis | 3.3145591e-05 | 7.9789799e-06 | -1.5448483 | -2.0687524e-05 | 2.7184984e-08 | 1.2886128 | 0.35871522 | 0 | 603 | — | 0 |
| diverted-single-null | 750 | saddle | 0.00012862968 | 4.6730496e-05 | 1.4486873 | -1.4825354e-05 | 2.4402636e-07 | 0.85385645 | 0.72767781 | 2 | 39 | 64 | 0 |
| diverted-single-null | 1000 | axis | 1.1344069e-05 | 5.7928201e-06 | -1.5624018 | -1.5151663e-05 | 1.9937602e-08 | 0.60746726 | 0.27497503 | 0 | 766 | — | 0 |
| diverted-single-null | 1000 | saddle | 4.0075182e-05 | 2.9080359e-05 | 1.4627887 | -1.0245964e-05 | 1.9280659e-07 | 0.42748394 | 0.5860806 | 2 | 26 | 26 | 0 |
| diverted-single-null | 2500 | axis | 3.7694359e-06 | 2.319785e-06 | -1.5368765 | -6.0130954e-06 | 4.3131666e-09 | 0.50404893 | 0.29198455 | 0 | 1949 | — | 0 |
| diverted-single-null | 2500 | saddle | 2.0692081e-05 | 1.1604981e-05 | 1.4944074 | -3.8753396e-06 | 6.0678808e-08 | 0.55310018 | 0.57382924 | 2 | 1479 | 1479 | 0 |
| weak-rotation-reactor-static | 300 | axis | 0.54309901 | 0.18602828 | -1.5707963 | -0.30413472 | 0.0033567373 | 0.90561601 | 0.43143593 | 0 | 204 | — | 0 |
| weak-rotation-reactor-static | 1000 | axis | 0.031524483 | 0.052586264 | -1.5707963 | -0.088023096 | 0.0005433577 | 0.18596008 | 0.26912769 | 0 | 227 | — | 0 |
| moderate-rotation-conventional-static | 300 | axis | 0.011410464 | 0.0048759721 | 1.5707963 | -0.0082425938 | 9.2988231e-05 | 0.7259156 | 0.37448759 | 0 | 205 | — | 0 |
| moderate-rotation-conventional-static | 1000 | axis | 0.001807394 | 0.0013735402 | 1.5707963 | -0.0023836668 | 1.5044836e-05 | 0.40818358 | 0.29870962 | 0 | 311 | — | 0 |
| strong-rotation-compact-static | 300 | axis | 0.0011655834 | 0.00034936934 | 1.5707963 | -0.0006316624 | 5.528012e-06 | 1.0349101 | 0.44949043 | 0 | 215 | — | 0 |
| strong-rotation-compact-static | 1000 | axis | 0.00044627225 | 0.00011449738 | -1.5707963 | -0.00019919175 | 9.2784389e-07 | 1.2090615 | 0.51153437 | 0 | 311 | — | 0 |

The extremal-centroid seed and definite-Hessian mode criterion agree on **15 of 15 rows**.

## H200 batch cost

| requested | realised | vertex_census_batch16_ms_per_state | production_batch16_ms_per_state |
|---:|---:|---:|---:|
| 500 | 550 | 1.96993 | 1.4 |
| 1000 | 1074 | 6.06689 | 4.9 |
| 2500 | 2616 | 32.4144 | 30.0 |

## Controls and figures

- Cyclic-origin control at the 300-requested rung: cyclic count changes after shifting all six values by one position = 0; open-chain changes = 256.
- Smooth perturbation amplitude is 1e-4 of the analytic axis-to-X span; saddle admission survived on 9 of 9 rungs.
- Vertex signs about the fitted stationary value produced four cyclic changes in the geometric analytic-X cell on 3 of 9 rungs.
- The retained candidate's source cell produced four stationary-level changes on 1 of 9 rungs. The quarter-pitch exterior tolerance can admit an indefinite fitted root whose hyperbola has only one branch pair crossing the sampled ring, so this sign count is corroborating mode evidence rather than an equivalent admission rule for tolerance-band roots.
- [saddle-error-comparison](/nova/figures/cut-cell-current-attribution/vertex-census/saddle-error-comparison.svg)
- [saddle-region-cells-110](/nova/figures/cut-cell-current-attribution/vertex-census/saddle-region-cells-110.svg)
- [saddle-region-cells-300](/nova/figures/cut-cell-current-attribution/vertex-census/saddle-region-cells-300.svg)
