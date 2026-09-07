# Zero-converged labeller block census

Generated at `2026-09-07T06:14:15.042341+00:00` from Nova `2c727480bb79`. The table contains every integer shot from 22475 through 22626, plus the nearest 20 complete converging corpus shots on each side. Missing or partial inputs remain explicit rows.

## Headline

The block has 70 complete zero-converged shots, 0 complete converging shots, and 82 rows with an unreadable or incomplete required input.

The readable block current range is -1–-0.09879 MA; the selected converging neighbours span 0.0884–0.9881 MA on the left and 0.1416–0.9237 MA on the right.

## Seed-degeneracy contrast

The boundary value is the writer topology's wall extremum from the spline-extrapolated limiter tail of `_slices_seed`; it bounds the map when it lies inside the seed grid's finite flux range. The axis candidate count is the finite O-point census produced by the same fixed-design topology grid at the first admitted row.

|cohort|shot|row|seed finite|boundary bounds map|axis candidates|stored axis on lattice|boundary Wb|grid Wb range|error|
|---|---:|---:|---|---|---:|---|---:|---|---|
|block|22475|1|True|True|4|True|-0.120219|-0.4248–-0.003142|—|
|block|22550|1|True|True|4|True|-0.116332|-0.3948–-0.003151|—|
|block|22626|1|True|True|4|True|-0.120885|-0.4295–-0.003154|—|
|neighbour|22235|1|True|True|3|True|0.369746|0.002977–0.4606|—|
|neighbour|22630|1|True|True|3|True|0.380679|0.003063–0.3858|—|
|neighbour|22634|1|True|True|3|True|0.379678|0.003045–0.385|—|

## Per-shot census

`first failure` is the earliest admitted manifest row that did not complete the writer pipeline, classified at the earliest stage explicitly evidenced by its exception and status fields. Excluded reconstruction rows are counted separately and do not mask the first admitted failure.

|cohort|shot|read|EFM rows|time s|Ip MA|axis z m|centroid z m|finite fcoil rows|admitted|excluded|converged|first failure|
|---|---:|---|---:|---|---|---|---|---|---:|---:|---:|---|
|left_neighbours|22115|ok|66|0–0.33|0.1844–0.9327|-0.02014–0.01396|-0.01977–0.02358|66/66|65|1|32|1:frame-assembly|
|left_neighbours|22116|ok|81|0–0.415|0.1839–0.957|-0.02579–0.0166|-0.02757–0.0204|81/81|79|2|48|1:frame-assembly|
|left_neighbours|22118|ok|69|0–0.355|0.181–0.9881|-0.02403–0.02112|-0.02358–0.02986|69/69|66|3|42|1:conditioned-nonconvergence|
|left_neighbours|22119|ok|66|0–0.335|0.183–0.9315|-0.01765–0.02145|-0.01705–0.03649|66/66|63|3|35|1:frame-assembly|
|left_neighbours|22120|ok|57|0–0.29|0.1843–0.9184|-0.04161–0.01697|-0.06185–0.02827|57/57|56|1|35|1:free-solve|
|left_neighbours|22121|ok|62|0–0.315|0.1832–0.9339|-0.02147–0.01903|-0.02262–0.03131|62/62|59|3|38|1:conditioned-nonconvergence|
|left_neighbours|22123|ok|73|0–0.375|0.1817–0.9401|-0.02094–0.1219|-0.02117–0.1391|73/73|69|4|51|1:conditioning-axis-admission|
|left_neighbours|22124|ok|66|0–0.335|0.1808–0.931|-0.01778–0.02377|-0.01783–0.03929|66/66|64|2|28|1:frame-assembly|
|left_neighbours|22125|ok|90|0–0.455|0.1831–0.9382|-0.02083–0.01163|-0.02072–0.01661|90/90|87|3|55|1:conditioned-nonconvergence|
|left_neighbours|22126|ok|86|0–0.435|0.1829–0.943|-0.02271–0.0186|-0.02297–0.03227|86/86|82|4|51|1:frame-assembly|
|left_neighbours|22127|ok|78|0–0.415|0.0884–0.9337|-0.02134–0.0641|-0.01842–0.06781|78/78|75|3|52|1:frame-assembly|
|left_neighbours|22168|ok|117|0–0.59|0.1722–0.4283|-0.1775–0.02177|-0.1845–0.02556|117/117|116|1|95|1:frame-assembly|
|left_neighbours|22170|ok|117|0–0.595|0.1775–0.4299|-0.01301–0.0381|-0.01367–0.03875|117/117|114|3|101|1:frame-assembly|
|left_neighbours|22171|ok|129|0–0.655|0.1768–0.4317|-0.01303–0.064|-0.015–0.06738|129/129|127|2|110|1:conditioning-axis-admission|
|left_neighbours|22174|ok|129|0–0.655|0.1872–0.433|-0.01797–0.1539|-0.01969–0.1614|129/129|127|2|111|1:frame-assembly|
|left_neighbours|22199|ok|113|0–0.575|0.1896–0.4295|-0.02271–0.01485|-0.02228–0.0164|113/113|109|4|96|1:free-axis-admission|
|left_neighbours|22206|ok|129|0–0.655|0.1927–0.4319|-0.02828–0.1586|-0.03358–0.167|129/129|127|2|113|1:frame-assembly|
|left_neighbours|22211|ok|113|0–0.575|0.1899–0.4293|-0.01778–0.0159|-0.01966–0.01858|113/113|110|3|92|1:free-axis-admission|
|left_neighbours|22231|ok|95|0–0.515|0.1561–0.7441|-0.01399–0.01725|-0.01418–0.02267|95/95|90|5|68|1:frame-assembly|
|left_neighbours|22235|ok|96|0–0.515|0.1607–0.7447|-0.01424–0.01455|-0.01448–0.0181|96/96|94|2|65|1:frame-assembly|
|block|22475|ok|110|0–0.555|-0.7383–-0.1632|-0.02096–0.05079|-0.02383–0.05233|110/110|106|4|0|1:free-axis-admission|
|block|22476|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22476.manifest.json'|109|0–0.555|-0.7374–-0.1474|-0.1168–0.09262|-0.1256–0.09334|109/109|—|—|—|unreadable|
|block|22477|ok|107|0–0.535|-0.7394–-0.1318|-0.07394–0.1204|-0.1117–0.1366|107/107|105|2|0|1:free-axis-admission|
|block|22478|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22478.manifest.json'|34|0–0.175|-0.7553–-0.1922|0.0009302–0.1871|0.002079–0.2161|34/34|—|—|—|unreadable|
|block|22479|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22479.manifest.json'|62|0–0.315|-0.8282–-0.1467|-0.01576–0.0283|-0.02345–0.04491|62/62|—|—|—|unreadable|
|block|22480|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22480.manifest.json'|62|0–0.315|-0.8295–-0.1491|-0.01863–0.03434|-0.02411–0.03241|62/62|—|—|—|unreadable|
|block|22481|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22481.manifest.json'|93|0–0.475|-0.8251–-0.1458|-0.01536–0.0249|-0.02113–0.02818|93/93|—|—|—|unreadable|
|block|22482|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22482.manifest.json'|62|0–0.31|-0.8262–-0.1428|-0.01668–0.01715|-0.02143–0.02736|62/62|—|—|—|unreadable|
|block|22483|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22483.manifest.json'|102|0–0.515|-0.8261–-0.1483|-0.0146–0.02091|-0.02025–0.02991|102/102|—|—|—|unreadable|
|block|22484|ok|118|0–0.595|-0.4173–-0.1861|-0.01137–0.2601|-0.01208–0.2612|118/118|113|5|0|1:free-axis-admission|
|block|22485|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22485.manifest.json'|114|0–0.575|-0.4143–-0.1889|-0.01361–0.02747|-0.01717–0.02778|114/114|—|—|—|unreadable|
|block|22486|ok|125|0–0.655|-0.4152–-0.1867|-0.01404–0.02013|-0.01686–0.02081|125/125|119|6|0|1:free-axis-admission|
|block|22487|ok|109|0–0.555|-0.4154–-0.1915|-0.01144–0.01459|-0.01589–0.01526|109/109|105|4|0|1:free-axis-admission|
|block|22488|ok|126|0–0.655|-0.4161–-0.1904|-0.01449–0.02322|-0.01706–0.02413|126/126|121|5|0|1:free-axis-admission|
|block|22489|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22489.manifest.json'|124|0–0.655|-0.4142–-0.1944|-0.01338–0.01577|-0.01623–0.01638|124/124|—|—|—|unreadable|
|block|22490|ok|113|0–0.575|-0.4128–-0.1913|-0.01081–0.01517|-0.01459–0.01579|113/113|110|3|0|1:free-axis-admission|
|block|22491|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22491.manifest.json'|132|0–0.675|-0.4158–-0.1964|-0.01452–0.04087|-0.01602–0.04209|132/132|—|—|—|unreadable|
|block|22492|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22492.manifest.json'|124|0–0.655|-0.4144–-0.1264|-0.01055–0.01681|-0.01263–0.01741|124/124|—|—|—|unreadable|
|block|22493|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22493.manifest.json'|89|0–0.455|-0.4157–-0.2034|-0.01036–0.01345|-0.01479–0.0146|89/89|—|—|—|unreadable|
|block|22494|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22494.manifest.json'|117|0–0.595|-0.4155–-0.2034|-0.01276–0.0157|-0.01586–0.01618|117/117|—|—|—|unreadable|
|block|22495|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22495.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22496|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22496.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22497|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22497.manifest.json'|95|0–0.475|-0.6177–-0.1316|-0.01044–0.1226|-0.0128–0.1323|95/95|—|—|—|unreadable|
|block|22498|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22498.manifest.json'|94|0–0.475|-0.7399–-0.1658|-0.01263–0.1951|-0.01726–0.1988|94/94|—|—|—|unreadable|
|block|22499|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22499.manifest.json'; efm unreadable: FileNotFoundError: /work/projects/imas_gpu/mast/level1/shots/22499.zarr does not exist|—|—|—|—|—|—|—|—|—|unreadable|
|block|22500|ok|86|0–0.43|-0.7437–-0.161|-0.01464–0.02593|-0.01851–0.03323|86/86|85|1|0|1:free-axis-admission|
|block|22501|ok|86|0–0.435|-0.7511–-0.1634|-0.01123–0.02053|-0.01614–0.03218|86/86|82|4|0|1:free-axis-admission|
|block|22502|ok|97|0–0.495|-0.7527–-0.117|-0.0411–0.03519|-0.04032–0.04163|97/97|91|6|0|1:free-axis-admission|
|block|22503|ok|94|0–0.47|-0.7511–-0.1617|-0.009418–0.02639|-0.01595–0.03295|94/94|93|1|0|1:free-axis-admission|
|block|22504|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22504.manifest.json'|66|0–0.33|-0.7518–-0.1641|-0.01138–0.02763|-0.01632–0.03201|66/66|—|—|—|unreadable|
|block|22505|ok|106|0–0.535|-0.7525–-0.1624|-0.01262–0.02547|-0.01589–0.03267|106/106|103|3|0|1:free-axis-admission|
|block|22506|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22506.manifest.json'|85|0–0.435|-0.754–-0.1612|-0.01097–0.02448|-0.01596–0.03427|85/85|—|—|—|unreadable|
|block|22507|ok|90|0–0.455|-0.7498–-0.1202|-0.0755–0.1726|-0.08875–0.1919|90/90|88|2|0|1:free-axis-admission|
|block|22508|ok|102|0–0.515|-0.7534–-0.1608|-0.01402–0.0252|-0.0214–0.03133|102/102|98|4|0|1:free-axis-admission|
|block|22509|ok|107|0–0.535|-0.7537–-0.1593|-0.01045–0.02914|-0.01608–0.03273|107/107|102|5|0|1:free-axis-admission|
|block|22510|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22510.manifest.json'|58|0–0.295|-0.7522–-0.1625|-0.00916–0.02428|-0.01679–0.03254|58/58|—|—|—|unreadable|
|block|22511|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22511.manifest.json'|54|0–0.275|-0.7527–-0.1619|-0.01483–0.02466|-0.01889–0.03032|54/54|—|—|—|unreadable|
|block|22512|ok|66|0–0.335|-0.7562–-0.1633|-0.01068–0.0257|-0.01593–0.03254|66/66|63|3|0|1:free-axis-admission|
|block|22513|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22513.manifest.json'|55|0–0.275|-0.7272–-0.1628|-0.01736–0.06436|-0.02171–0.07084|55/55|—|—|—|unreadable|
|block|22514|ok|70|0–0.355|-0.7548–-0.1613|-0.01593–0.05152|-0.01347–0.05631|70/70|68|2|0|1:free-axis-admission|
|block|22515|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22515.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22516|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22516.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22517|ok|94|0–0.475|-0.6213–-0.1094|-0.05516–0.07029|-0.05991–0.07972|94/94|90|4|0|1:free-axis-admission|
|block|22518|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22518.manifest.json'|54|0–0.275|-0.7688–-0.1624|-0.009468–0.02599|-0.01454–0.03117|54/54|—|—|—|unreadable|
|block|22519|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22519.manifest.json'; efm unreadable: FileNotFoundError: /work/projects/imas_gpu/mast/level1/shots/22519.zarr does not exist|—|—|—|—|—|—|—|—|—|unreadable|
|block|22520|ok|66|0–0.335|-0.8331–-0.1643|-0.01476–0.07271|-0.01906–0.08522|66/66|61|5|0|1:free-axis-admission|
|block|22521|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22521.manifest.json'|54|0–0.275|-0.7534–-0.161|-0.01154–0.0301|-0.01511–0.03385|54/54|—|—|—|unreadable|
|block|22522|ok|58|0–0.29|-0.7715–-0.1622|-0.01974–0.03115|-0.01731–0.03491|58/58|57|1|0|1:free-axis-admission|
|block|22523|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22523.manifest.json'|54|0–0.275|-0.7534–-0.1623|-0.00992–0.02417|-0.0149–0.03109|54/54|—|—|—|unreadable|
|block|22524|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22524.manifest.json'|66|0–0.335|-0.719–-0.165|-0.0405–0.0313|-0.04592–0.03622|66/66|—|—|—|unreadable|
|block|22525|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22525.manifest.json'|54|0–0.275|-0.8467–-0.1615|-0.01034–0.0332|-0.01659–0.03845|54/54|—|—|—|unreadable|
|block|22526|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22526.manifest.json'|86|0–0.435|-0.8467–-0.1416|-0.01494–0.02497|-0.01643–0.02864|86/86|—|—|—|unreadable|
|block|22527|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22527.manifest.json'|90|0–0.455|-0.8489–-0.1633|-0.01076–0.0296|-0.01636–0.03392|90/90|—|—|—|unreadable|
|block|22528|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22528.manifest.json'|86|0–0.43|-0.8516–-0.161|-0.008221–0.02425|-0.01387–0.0308|86/86|—|—|—|unreadable|
|block|22529|ok|87|0–0.435|-0.6943–-0.1607|-0.09214–0.0284|-0.1166–0.03257|87/87|84|3|0|1:free-axis-admission|
|block|22530|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22530.manifest.json'|67|0–0.335|-0.7638–-0.1625|-0.01191–0.1342|-0.01592–0.1454|67/67|—|—|—|unreadable|
|block|22531|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22531.manifest.json'|54|0–0.295|-0.7648–-0.128|-0.01161–0.0125|-0.01553–0.01415|54/54|—|—|—|unreadable|
|block|22532|ok|71|0–0.375|-0.8023–-0.1284|-0.01153–0.02584|-0.01468–0.04828|71/71|70|1|0|1:free-axis-admission|
|block|22533|ok|84|0–0.435|-0.7292–-0.1623|-0.02136–0.04594|-0.02035–0.04787|84/84|80|4|0|1:free-axis-admission|
|block|22534|ok|103|0–0.52|-0.7406–-0.1602|-0.07655–0.06352|-0.0972–0.0638|103/103|102|1|0|1:free-axis-admission|
|block|22535|ok|82|0–0.41|-0.7372–-0.1597|-0.002946–0.1501|-0.00536–0.1748|82/82|81|1|0|1:free-axis-admission|
|block|22536|ok|86|0–0.435|-0.7266–-0.142|-0.09941–0.01858|-0.1294–0.02205|86/86|83|3|0|1:free-axis-admission|
|block|22537|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22537.manifest.json'; efm unreadable: FileNotFoundError: /work/projects/imas_gpu/mast/level1/shots/22537.zarr does not exist|—|—|—|—|—|—|—|—|—|unreadable|
|block|22538|ok|86|0–0.435|-0.7834–-0.1522|-0.01044–0.06585|-0.01208–0.08687|86/86|82|4|0|1:free-axis-admission|
|block|22539|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22539.manifest.json'; efm unreadable: FileNotFoundError: /work/projects/imas_gpu/mast/level1/shots/22539.zarr does not exist|—|—|—|—|—|—|—|—|—|unreadable|
|block|22540|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22540.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22541|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22541.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22542|ok|86|0–0.43|-0.6229–-0.194|-0.0121–0.02373|-0.01742–0.02558|86/86|85|1|0|1:free-axis-admission|
|block|22543|ok|94|0–0.475|-0.8299–-0.1377|-0.01714–0.03956|-0.01973–0.05165|94/94|91|3|0|1:free-axis-admission|
|block|22544|ok|98|0–0.495|-0.8217–-0.1321|-0.02149–0.02597|-0.02139–0.03316|98/98|95|3|0|1:free-axis-admission|
|block|22545|ok|86|0–0.435|-0.8141–-0.1327|-0.1503–0.02255|-0.1981–0.02952|86/86|84|2|0|1:free-axis-admission|
|block|22546|ok|94|0–0.475|-0.8138–-0.1346|-0.01728–0.01925|-0.01847–0.02551|94/94|91|3|0|1:free-axis-admission|
|block|22547|ok|98|0–0.495|-0.8227–-0.1314|-0.0195–0.02015|-0.021–0.03068|98/98|95|3|0|1:free-axis-admission|
|block|22548|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22548.manifest.json'|82|0–0.41|-0.8099–-0.1268|-0.1529–0.02143|-0.2119–0.0236|82/82|—|—|—|unreadable|
|block|22549|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22549.manifest.json'|66|0–0.335|-0.8062–-0.1297|-0.01767–0.01954|-0.02215–0.02487|66/66|—|—|—|unreadable|
|block|22550|ok|70|0–0.355|-0.8018–-0.1321|-0.01965–0.01886|-0.0224–0.02172|70/70|68|2|0|1:free-axis-admission|
|block|22551|ok|94|0–0.475|-0.8219–-0.1353|-0.01704–0.02033|-0.01973–0.02701|94/94|90|4|0|1:free-axis-admission|
|block|22552|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22552.manifest.json'|98|0–0.495|-0.8219–-0.1269|-0.01736–0.02052|-0.02063–0.02722|98/98|—|—|—|unreadable|
|block|22553|ok|99|0–0.495|-0.8221–-0.1304|-0.01583–0.06599|-0.0183–0.08541|99/99|97|2|0|1:free-axis-admission|
|block|22554|ok|98|0–0.495|-0.8223–-0.1262|-0.0177–0.0206|-0.01691–0.02598|98/98|93|5|0|1:free-axis-admission|
|block|22555|ok|87|0–0.435|-0.8056–-0.129|-0.01951–0.02149|-0.022–0.02757|87/87|83|4|0|1:free-axis-admission|
|block|22556|ok|79|0–0.395|-0.8416–-0.1188|-0.1932–0.02029|-0.2719–0.02313|79/79|75|4|0|1:free-axis-admission|
|block|22557|ok|62|0–0.315|-0.8316–-0.1361|-0.01725–0.01928|-0.02131–0.02844|62/62|58|4|0|1:free-axis-admission|
|block|22558|ok|62|0–0.315|-0.8497–-0.1405|-0.01843–0.02442|-0.02119–0.03996|62/62|57|5|0|1:free-axis-admission|
|block|22559|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22559.manifest.json'|58|0–0.29|-0.838–-0.1391|-0.04848–0.02205|-0.06797–0.01528|58/58|—|—|—|unreadable|
|block|22560|ok|54|0–0.295|-0.8399–-0.1433|-0.0504–0.01827|-0.07966–0.01501|54/54|52|2|0|1:free-axis-admission|
|block|22561|ok|58|0–0.295|-0.8348–-0.1403|-0.04769–0.01777|-0.09199–0.01915|58/58|56|2|0|1:free-axis-admission|
|block|22562|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22562.manifest.json'|58|0–0.29|-0.8325–-0.1406|-0.04794–0.02664|-0.09267–0.009299|58/58|—|—|—|unreadable|
|block|22563|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22563.manifest.json'|42|0–0.215|-0.7611–-0.1449|-0.1142–0.02041|-0.1409–0.01186|42/42|—|—|—|unreadable|
|block|22564|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22564.manifest.json'|46|0–0.235|-0.8371–-0.1448|-0.1708–0.02541|-0.2182–0.007512|46/46|—|—|—|unreadable|
|block|22565|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22565.manifest.json'|46|0–0.235|-0.8753–-0.1401|-0.05704–0.0177|-0.08673–0.02359|46/46|—|—|—|unreadable|
|block|22566|ok|58|0–0.29|-0.8285–-0.1428|-0.04553–0.02177|-0.08156–0.02794|58/58|57|1|0|1:free-axis-admission|
|block|22567|ok|62|0–0.315|-0.8317–-0.1362|-0.04647–0.02416|-0.09055–0.02948|62/62|60|2|0|1:free-axis-admission|
|block|22568|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22568.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22569|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22569.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22570|ok|95|0–0.475|-0.6182–-0.1034|-0.07514–0.08725|-0.06938–0.09668|95/95|91|4|0|1:free-axis-admission|
|block|22571|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22571.manifest.json'|108|0–0.555|-0.7364–-0.1577|-0.009605–0.0338|-0.01382–0.03392|108/108|—|—|—|unreadable|
|block|22572|ok|68|0–0.34|-0.7544–-0.1591|-0.1496–0.07631|-0.1536–0.0766|68/68|67|1|0|1:free-axis-admission|
|block|22573|ok|86|0–0.435|-0.7399–-0.16|-0.01311–0.02425|-0.01751–0.02894|86/86|83|3|0|1:free-axis-admission|
|block|22574|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22574.manifest.json'|65|0–0.335|-0.7389–-0.1569|-0.009452–0.02746|-0.01477–0.03341|65/65|—|—|—|unreadable|
|block|22575|ok|87|0–0.435|-0.8438–-0.1571|-0.202–0.02305|-0.2552–0.03295|87/87|84|3|0|1:free-axis-admission|
|block|22576|ok|86|0–0.435|-0.7408–-0.1594|-0.01101–0.02484|-0.01576–0.03149|86/86|83|3|0|1:free-axis-admission|
|block|22577|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22577.manifest.json'|66|0–0.335|-0.757–-0.1584|-0.01104–0.02301|-0.01651–0.02944|66/66|—|—|—|unreadable|
|block|22578|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22578.manifest.json'|66|0–0.335|-0.7426–-0.1595|-0.0113–0.02711|-0.01496–0.03343|66/66|—|—|—|unreadable|
|block|22579|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22579.manifest.json'|78|0–0.395|-0.8442–-0.1608|-0.01253–0.02718|-0.01821–0.0309|78/78|—|—|—|unreadable|
|block|22580|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22580.manifest.json'|86|0–0.435|-0.8472–-0.1605|-0.01098–0.02635|-0.01599–0.03658|86/86|—|—|—|unreadable|
|block|22581|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22581.manifest.json'|82|0–0.415|-0.8466–-0.1586|-0.01218–0.02281|-0.01748–0.03244|82/82|—|—|—|unreadable|
|block|22582|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22582.manifest.json'|50|0–0.25|-0.7886–-0.1881|0.001042–0.301|0.001969–0.3226|50/50|—|—|—|unreadable|
|block|22583|ok|79|0–0.395|-0.8082–-0.1566|-0.002385–0.1696|-0.00717–0.1507|79/79|78|1|0|1:free-axis-admission|
|block|22584|ok|78|0–0.395|-0.735–-0.1576|0.001949–0.1374|-0.002137–0.1792|78/78|75|3|0|1:free-axis-admission|
|block|22585|ok|78|0–0.395|-0.7334–-0.1578|-0.001931–0.08561|-0.00517–0.1138|78/78|76|2|0|1:free-axis-admission|
|block|22586|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22586.manifest.json'; efm unreadable: FileNotFoundError: /work/projects/imas_gpu/mast/level1/shots/22586.zarr does not exist|—|—|—|—|—|—|—|—|—|unreadable|
|block|22587|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22587.manifest.json'|78|0–0.39|-0.8415–-0.1608|-0.007194–0.02186|-0.0123–0.03013|78/78|—|—|—|unreadable|
|block|22588|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22588.manifest.json'|82|0–0.415|-0.7376–-0.1551|-0.009342–0.02007|-0.01355–0.03257|82/82|—|—|—|unreadable|
|block|22589|ok|109|0–0.555|-0.7298–-0.1542|-0.009203–0.05844|-0.01325–0.058|109/109|107|2|0|1:free-axis-admission|
|block|22590|ok|73|0–0.395|-0.7286–-0.1544|-0.01092–0.1038|-0.01414–0.1103|73/73|70|3|0|1:free-axis-admission|
|block|22591|ok|79|0–0.395|-0.7351–-0.1481|0.001455–0.06187|0.0008149–0.07709|79/79|77|2|0|1:free-axis-admission|
|block|22592|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22592.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22593|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22593.manifest.json'; efm unreadable: FileNotFoundError: /work/projects/imas_gpu/mast/level1/shots/22593.zarr does not exist|—|—|—|—|—|—|—|—|—|unreadable|
|block|22594|ok|94|0–0.475|-0.8131–-0.1425|-0.045–0.01435|-0.0782–0.00917|94/94|92|2|0|1:free-axis-admission|
|block|22595|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22595.manifest.json'|58|0–0.29|-0.8292–-0.1331|-0.04378–0.01466|-0.07791–0.01685|58/58|—|—|—|unreadable|
|block|22596|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22596.manifest.json'|62|0–0.31|-0.8249–-0.1384|-0.04415–0.012|-0.06428–0.01154|62/62|—|—|—|unreadable|
|block|22597|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22597.manifest.json'|41|0–0.215|-0.8147–-0.1821|-0.05096–-0.01333|-0.07288–-0.01846|41/41|—|—|—|unreadable|
|block|22598|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22598.manifest.json'|49|0–0.255|-0.8343–-0.2067|-0.05237–-0.001767|-0.08967–-0.02108|49/49|—|—|—|unreadable|
|block|22599|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22599.manifest.json'|54|0–0.275|-0.8349–-0.1472|-0.1759–0.01211|-0.1998–0.01524|54/54|—|—|—|unreadable|
|block|22600|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22600.manifest.json'|34|0–0.175|-0.8456–-0.1523|-0.1336–0.01304|-0.1672–0.012|34/34|—|—|—|unreadable|
|block|22601|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22601.manifest.json'|50|0–0.25|-0.8363–-0.1537|-0.1994–0.009319|-0.224–0.009307|50/50|—|—|—|unreadable|
|block|22602|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22602.manifest.json'|62|0–0.315|-0.8201–-0.1495|-0.04112–0.01236|-0.06118–0.01665|62/62|—|—|—|unreadable|
|block|22603|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22603.manifest.json'|56|0–0.295|-0.8317–-0.1533|-0.1395–0.08034|-0.1587–0.09643|56/56|—|—|—|unreadable|
|block|22604|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22604.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22605|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22605.manifest.json'; efm unreadable: KeyError: "'plasma_current_c' not found in consolidated metadata."|—|—|—|—|—|—|—|—|—|unreadable|
|block|22606|ok|119|0–0.595|-0.4146–-0.1037|-0.0674–0.01632|-0.06339–0.01688|119/119|114|5|0|1:free-axis-admission|
|block|22607|ok|106|0–0.535|-0.4111–-0.1822|-0.01115–0.03693|-0.01365–0.03729|106/106|105|1|0|1:free-axis-admission|
|block|22608|ok|106|0–0.535|-0.4123–-0.1801|-0.01076–0.0368|-0.01204–0.03701|106/106|102|4|0|1:free-axis-admission|
|block|22609|ok|105|0–0.535|-0.4104–-0.184|-0.00981–0.01837|-0.01182–0.01875|105/105|101|4|0|1:free-axis-admission|
|block|22610|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22610.manifest.json'|102|0–0.515|-0.4103–-0.1841|-0.01244–0.02152|-0.01452–0.02231|102/102|—|—|—|unreadable|
|block|22611|ok|103|0–0.515|-0.4096–-0.1854|-0.01415–0.02165|-0.01588–0.02237|103/103|99|4|0|1:free-axis-admission|
|block|22612|ok|86|0–0.455|-0.7368–-0.1508|-0.0009044–0.09034|-0.003677–0.1048|86/86|83|3|0|1:free-axis-admission|
|block|22613|ok|82|0–0.41|-0.7366–-0.1441|0.002133–0.07747|0.004111–0.1043|82/82|81|1|0|1:free-axis-admission|
|block|22614|ok|83|0–0.415|-0.7366–-0.1498|0.001904–0.09965|0.001794–0.131|83/83|81|2|0|1:free-axis-admission|
|block|22615|ok|75|0–0.395|-0.7306–-0.1487|-0.01291–0.03563|-0.01569–0.03634|75/75|72|3|0|1:free-axis-admission|
|block|22616|ok|66|0–0.335|-0.7191–-0.1465|0.004212–0.05445|0.002635–0.06907|66/66|62|4|0|1:free-axis-admission|
|block|22617|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22617.manifest.json'|70|0–0.355|-0.7303–-0.1469|0.002425–0.05707|0.001313–0.07084|70/70|—|—|—|unreadable|
|block|22618|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22618.manifest.json'|58|0–0.295|-0.8615–-0.1374|-0.04407–0.01229|-0.06461–0.01338|58/58|—|—|—|unreadable|
|block|22619|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22619.manifest.json'|79|0–0.395|-1–-0.1187|-0.1798–0.0187|-0.2226–0.0228|79/79|—|—|—|unreadable|
|block|22620|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22620.manifest.json'|50|0–0.255|-0.9438–-0.1192|-0.1934–0.08394|-0.2311–0.1032|50/50|—|—|—|unreadable|
|block|22621|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22621.manifest.json'|78|0–0.395|-0.7571–-0.1047|-0.06013–0.02521|-0.07835–0.02941|78/78|—|—|—|unreadable|
|block|22622|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22622.manifest.json'|70|0–0.35|-0.8209–-0.09886|-0.04213–0.02977|-0.06713–0.03439|70/70|—|—|—|unreadable|
|block|22623|manifest unreadable: FileNotFoundError: [Errno 2] No such file or directory: '/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/22623.manifest.json'|118|0–0.595|-0.4117–-0.1048|-0.02753–0.01641|-0.02698–0.01693|118/118|—|—|—|unreadable|
|block|22624|ok|109|0–0.55|-0.4107–-0.1543|-0.01279–0.01003|-0.01536–0.01026|109/109|108|1|0|1:free-axis-admission|
|block|22625|ok|82|0–0.415|-0.8195–-0.09879|-0.07604–0.02412|-0.108–0.02485|82/82|80|2|0|1:free-axis-admission|
|block|22626|ok|95|0–0.475|-0.7265–-0.1194|-0.00996–0.05266|-0.01433–0.06192|95/95|91|4|0|1:free-axis-admission|
|right_neighbours|22630|ok|61|0–0.305|0.1507–0.7558|-0.05378–0.02785|-0.07073–0.04343|61/61|60|1|42|1:frame-assembly|
|right_neighbours|22634|ok|62|0–0.31|0.1548–0.8101|-0.05964–0.09965|-0.02609–0.1459|62/62|61|1|44|1:frame-assembly|
|right_neighbours|22635|ok|54|0–0.275|0.1416–0.7833|-0.01716–0.0327|-0.02391–0.04939|54/54|51|3|33|1:frame-assembly|
|right_neighbours|22646|ok|86|0–0.43|0.2031–0.6262|-0.07003–0.01601|-0.05189–0.01797|86/86|85|1|72|1:frame-assembly|
|right_neighbours|22647|ok|65|0–0.335|0.2263–0.6865|-0.02962–0.01496|-0.02327–0.01719|65/65|63|2|58|1:frame-assembly|
|right_neighbours|22648|ok|57|0–0.295|0.2493–0.6863|-0.08819–0.01454|-0.0253–0.01738|57/57|55|2|47|1:frame-assembly|
|right_neighbours|22649|ok|97|0–0.495|0.2539–0.6877|-0.01792–0.009796|-0.02507–0.01139|97/97|95|2|77|1:conditioned-nonconvergence|
|right_neighbours|22650|ok|61|0–0.315|0.2454–0.707|-0.09772–0.1395|-0.1108–0.1719|61/61|59|2|51|1:conditioning-axis-admission|
|right_neighbours|22651|ok|76|0–0.395|0.2485–0.6934|-0.04103–0.03023|-0.02073–0.008993|76/76|73|3|68|1:frame-assembly|
|right_neighbours|22652|ok|65|0–0.335|0.2531–0.6986|-0.08107–0.01518|-0.02897–0.01754|65/65|61|4|53|1:frame-assembly|
|right_neighbours|22654|ok|65|0–0.335|0.2423–0.6899|-0.03095–0.01869|-0.02246–0.01085|65/65|61|4|53|1:frame-assembly|
|right_neighbours|22655|ok|61|0–0.315|0.2564–0.686|-0.02788–0.006871|-0.02226–0.008288|61/61|59|2|55|1:frame-assembly|
|right_neighbours|22656|ok|65|0–0.335|0.2529–0.6971|-0.01673–0.006871|-0.02476–0.00781|65/65|61|4|54|1:qualification|
|right_neighbours|22657|ok|70|0–0.355|0.1848–0.9237|-0.01157–0.008409|-0.01078–0.01433|70/70|66|4|26|1:frame-assembly|
|right_neighbours|22659|ok|62|0–0.315|0.1845–0.897|-0.01912–0.009095|-0.01909–0.01404|62/62|57|5|38|1:frame-assembly|
|right_neighbours|22660|ok|70|0–0.355|0.1836–0.8801|-0.01441–0.006407|-0.01319–0.008292|70/70|66|4|33|1:frame-assembly|
|right_neighbours|22662|ok|78|0–0.395|0.182–0.9063|-0.01413–0.009708|-0.01335–0.01403|78/78|74|4|41|1:frame-assembly|
|right_neighbours|22664|ok|74|0–0.375|0.1806–0.9044|-0.02316–0.0108|-0.02408–0.01467|74/74|70|4|43|1:frame-assembly|
|right_neighbours|22671|ok|90|0–0.45|0.1501–0.6262|-0.1573–0.1782|-0.1565–0.1813|90/90|89|1|76|1:frame-assembly|
|right_neighbours|22674|ok|67|0–0.335|0.1698–0.72|-0.01153–0.2128|-0.01216–0.3117|67/67|63|4|43|1:frame-assembly|

## First-failure census

```json
{
  "block": {
    "free-axis-admission": 70,
    "unreadable": 82
  },
  "left_neighbours": {
    "conditioned-nonconvergence": 3,
    "conditioning-axis-admission": 2,
    "frame-assembly": 12,
    "free-axis-admission": 2,
    "free-solve": 1
  },
  "right_neighbours": {
    "conditioned-nonconvergence": 1,
    "conditioning-axis-admission": 1,
    "frame-assembly": 17,
    "qualification": 1
  }
}
```

## Recommendation

**Exclude the 70 complete zero-converged block shots from the demo cohort; the six-shot seed contrast does not isolate a block-specific repairable seed defect.** Seed degeneracy appears in 0 of 3 block samples and 0 of 3 neighbour samples. The readable block EFM current is entirely negative (-1–-0.09879 MA), while the converging neighbours are entirely positive (0.0884–0.9881 MA).
