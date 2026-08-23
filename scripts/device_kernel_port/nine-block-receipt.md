# Nine-block exact-section device receipt

Biot selection: 2,618 selected, zero failures (2,612 passed, 3 skipped, 3 xfailed).

CPU cold build: 58.452205 s; H200 cold build: 26.576202 s (120 s bar). CPU/H200 kernel walls were 39.187988/0.391012 s. The banked NumPy three-flux-order grid stage was 76.676843 s and the complete exact kernel-family profile was 286.023 s.

GPU-vs-CPU ULP: p50=4763908, p90=192114811, p99=478712534412, p99.9=8617350547552581613, max=8964084725603857419; byte-identical 2.204574791%. The original baseline was 1.434% byte-identical with p99.9 1.875e12 ULP.

Column 184 against the 1024-rung oracle: CPU/GPU median absolute 4.8956961411783714e-10/5.1345238679920789e-10; target 525 absolute CPU-XLA/H200/scalar-NumPy 1.6082512401776555e-09/1.2244882396135323e-09/1.3472923997940644e-09. The measured unpaired-path bound is therefore 1.6082512401776555e-09; the banked paired-path value was 1.49e-09. Under the locked absolute-error disposition these values are receipt evidence, not a request for another pairing layer. The single-source traced path is the production route.
