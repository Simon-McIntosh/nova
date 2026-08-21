# Packed flux-kernel measurement

`measure.py` constructs the coarse recovery fixture's `G0` grid block from the
existing packed, array-namespace-threaded closed-form evaluator. The same JAX
graph runs in fp64 on CPU and H200 with a fixed 32×32 pair shape; the independent
NumPy lane rebuilds the uniform block through the current production function.

The committed `receipt.json` is the machine-readable result and `receipt.md` is
its compact review form. Large matrices and scheduler logs live under ignored
`work/` and `*.log` paths. The receipt records their hashes, numerical comparison,
compile and evaluation timing, backend identity, and the linear fine-fixture
projection. The production implementation is an imported reference and is not
changed by this measurement.

